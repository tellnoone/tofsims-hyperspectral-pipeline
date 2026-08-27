"""
decomposition.py
Dimensionality reduction of the hyperspectral cube.

Three complementary views of the same pixels:
  * PCA on standardized intensities  - variance structure, signed loadings.
  * NMF on raw counts                - additive, non-negative chemical endmembers.
  * UMAP on TIC-normalized spectra   - non-linear neighbourhood structure.

Each operates on a shared preprocessing of the cube (see `prepare_pixels`) so the
three results are directly comparable pixel-for-pixel.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler

from src.utils.constants import FOREGROUND_MASK_FILE, NMF_FILE, PCA_FILE, UMAP_FILE


@dataclass
class PreparedPixels:
    """The cube flattened to pixel vectors in the three representations used below."""

    X_raw: np.ndarray       # (n_pixels, n_channels) raw counts, non-negative
    X_std: np.ndarray       # standardized, for PCA
    X_tic: np.ndarray       # total-ion-count normalized, for UMAP
    fg_mask: np.ndarray     # (n_pixels,) bool - pixels with any signal
    fg_idx: np.ndarray      # flat indices of foreground pixels
    height: int
    width: int
    n_channels: int

    @property
    def n_foreground(self) -> int:
        return int(self.fg_mask.sum())


def prepare_pixels(cube: np.ndarray, eps: float = 1e-8) -> PreparedPixels:
    """Flatten an (H, W, C) cube into the pixel representations each method needs."""
    height, width, n_channels = cube.shape
    X_raw = cube.reshape(-1, n_channels).astype(np.float64)

    X_std = StandardScaler().fit_transform(X_raw)

    tic = X_raw.sum(axis=1, keepdims=True)
    tic[tic == 0] = eps
    X_tic = X_raw / tic

    fg_mask = X_raw.sum(axis=1) > 0
    fg_idx = np.where(fg_mask)[0]

    return PreparedPixels(
        X_raw=X_raw,
        X_std=X_std,
        X_tic=X_tic,
        fg_mask=fg_mask,
        fg_idx=fg_idx,
        height=height,
        width=width,
        n_channels=n_channels,
    )


# ------------------------------------------------------------------------- PCA
def run_pca(
    prepared: PreparedPixels,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit PCA on standardized pixel vectors. `None` keeps every component."""
    n_components = n_components or prepared.n_channels
    n_components = min(n_components, prepared.n_channels)

    model = PCA(n_components=n_components, random_state=random_state)
    scores = model.fit_transform(prepared.X_std)

    return {
        "scores": scores,
        "score_maps": scores.reshape(prepared.height, prepared.width, n_components),
        "loadings": model.components_,
        "explained_variance": model.explained_variance_ratio_,
        "n_components": n_components,
        "model": model,
    }


# ------------------------------------------------------------------------- NMF
def run_nmf_sweep(
    prepared: PreparedPixels,
    sweep: Sequence[int],
    max_iter: int = 1000,
    init: str = "nndsvda",
    random_state: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """Fit NMF at several ranks so the rank can be chosen from the error curve."""
    results: Dict[int, Dict[str, Any]] = {}
    for k in sweep:
        if k < 1 or k > prepared.n_channels:
            continue
        model = NMF(n_components=k, max_iter=max_iter, random_state=random_state, init=init)
        W = model.fit_transform(prepared.X_raw)
        results[int(k)] = {
            "W": W,
            "H": model.components_,
            "error": float(model.reconstruction_err_),
        }
    if not results:
        raise ValueError(
            f"NMF sweep {list(sweep)} contains no rank valid for "
            f"{prepared.n_channels} channels."
        )
    return results


def select_nmf_rank(sweep_results: Dict[int, Dict[str, Any]]) -> int:
    """
    Pick the NMF rank at the elbow of the reconstruction-error curve.

    The elbow is the rank where the error stops falling steeply — the largest
    second difference of the error curve. This replaces the previous hand-picked
    constant, which was read off a plot for one specific dataset.
    """
    ranks = sorted(sweep_results)
    if len(ranks) < 3:
        return ranks[0]

    errors = [sweep_results[k]["error"] for k in ranks]
    curvature = [
        (errors[i - 1] - errors[i]) - (errors[i] - errors[i + 1])
        for i in range(1, len(ranks) - 1)
    ]
    return ranks[1 + int(np.argmax(curvature))]


def run_nmf(
    prepared: PreparedPixels,
    n_components: int,
    max_iter: int = 1000,
    init: str = "nndsvda",
    random_state: int = 42,
    clip_percentile: float = 99.5,
    sweep_results: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Produce endmember spectra and abundance maps at the chosen rank.

    Reuses the fit from the sweep when one is available rather than refitting.
    """
    if sweep_results and n_components in sweep_results:
        W, H = sweep_results[n_components]["W"], sweep_results[n_components]["H"]
        error = sweep_results[n_components]["error"]
    else:
        model = NMF(
            n_components=n_components, max_iter=max_iter, random_state=random_state, init=init
        )
        W = model.fit_transform(prepared.X_raw)
        H = model.components_
        error = float(model.reconstruction_err_)

    abundance_maps = W.reshape(prepared.height, prepared.width, n_components)

    # Percentile-normalized copy for display; the raw maps stay untouched for analysis.
    abundance_maps_norm = np.zeros_like(abundance_maps)
    for i in range(n_components):
        ceiling = np.percentile(abundance_maps[:, :, i], clip_percentile)
        abundance_maps_norm[:, :, i] = np.clip(abundance_maps[:, :, i] / (ceiling + 1e-8), 0, 1)

    return {
        "W": W,
        "H": H,
        "abundance_maps": abundance_maps,
        "abundance_maps_norm": abundance_maps_norm,
        "reconstruction_error": error,
        "n_components": n_components,
    }


# ------------------------------------------------------------------------ UMAP
def run_umap(
    prepared: PreparedPixels,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Embed foreground pixels with UMAP.

    Only foreground pixels are embedded — background is a single degenerate point
    that would otherwise dominate the neighbourhood graph.
    """
    import umap  # imported lazily; it is slow to load and optional

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(prepared.n_foreground - 1, 2)),
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    embedding = reducer.fit_transform(prepared.X_tic[prepared.fg_mask])

    # Scatter the embedding back onto the image grid, leaving background as NaN.
    coordinate_maps = []
    for dim in range(n_components):
        flat = np.full(prepared.height * prepared.width, np.nan)
        flat[prepared.fg_idx] = embedding[:, dim]
        coordinate_maps.append(flat.reshape(prepared.height, prepared.width))

    return {
        "embedding": embedding,
        "coordinate_maps": np.stack(coordinate_maps, axis=-1),
        "umap_x_img": coordinate_maps[0],
        "umap_y_img": coordinate_maps[1] if n_components > 1 else coordinate_maps[0],
    }


# ------------------------------------------------------------------ persistence
def save_decomposition(
    processed_dir: Path,
    prepared: PreparedPixels,
    pca_result: Dict[str, Any],
    nmf_result: Dict[str, Any],
    umap_result: Optional[Dict[str, Any]],
    channel_labels: List[str],
    channel_keys: List[str],
) -> Dict[str, Path]:
    """Persist the decomposition artefacts consumed by the segmentation stage."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    pca_path = processed_dir / PCA_FILE
    np.savez_compressed(
        pca_path,
        scores=pca_result["scores"],
        score_maps=pca_result["score_maps"],
        loadings=pca_result["loadings"],
        explained_variance=pca_result["explained_variance"],
        channel_labels=np.array(channel_labels),
        channel_keys=np.array(channel_keys),
    )
    written["pca"] = pca_path

    # Note: the endmember matrices are written explicitly as W/H here. An earlier
    # version of this analysis saved the image dimensions by mistake, because the
    # cube unpacking `H, W, C = cube.shape` shadowed the NMF matrices of the same name.
    nmf_path = processed_dir / NMF_FILE
    np.savez_compressed(
        nmf_path,
        W=nmf_result["W"],
        H=nmf_result["H"],
        abundance_maps=nmf_result["abundance_maps"],
        abundance_maps_norm=nmf_result["abundance_maps_norm"],
        reconstruction_error=np.array(nmf_result["reconstruction_error"]),
        n_components=np.array(nmf_result["n_components"]),
        channel_labels=np.array(channel_labels),
        channel_keys=np.array(channel_keys),
    )
    written["nmf"] = nmf_path

    if umap_result is not None:
        umap_path = processed_dir / UMAP_FILE
        np.savez_compressed(
            umap_path,
            embedding=umap_result["embedding"],
            umap_x_img=umap_result["umap_x_img"],
            umap_y_img=umap_result["umap_y_img"],
            fg_idx=prepared.fg_idx,
            channel_labels=np.array(channel_labels),
        )
        written["umap"] = umap_path

    mask_path = processed_dir / FOREGROUND_MASK_FILE
    np.save(mask_path, prepared.fg_mask)
    written["foreground_mask"] = mask_path

    return written


def load_decomposition(processed_dir: Path) -> Dict[str, Any]:
    """Load the artefacts written by `save_decomposition`."""
    processed_dir = Path(processed_dir)
    required = processed_dir / NMF_FILE
    if not required.exists():
        raise FileNotFoundError(
            f"{required} not found. Run the 'decompose' stage first "
            f"(python scripts/run_pipeline.py --stage decompose)."
        )

    result: Dict[str, Any] = {}
    with np.load(processed_dir / PCA_FILE, allow_pickle=False) as archive:
        result["pca"] = {key: archive[key] for key in archive.files}
    with np.load(required, allow_pickle=False) as archive:
        result["nmf"] = {key: archive[key] for key in archive.files}

    umap_path = processed_dir / UMAP_FILE
    if umap_path.exists():
        with np.load(umap_path, allow_pickle=False) as archive:
            result["umap"] = {key: archive[key] for key in archive.files}

    result["fg_mask"] = np.load(processed_dir / FOREGROUND_MASK_FILE)
    return result
