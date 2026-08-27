"""
decomposition_plots.py
Stage 02b figures: PCA, NMF and UMAP.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.viz.style import (
    DEFAULT_PALETTE,
    flatten_axes,
    grid_dimensions,
    hide_unused,
    normalise,
    save_figure,
    symmetric_limits,
)


# -------------------------------------------------------------------------- PCA
def plot_pca_scree(
    pca: Dict[str, Any], labels: Sequence[str], output_path: Path, dpi: int = 200
) -> Path:
    """Explained variance and component loadings side by side."""
    explained = np.asarray(pca["explained_variance"]) * 100
    components = np.arange(1, len(explained) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].bar(components, explained, color="steelblue", alpha=0.85)
    axes[0].plot(components, np.cumsum(explained), "o-", color="darkred", label="Cumulative")
    axes[0].axhline(90, ls="--", color="grey", lw=1, label="90% threshold")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree plot", fontsize=12, fontweight="bold")
    axes[0].set_xticks(components)
    axes[0].legend(fontsize=8)

    loadings = np.asarray(pca["loadings"])
    handle = axes[1].imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(loadings.shape[0]))
    axes[1].set_yticklabels([f"PC{i + 1}" for i in range(loadings.shape[0])], fontsize=8)
    axes[1].set_title("Component loadings", fontsize=12, fontweight="bold")
    fig.colorbar(handle, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_pca_component_maps(
    pca: Dict[str, Any],
    fg_mask: np.ndarray,
    output_path: Path,
    max_components: int = 4,
    dpi: int = 200,
) -> Path:
    """Spatial score maps with the matching foreground score distributions."""
    score_maps = np.asarray(pca["score_maps"])
    explained = np.asarray(pca["explained_variance"]) * 100
    n_shown = min(max_components, score_maps.shape[-1])

    fig, axes = plt.subplots(2, n_shown, figsize=(4.6 * n_shown, 8.4), squeeze=False)

    for index in range(n_shown):
        component = score_maps[:, :, index]
        limit = symmetric_limits(component, 99)
        handle = axes[0][index].imshow(component, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0][index].set_title(
            f"PC{index + 1} ({explained[index]:.1f}%)", fontsize=11, fontweight="bold"
        )
        axes[0][index].axis("off")
        fig.colorbar(handle, ax=axes[0][index], fraction=0.046)

        foreground = component.reshape(-1)[fg_mask]
        axes[1][index].hist(foreground, bins=100, color="steelblue")
        axes[1][index].set_yscale("log")
        axes[1][index].set_xlabel(f"PC{index + 1} score")
        axes[1][index].set_ylabel("Pixels (log)")

    fig.suptitle("PCA component maps (foreground pixels)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_pca_scatter(
    pca: Dict[str, Any],
    prepared,
    output_path: Path,
    subsample: int = 20000,
    random_state: int = 42,
    dpi: int = 200,
) -> Path:
    """PC1 vs PC2 for a random subsample of foreground pixels, coloured by total ion count."""
    rng = np.random.default_rng(random_state)
    fg_idx = prepared.fg_idx
    size = min(subsample, len(fg_idx))
    chosen = rng.choice(fg_idx, size=size, replace=False) if size else fg_idx

    scores = np.asarray(pca["scores"])
    totals = prepared.X_raw[chosen].sum(axis=1)

    fig, axis = plt.subplots(figsize=(7.6, 6.4))
    handle = axis.scatter(
        scores[chosen, 0], scores[chosen, 1], c=totals, cmap="hot", s=2, alpha=0.5
    )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(
        f"PCA score space ({size:,} foreground pixels)", fontsize=12, fontweight="bold"
    )
    fig.colorbar(handle, ax=axis, label="Total ion count")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# -------------------------------------------------------------------------- NMF
def plot_nmf_model_selection(
    sweep: Dict[int, Dict[str, Any]], chosen_k: int, output_path: Path, dpi: int = 200
) -> Path:
    """Reconstruction error against rank, marking the selected elbow."""
    ranks = sorted(sweep)
    errors = [sweep[k]["error"] for k in ranks]

    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(ranks, errors, "o-", color="steelblue", lw=2)
    axis.axvline(chosen_k, ls="--", color="darkred", label=f"Selected k = {chosen_k}")
    axis.set_xlabel("Number of components (k)")
    axis.set_ylabel("Reconstruction error")
    axis.set_title("NMF rank selection", fontsize=12, fontweight="bold")
    axis.set_xticks(ranks)
    axis.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_nmf_endmembers(
    nmf: Dict[str, Any], labels: Sequence[str], output_path: Path, dpi: int = 200
) -> Path:
    """Endmember spectra — the chemical signature of each component."""
    H = np.asarray(nmf["H"])
    k, n_channels = H.shape
    positions = np.arange(n_channels)
    width = 0.8 / k

    fig, axis = plt.subplots(figsize=(max(8.0, 1.6 * n_channels), 5.2))
    for index in range(k):
        axis.bar(
            positions + index * width - 0.4 + width / 2,
            H[index],
            width=width,
            label=f"EM{index + 1}",
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axis.set_ylabel("Endmember weight")
    axis.set_title("NMF endmember spectra", fontsize=12, fontweight="bold")
    axis.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_nmf_abundance_maps(nmf: Dict[str, Any], output_path: Path, dpi: int = 200) -> Path:
    """Where each endmember sits, with its abundance distribution."""
    maps = np.asarray(nmf["abundance_maps"])
    k = maps.shape[-1]

    fig, axes = plt.subplots(2, k, figsize=(4.6 * k, 8.4), squeeze=False)
    for index in range(k):
        handle = axes[0][index].imshow(maps[:, :, index], cmap="hot", vmin=0)
        axes[0][index].set_title(f"EM{index + 1} abundance", fontsize=11, fontweight="bold")
        axes[0][index].axis("off")
        fig.colorbar(handle, ax=axes[0][index], fraction=0.046)

        nonzero = maps[:, :, index][maps[:, :, index] > 0]
        if nonzero.size:
            axes[1][index].hist(nonzero, bins=100, color="darkorange")
            axes[1][index].set_yscale("log")
        axes[1][index].set_xlabel("Abundance")
        axes[1][index].set_ylabel("Pixels (log)")

    fig.suptitle("NMF abundance maps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_nmf_rgb_composite(nmf: Dict[str, Any], output_path: Path, dpi: int = 200) -> Path:
    """First three endmembers as an RGB overlay."""
    maps = np.asarray(nmf["abundance_maps_norm"])
    k = maps.shape[-1]
    layers = [maps[:, :, i] for i in range(min(3, k))]
    while len(layers) < 3:
        layers.append(np.zeros_like(layers[0]))
    composite = np.stack(layers, axis=-1)

    fig, axis = plt.subplots(figsize=(7.6, 7.2))
    axis.imshow(composite)
    axis.set_title("NMF endmember composite", fontsize=13, fontweight="bold")
    axis.axis("off")
    axis.legend(
        handles=[
            Patch(color=colour, label=f"EM{i + 1}")
            for i, colour in enumerate(["red", "green", "blue"][: min(3, k)])
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# ------------------------------------------------------------------------- UMAP
def plot_umap_scatter(
    umap_result: Dict[str, Any],
    prepared,
    nmf: Dict[str, Any],
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """The embedding coloured by intensity and by dominant endmember."""
    embedding = np.asarray(umap_result["embedding"])
    totals = prepared.X_raw[prepared.fg_mask].sum(axis=1)
    dominant = np.argmax(np.asarray(nmf["W"])[prepared.fg_idx], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2))

    handle = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=totals, cmap="hot", s=1.5, alpha=0.5)
    axes[0].set_title("UMAP — total ion count", fontsize=12, fontweight="bold")
    fig.colorbar(handle, ax=axes[0], label="Total ion count")

    for index in range(int(dominant.max()) + 1):
        mask = dominant == index
        axes[1].scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=1.5, alpha=0.5, label=f"EM{index + 1}",
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        )
    axes[1].set_title("UMAP — dominant endmember", fontsize=12, fontweight="bold")
    axes[1].legend(markerscale=6, fontsize=9)

    for axis in axes:
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_umap_spatial_maps(
    umap_result: Dict[str, Any], output_path: Path, dpi: int = 200
) -> Path:
    """Embedding coordinates painted back onto the image, plus embedding density."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))

    for axis, key, title in zip(
        axes[:2], ["umap_x_img", "umap_y_img"], ["UMAP-1 spatial map", "UMAP-2 spatial map"]
    ):
        image = np.asarray(umap_result[key])
        limit = symmetric_limits(image, 99)
        handle = axis.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.axis("off")
        fig.colorbar(handle, ax=axis, fraction=0.046)

    embedding = np.asarray(umap_result["embedding"])
    axes[2].hist2d(embedding[:, 0], embedding[:, 1], bins=100, cmap="hot", cmin=1)
    axes[2].set_title("Embedding density", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("UMAP-1")
    axes[2].set_ylabel("UMAP-2")

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_method_comparison(
    pca: Dict[str, Any],
    nmf: Dict[str, Any],
    umap_result: Optional[Dict[str, Any]],
    labels: Sequence[str],
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """One figure placing the three decompositions side by side."""
    k = np.asarray(nmf["abundance_maps"]).shape[-1]
    n_pcs = min(3, np.asarray(pca["score_maps"]).shape[-1])
    columns = max(n_pcs, k, 2)

    fig, axes = plt.subplots(3, columns, figsize=(4.6 * columns, 13.2), squeeze=False)

    score_maps = np.asarray(pca["score_maps"])
    explained = np.asarray(pca["explained_variance"]) * 100
    for index in range(n_pcs):
        limit = symmetric_limits(score_maps[:, :, index], 99)
        axes[0][index].imshow(score_maps[:, :, index], cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0][index].set_title(f"PC{index + 1} ({explained[index]:.1f}%)", fontsize=10)
        axes[0][index].axis("off")
    hide_unused(axes[0], n_pcs)

    abundance = np.asarray(nmf["abundance_maps"])
    for index in range(k):
        axes[1][index].imshow(abundance[:, :, index], cmap="hot", vmin=0)
        axes[1][index].set_title(f"NMF EM{index + 1}", fontsize=10)
        axes[1][index].axis("off")
    hide_unused(axes[1], k)

    if umap_result is not None:
        embedding = np.asarray(umap_result["embedding"])
        axes[2][0].hist2d(embedding[:, 0], embedding[:, 1], bins=100, cmap="hot", cmin=1)
        axes[2][0].set_title("UMAP density", fontsize=10)
        image = np.asarray(umap_result["umap_x_img"])
        limit = symmetric_limits(image, 99)
        axes[2][1].imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[2][1].set_title("UMAP-1 spatial", fontsize=10)
        axes[2][1].axis("off")
        hide_unused(axes[2], 2)
    else:
        hide_unused(axes[2], 0)

    for row, name in enumerate(["PCA", "NMF", "UMAP"]):
        axes[row][0].set_ylabel(name, fontsize=12, fontweight="bold")

    fig.suptitle("Decomposition method comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)
