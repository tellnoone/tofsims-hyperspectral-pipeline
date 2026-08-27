"""
cluster.py
Spatial chemical segmentation and cross-method validation.

Four segmentations are produced from the same foreground pixels:
  * dominant NMF endmember - the primary, chemically interpretable partition
  * K-Means on PCA scores  - a conventional baseline (plus an over-segmented run)
  * Gaussian mixture on PCA - a soft partition, yielding per-pixel uncertainty
  * HDBSCAN on the UMAP embedding - density-based, allows unassigned pixels

Agreement between them is reported rather than assumed: methods that disagree are
evidence about how well-separated the chemistry actually is, and the metrics below
(ARI, NMI, purity against a majority-class baseline) are what make that honest.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.constants import SEGMENTATION_METRICS_FILE, SEGMENTS_FILE

NOISE_LABEL = -1
BACKGROUND_LABEL = -1


# ------------------------------------------------------------------ feature prep
def pca_feature_space(
    score_maps: np.ndarray, fg_idx: np.ndarray, n_components: int = 3
) -> np.ndarray:
    """Foreground PCA scores, standardized so K-Means and GMM see isotropic axes."""
    n_components = min(n_components, score_maps.shape[-1])
    flat = score_maps.reshape(-1, score_maps.shape[-1])[fg_idx, :n_components]
    return StandardScaler().fit_transform(flat)


def nmf_feature_space(abundance_maps: np.ndarray, fg_idx: np.ndarray) -> np.ndarray:
    """Foreground NMF abundances."""
    return abundance_maps.reshape(-1, abundance_maps.shape[-1])[fg_idx]


# -------------------------------------------------------------------- clustering
def dominant_endmember_labels(abundances: np.ndarray) -> np.ndarray:
    """Assign each pixel to the endmember contributing most of its signal."""
    return np.argmax(abundances, axis=1)


def run_kmeans(X: np.ndarray, n_clusters: int, n_init: int = 10, random_state: int = 42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return model.fit_predict(X), model


def run_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Soft clustering; per-pixel entropy quantifies assignment confidence."""
    model = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type, random_state=random_state
    )
    labels = model.fit_predict(X)
    probabilities = model.predict_proba(X)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    return {
        "labels": labels,
        "probabilities": probabilities,
        "entropy": entropy,
        "mean_entropy": float(entropy.mean()),
        "model": model,
    }


def resolve_min_cluster_size(value: Any, n_foreground: int, fraction: float = 0.02) -> int:
    """
    Resolve HDBSCAN's `min_cluster_size`, scaling it to the dataset when set to auto.

    A fixed pixel count tuned for one image is meaningless on a dataset of a
    different size, so `auto` asks for clusters covering at least `fraction` of the
    foreground, with a floor that keeps tiny datasets workable.
    """
    if isinstance(value, str) and value.lower() == "auto":
        return max(50, int(round(fraction * n_foreground)))
    return int(value)


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int = 20,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> Dict[str, Any]:
    """
    Density-based clustering of the UMAP embedding.

    Uses scikit-learn's HDBSCAN (>=1.3) rather than the standalone `hdbscan`
    package, which conflicts with recent numpy builds.
    """
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        copy=True,  # leave the caller's embedding untouched
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    return {
        "labels": labels,
        "n_clusters": int(n_clusters),
        "n_noise": int(np.sum(labels == NOISE_LABEL)),
        "noise_pct": float(np.mean(labels == NOISE_LABEL) * 100),
        "model": clusterer,
    }


def hdbscan_sensitivity(
    X: np.ndarray,
    reference_labels: np.ndarray,
    min_cluster_sizes: Sequence[int],
    min_samples_values: Sequence[int],
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> pd.DataFrame:
    """Grid over HDBSCAN's two main knobs, reporting stability against a reference."""
    rows = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_values:
            result = run_hdbscan(X, mcs, ms, metric, cluster_selection_method)
            labels = result["labels"]
            valid = labels != NOISE_LABEL
            ari = (
                adjusted_rand_score(reference_labels[valid], labels[valid])
                if valid.sum() > 1
                else np.nan
            )
            rows.append(
                {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": result["n_clusters"],
                    "noise_pct": round(result["noise_pct"], 2),
                    "ari_vs_reference": round(ari, 4) if ari == ari else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- metrics
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score, guarding the degenerate single-cluster case."""
    unique = set(labels) - {NOISE_LABEL}
    if len(unique) < 2:
        return float("nan")
    mask = labels != NOISE_LABEL
    if mask.sum() <= len(unique):
        return float("nan")
    return float(silhouette_score(X[mask], labels[mask]))


def agreement_metrics(
    label_sets: Dict[str, np.ndarray], restrict_mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Pairwise ARI and NMI between every pair of segmentations."""
    names = list(label_sets)
    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = label_sets[first], label_sets[second]
            if restrict_mask is not None:
                a, b = a[restrict_mask], b[restrict_mask]
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "ARI": round(float(adjusted_rand_score(a, b)), 4),
                    "NMI": round(float(normalized_mutual_info_score(a, b)), 4),
                }
            )
    return pd.DataFrame(rows)


def segment_purity(reference_labels: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    How cleanly each cluster maps onto a single reference segment.

    Reported against the majority-class baseline, because a dataset dominated by one
    segment yields high purity for free and the raw number would overstate agreement.
    """
    rows = []
    for cluster in sorted(set(cluster_labels) - {NOISE_LABEL}):
        mask = cluster_labels == cluster
        if not mask.any():
            continue
        counts = np.bincount(reference_labels[mask])
        majority = int(np.argmax(counts))
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(mask.sum()),
                "majority_segment": majority,
                "purity": round(float(counts[majority] / mask.sum()), 4),
            }
        )

    frame = pd.DataFrame(rows)
    baseline = float(np.max(np.bincount(reference_labels)) / len(reference_labels))
    weighted = (
        float(np.average(frame["purity"], weights=frame["size"])) if not frame.empty else float("nan")
    )
    return {
        "per_cluster": frame,
        "weighted_purity": round(weighted, 4),
        "mean_purity": round(float(frame["purity"].mean()), 4) if not frame.empty else float("nan"),
        "majority_baseline": round(baseline, 4),
    }


# ------------------------------------------------------------------- image maps
def labels_to_image(
    labels: np.ndarray, fg_idx: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Scatter foreground-only labels back onto the full image grid."""
    image = np.full(shape, BACKGROUND_LABEL, dtype=int)
    image.reshape(-1)[fg_idx] = labels
    return image


def values_to_image(
    values: np.ndarray, fg_idx: np.ndarray, shape: Tuple[int, int], fill: float = np.nan
) -> np.ndarray:
    """Scatter a foreground-only continuous quantity onto the full image grid."""
    image = np.full(shape[0] * shape[1], fill, dtype=float)
    image[fg_idx] = values
    return image.reshape(shape)


# ------------------------------------------------------------------ persistence
def save_segments(
    processed_dir: Path,
    label_sets: Dict[str, np.ndarray],
    images: Dict[str, np.ndarray],
    extras: Dict[str, np.ndarray],
    metrics: pd.DataFrame,
    channel_labels: List[str],
) -> Dict[str, Path]:
    """Persist labels, label images and the metrics table."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {f"labels_{name}": arr for name, arr in label_sets.items()}
    payload.update({f"seg_{name}": arr for name, arr in images.items()})
    payload.update(extras)
    payload["channel_labels"] = np.array(channel_labels)

    segments_path = processed_dir / SEGMENTS_FILE
    np.savez_compressed(segments_path, **payload)

    metrics_path = processed_dir / SEGMENTATION_METRICS_FILE
    metrics.to_csv(metrics_path, index=False)

    return {"segments": segments_path, "metrics": metrics_path}


def load_segments(processed_dir: Path) -> Dict[str, np.ndarray]:
    """Load the artefacts written by `save_segments`."""
    path = Path(processed_dir) / SEGMENTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the 'segment' stage first "
            f"(python scripts/run_pipeline.py --stage segment)."
        )
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
