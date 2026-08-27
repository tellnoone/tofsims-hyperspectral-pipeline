"""
segmentation_plots.py
Stage 03 figures: segmentation maps and cross-method validation.
"""
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import DEFAULT_PALETTE, flatten_axes, hide_unused, save_figure


def plot_segmentation_comparison(
    label_images: Dict[str, np.ndarray],
    output_path: Path,
    entropy_image: np.ndarray = None,
    cmap: str = "Set1",
    dpi: int = 250,
) -> Path:
    """Every segmentation on the same grid, plus the GMM uncertainty map."""
    panels = list(label_images.items())
    if entropy_image is not None:
        panels.append(("GMM assignment entropy", entropy_image))

    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 5.0 * rows), squeeze=False)
    axes = flatten_axes(axes)

    for axis, (title, image) in zip(axes, panels):
        if title.startswith("GMM assignment entropy"):
            limit = np.nanpercentile(image, 95)
            handle = axis.imshow(image, cmap="YlOrRd", vmax=limit)
            fig.colorbar(handle, ax=axis, fraction=0.046, label="Entropy (nats)")
        else:
            masked = np.ma.masked_where(image < 0, image)
            handle = axis.imshow(masked, cmap=cmap, interpolation="nearest")
            fig.colorbar(handle, ax=axis, fraction=0.046, label="Segment")
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")

    hide_unused(axes, len(panels))
    fig.suptitle("Segmentation methods compared", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_segment_chemistry(
    profiles: pd.DataFrame, output_path: Path, dpi: int = 250
) -> Path:
    """
    Segment chemistry two ways.

    The stacked bars show composition within each segment; the z-scored heatmap
    shows which channels distinguish segments from one another, which the
    composition view alone hides when one channel dominates every segment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.8))

    percentages = profiles.div(profiles.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(percentages))
    for index, channel in enumerate(percentages.columns):
        axes[0].bar(
            percentages.index, percentages[channel], bottom=bottom,
            label=channel, color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        )
        bottom += percentages[channel].to_numpy()
    axes[0].set_ylabel("Share of segment signal (%)")
    axes[0].set_title("Segment composition", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8, bbox_to_anchor=(1.0, 1.0))
    axes[0].tick_params(axis="x", rotation=20)

    z_scores = (profiles - profiles.mean()) / profiles.std().replace(0, 1)
    handle = axes[1].imshow(z_scores.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axes[1].set_xticks(range(len(profiles.columns)))
    axes[1].set_xticklabels(profiles.columns, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(profiles.index)))
    axes[1].set_yticklabels(profiles.index, fontsize=9)
    axes[1].set_title("Channel enrichment (z-score)", fontsize=12, fontweight="bold")
    fig.colorbar(handle, ax=axes[1], fraction=0.046, label="z-score")

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_agreement_table(agreement: pd.DataFrame, output_path: Path, dpi: int = 250) -> Path:
    """ARI and NMI between every pair of segmentations."""
    fig, axis = plt.subplots(figsize=(9.0, 0.5 * len(agreement) + 2.2))
    axis.axis("off")

    table = axis.table(
        cellText=agreement.round(4).values,
        colLabels=agreement.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    axis.set_title(
        "Cross-method agreement\n(ARI/NMI near 0 means the methods find different structure)",
        fontsize=12, fontweight="bold", pad=18,
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_segment_purity(purity: Dict[str, Any], output_path: Path, dpi: int = 250) -> Path:
    """Per-cluster purity against the majority-class baseline."""
    frame = purity["per_cluster"]
    fig, axis = plt.subplots(figsize=(8.4, 5.2))

    axis.bar(
        [f"Cluster {int(c) + 1}" for c in frame["cluster"]],
        frame["purity"],
        color=[DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(len(frame))],
    )
    axis.axhline(
        purity["majority_baseline"], ls="--", color="darkred",
        label=f"Majority-class baseline ({purity['majority_baseline']:.2f})",
    )
    axis.axhline(
        purity["weighted_purity"], ls=":", color="black",
        label=f"Pixel-weighted purity ({purity['weighted_purity']:.2f})",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Purity vs dominant-endmember segments")
    axis.set_title("K-Means cluster purity", fontsize=12, fontweight="bold")
    axis.legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_hdbscan_sensitivity(
    sweep: pd.DataFrame, output_path: Path, dpi: int = 250
) -> Path:
    """How HDBSCAN's cluster count and noise fraction respond to its parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))

    for min_samples, group in sweep.groupby("min_samples"):
        axes[0].plot(
            group["min_cluster_size"], group["n_clusters"], "o-", label=f"min_samples={min_samples}"
        )
        axes[1].plot(
            group["min_cluster_size"], group["noise_pct"], "o-", label=f"min_samples={min_samples}"
        )

    axes[0].set_xlabel("min_cluster_size")
    axes[0].set_ylabel("Clusters found")
    axes[0].set_title("Cluster count sensitivity", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("min_cluster_size")
    axes[1].set_ylabel("Unassigned pixels (%)")
    axes[1].set_title("Noise fraction sensitivity", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)
