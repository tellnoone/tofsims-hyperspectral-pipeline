"""
stats_plots.py
Stage 04 figures: statistical validation and cross-polarity colocalization.
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.formatting import fmt_mass, fmt_p
from src.viz.style import DEFAULT_PALETTE, log_stretch, save_figure


def plot_dunn_heatmap(
    posthoc: pd.DataFrame,
    channel: str,
    output_path: Path,
    p_floor: float = 1e-50,
    dpi: int = 250,
) -> Path:
    """
    Pairwise post-hoc significance for the most discriminating channel.

    Displayed as -log10(p) with the diagonal masked; the floor keeps
    underflowed p-values from blowing out the colour scale.
    """
    matrix = posthoc.to_numpy(dtype=float)
    display = -np.log10(np.clip(matrix, p_floor, 1.0))
    np.fill_diagonal(display, np.nan)

    labels = [f"Segment {int(c) + 1}" for c in posthoc.columns]

    fig, axis = plt.subplots(figsize=(1.5 * len(labels) + 3.2, 1.4 * len(labels) + 2.6))
    handle = axis.imshow(np.ma.masked_invalid(display), cmap="viridis")

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axis.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                axis.text(
                    j, i, fmt_p(matrix[i, j], floor=p_floor),
                    ha="center", va="center", fontsize=8, color="white",
                )

    fig.colorbar(handle, ax=axis, fraction=0.046, label="-log10(p)")
    axis.set_title(
        f"Dunn's post-hoc — {fmt_mass(channel)}\n(largest effect size)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_segment_chemistry_profiles(
    profiles: pd.DataFrame, results: pd.DataFrame, output_path: Path, dpi: int = 250
) -> Path:
    """Segment composition beside the per-channel effect sizes that validate it."""
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

    axes[1].barh(
        results["mass_label"], results["eta_squared"],
        color=[DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(len(results))],
    )
    axes[1].axvline(0.14, ls="--", color="darkred", label="Large effect (0.14)")
    axes[1].axvline(0.06, ls=":", color="grey", label="Medium effect (0.06)")
    axes[1].set_xlabel("Effect size (epsilon-squared)")
    axes[1].set_title("Segment separation per channel", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_secondary_overlay(
    secondary_image: np.ndarray,
    endmember_map: np.ndarray,
    output_path: Path,
    endmember_index: int,
    clip_percentile: float = 75,
    dpi: int = 250,
) -> Path:
    """
    Secondary-polarity signal overlaid on the endmember it best tracks.

    The abundance map is clipped at a mid percentile rather than the usual 99th:
    these distributions are so right-skewed that the 99th percentile equals the
    maximum, which would flatten the map to near-black.
    """
    nonzero = endmember_map[endmember_map > 0]
    ceiling = np.percentile(nonzero, clip_percentile) if nonzero.size else endmember_map.max()
    display = log_stretch(np.clip(endmember_map, 0, ceiling))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))

    handle = axes[0].imshow(log_stretch(secondary_image), cmap="viridis")
    axes[0].set_title("Secondary-polarity signal\n(resampled to analysis grid)", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(handle, ax=axes[0], fraction=0.046)

    handle = axes[1].imshow(display, cmap="hot")
    axes[1].set_title(f"EM{endmember_index + 1} abundance", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(handle, ax=axes[1], fraction=0.046)

    axes[2].imshow(display, cmap="hot", alpha=0.55)
    nonzero_secondary = secondary_image[secondary_image > 0]
    if nonzero_secondary.size:
        threshold = np.percentile(nonzero_secondary, 95)
        hotspots = secondary_image > threshold
        overlay = np.zeros((*secondary_image.shape, 4))
        overlay[hotspots] = [1.0, 0.0, 0.0, 0.75]
        axes[2].imshow(overlay)
    axes[2].set_title("Top 5% secondary hotspots\non endmember map", fontsize=11)
    axes[2].axis("off")

    fig.suptitle("Cross-polarity comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_colocalization(coloc: pd.DataFrame, output_path: Path, dpi: int = 250) -> Path:
    """Correlation of the secondary channel with each segment's footprint."""
    fig, axis = plt.subplots(figsize=(8.6, 5.2))

    colours = [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(len(coloc))]
    axis.bar(coloc["segment_name"], coloc["pearson_r"], color=colours)
    axis.axhline(0, color="black", lw=1)
    axis.axhline(0.5, ls="--", color="darkred", lw=1, label="Strong (|r| > 0.5)")
    axis.axhline(-0.5, ls="--", color="darkred", lw=1)
    axis.axhline(0.3, ls=":", color="grey", lw=1, label="Moderate (|r| > 0.3)")
    axis.axhline(-0.3, ls=":", color="grey", lw=1)

    axis.set_ylabel("Pearson r (secondary ion vs segment membership)")
    axis.set_title("Cross-polarity colocalization", fontsize=12, fontweight="bold")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)
