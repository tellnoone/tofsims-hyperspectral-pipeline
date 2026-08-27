"""
overview.py
Stage 01 and 02a figures: raw data inspection and cleaned channel overview.

These are the sanity-check plots — they exist to confirm the binary parse produced
real images and to show how the channels relate before any modelling happens.
"""
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.features.selection import ChannelSelection, rgb_channel_indices
from src.utils.formatting import fmt_mass
from src.viz.style import (
    flatten_axes,
    grid_dimensions,
    hide_unused,
    log_stretch,
    rgb_composite,
    save_figure,
)


def plot_raw_overview(
    images: Dict[str, np.ndarray], output_path: Path, dpi: int = 200, cmap: str = "hot"
) -> Path:
    """One log-stretched panel per raw image."""
    keys = list(images)
    rows, columns = grid_dimensions(len(keys), max_columns=4)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows))
    axes = flatten_axes(axes)

    for axis, key in zip(axes, keys):
        image = images[key]
        handle = axis.imshow(log_stretch(image), cmap=cmap)
        axis.set_title(f"{fmt_mass(key)}\n{image.shape[0]}x{image.shape[1]}", fontsize=9)
        axis.axis("off")
        fig.colorbar(handle, ax=axis, fraction=0.046, shrink=0.8)

    hide_unused(axes, len(keys))
    fig.suptitle("Raw ion-count maps (log stretch)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_intensity_histograms(
    images: Dict[str, np.ndarray], output_path: Path, dpi: int = 200
) -> Path:
    """Intensity distributions — the zero spike confirms sparse, real count data."""
    keys = list(images)
    rows, columns = grid_dimensions(len(keys), max_columns=4)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.2 * rows))
    axes = flatten_axes(axes)

    for axis, key in zip(axes, keys):
        axis.hist(images[key].ravel(), bins=100, color="steelblue")
        axis.set_yscale("log")
        axis.set_title(fmt_mass(key), fontsize=9)
        axis.set_xlabel("Ion count")
        axis.set_ylabel("Pixels (log)")

    hide_unused(axes, len(keys))
    fig.suptitle("Per-channel intensity distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_preprocessing_comparison(
    image: np.ndarray, output_path: Path, dpi: int = 200, sigma: float = 1.0
) -> Path:
    """Raw vs TIC-normalized vs denoised, on one representative channel."""
    from scipy.ndimage import gaussian_filter

    total = image.sum()
    tic = image / total if total > 0 else image.copy()
    denoised = gaussian_filter(tic, sigma=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (data, title) in zip(
        axes,
        [(image, "Raw"), (tic, "TIC normalized"), (denoised, f"Denoised (sigma={sigma})")],
    ):
        handle = axis.imshow(log_stretch(data), cmap="hot")
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")
        fig.colorbar(handle, ax=axis, fraction=0.046)

    fig.suptitle("Preprocessing steps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def channel_correlation_matrix(cube: np.ndarray) -> np.ndarray:
    """Pixel-wise Pearson correlation between every pair of channels."""
    n_channels = cube.shape[-1]
    flat = cube.reshape(-1, n_channels)
    matrix = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            r, _ = pearsonr(flat[:, i], flat[:, j])
            matrix[i, j] = matrix[j, i] = r
    return matrix


def plot_correlation_matrix(
    matrix: np.ndarray, labels: Sequence[str], output_path: Path, dpi: int = 200
) -> Path:
    """Annotated heatmap of inter-channel correlation."""
    fig, axis = plt.subplots(figsize=(1.4 * len(labels) + 3, 1.4 * len(labels) + 2))
    handle = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(matrix[i, j]) > 0.5 else "black",
            )

    fig.colorbar(handle, ax=axis, fraction=0.046, label="Pearson r")
    axis.set_title("Inter-channel correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_channel_series(
    images: Dict[str, np.ndarray],
    selection: ChannelSelection,
    output_path: Path,
    config,
    dpi: int = 200,
) -> Path:
    """Every analysis channel plus an RGB composite of three of them."""
    keys = selection.keys
    rows, columns = grid_dimensions(len(keys) + 1, max_columns=3)
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.6 * rows))
    axes = flatten_axes(axes)

    for axis, key, label in zip(axes, keys, selection.labels):
        axis.imshow(log_stretch(images[key]), cmap="inferno")
        axis.set_title(label, fontsize=10)
        axis.axis("off")

    rgb_indices = rgb_channel_indices(selection, config)
    composite_axis = axes[len(keys)]
    composite_axis.imshow(rgb_composite([images[keys[i]] for i in rgb_indices]))
    composite_axis.set_title(
        "RGB composite\n"
        + " / ".join(selection.labels[i] for i in rgb_indices),
        fontsize=9,
    )
    composite_axis.axis("off")

    hide_unused(axes, len(keys) + 1)
    polarity = selection.polarity or "analysis"
    fig.suptitle(f"{polarity}-mode channel series", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_polarity_comparison(
    images: Dict[str, np.ndarray],
    selection: ChannelSelection,
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """
    Primary vs secondary acquisition side by side.

    Difference imaging is only attempted when the two grids match; otherwise the
    panels are shown independently, since the acquisitions cover different fields.
    """
    primary_key = selection.keys[0]
    secondary_key = selection.secondary_keys[0] if selection.secondary_keys else None

    n_panels = 3 if secondary_key and images[secondary_key].shape == images[primary_key].shape else 2
    if secondary_key is None:
        n_panels = 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.4))
    axes = flatten_axes(axes)

    axes[0].imshow(log_stretch(images[primary_key]), cmap="viridis")
    axes[0].set_title(f"{selection.polarity or 'Primary'}: {selection.labels[0]}", fontsize=10)
    axes[0].axis("off")

    if secondary_key is not None:
        secondary = images[secondary_key]
        axes[1].imshow(log_stretch(secondary), cmap="plasma")
        axes[1].set_title(
            f"{selection.secondary_polarity or 'Secondary'}: {fmt_mass(secondary_key)}\n"
            f"{secondary.shape[0]}x{secondary.shape[1]}",
            fontsize=10,
        )
        axes[1].axis("off")

        if n_panels == 3:
            difference = images[primary_key] - secondary
            limit = float(np.abs(difference).max()) or 1.0
            handle = axes[2].imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[2].set_title("Difference", fontsize=10)
            axes[2].axis("off")
            fig.colorbar(handle, ax=axes[2], fraction=0.046)

    fig.suptitle("Acquisition comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_mean_comparison(
    images: Dict[str, np.ndarray],
    selection: ChannelSelection,
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """Mean image of the primary group beside the secondary acquisition."""
    primary_mean = np.mean([images[key] for key in selection.keys], axis=0)
    panels = [(primary_mean, f"{selection.polarity or 'Primary'} mean ({selection.n_channels} channels)")]

    if selection.secondary_keys:
        secondary_shapes = {images[k].shape for k in selection.secondary_keys}
        if len(secondary_shapes) == 1:
            secondary_mean = np.mean([images[k] for k in selection.secondary_keys], axis=0)
            panels.append(
                (
                    secondary_mean,
                    f"{selection.secondary_polarity or 'Secondary'} mean "
                    f"({len(selection.secondary_keys)} image(s))",
                )
            )
        else:
            panels.append((images[selection.secondary_keys[0]], "Secondary (first image)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4))
    axes = flatten_axes(axes)
    for axis, (data, title) in zip(axes, panels):
        handle = axis.imshow(log_stretch(data), cmap="hot")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        fig.colorbar(handle, ax=axis, fraction=0.046)

    fig.suptitle("Mean intensity by acquisition group", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_rgb_composites(
    images: Dict[str, np.ndarray],
    selection: ChannelSelection,
    output_path: Path,
    config,
    dpi: int = 200,
) -> Path:
    """Chemical composite maps built from the analysis channels."""
    indices = rgb_channel_indices(selection, config)
    composites = [
        (
            rgb_composite([images[selection.keys[i]] for i in indices]),
            "Channel composite\n" + " / ".join(selection.labels[i] for i in indices),
        )
    ]

    if selection.secondary_keys:
        secondary = images[selection.secondary_keys[0]]
        if secondary.shape == selection.shape:
            composites.append(
                (
                    rgb_composite(
                        [
                            images[selection.keys[indices[0]]],
                            secondary,
                            images[selection.keys[indices[1]]],
                        ]
                    ),
                    "Cross-polarity composite",
                )
            )

    fig, axes = plt.subplots(1, len(composites), figsize=(6.4 * len(composites), 6.0))
    axes = flatten_axes(axes)
    for axis, (composite, title) in zip(axes, composites):
        axis.imshow(composite)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    fig.suptitle("RGB chemical composites", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)
