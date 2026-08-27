"""
report.py
Stage 05: publication-style master figures.

These assemble the outputs of the earlier stages into the figures that go into a
write-up. The captions and summary panels are generated from the data and metadata
actually loaded, rather than hand-typed for one specimen, so they stay truthful when
the pipeline is pointed at a different dataset.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.selection import ChannelSelection, rgb_channel_indices
from src.utils.formatting import fmt_mass
from src.viz.style import (
    DEFAULT_PALETTE,
    channel_annotation,
    log_stretch,
    normalise,
    panel_label,
    rgb_composite,
    save_figure,
    segment_names,
)


def _text_panel(axis, title: str, lines: Sequence[str]) -> None:
    """Render a titled block of summary text in place of a plot."""
    axis.axis("off")
    axis.text(
        0.0, 1.0, title, transform=axis.transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    axis.text(
        0.0, 0.88, "\n".join(lines), transform=axis.transAxes,
        fontsize=9, va="top", family="monospace", linespacing=1.6,
    )


def _acquisition_summary(
    selection: ChannelSelection, metadata: Sequence[Dict[str, Any]]
) -> List[str]:
    """Summarise the acquisition from parsed metadata — no hardcoded numbers."""
    by_key = {record["key"]: record for record in metadata}
    masses = [m for m in selection.masses if m is not None]
    shots = {by_key.get(k, {}).get("shots") for k in selection.keys}
    shots = sorted(s for s in shots if s is not None)
    currents = sorted({by_key.get(k, {}).get("current") for k in selection.keys} - {None})

    lines = [
        f"Analysis channels : {selection.n_channels}",
        f"Pixel grid        : {selection.shape[0]} x {selection.shape[1]}",
        f"Polarity          : {selection.polarity or 'mixed'}",
    ]
    if masses:
        lines.append(f"m/z range         : {min(masses):.2f} - {max(masses):.2f}")
    if shots:
        lines.append(f"Shots per pixel   : {', '.join(str(s) for s in shots)}")
    if currents:
        lines.append(f"Beam current      : {', '.join(currents)}")
    if selection.secondary_keys:
        lines.append(
            f"Secondary images  : {len(selection.secondary_keys)} "
            f"({selection.secondary_polarity or 'other'} mode)"
        )
    return lines


def figure_dataset_overview(
    images: Dict[str, np.ndarray],
    selection: ChannelSelection,
    metadata: Sequence[Dict[str, Any]],
    output_path: Path,
    config,
    dpi: int = 300,
) -> Path:
    """Figure 1 — every analysis channel, an RGB composite, and the acquisition summary."""
    n_channels = selection.n_channels
    columns = max(n_channels, 3)
    fig = plt.figure(figsize=(4.0 * columns, 9.0))
    grid = fig.add_gridspec(2, columns, hspace=0.28, wspace=0.22)

    for index, (key, label) in enumerate(zip(selection.keys, selection.labels)):
        axis = fig.add_subplot(grid[0, index])
        axis.imshow(log_stretch(images[key]), cmap="hot")
        annotation = channel_annotation(key, config)
        axis.set_title(f"{label}\n{annotation}" if annotation else label, fontsize=10)
        axis.axis("off")
        panel_label(axis, f"({chr(97 + index)})")

    rgb_indices = rgb_channel_indices(selection, config)
    composite_axis = fig.add_subplot(grid[1, 0])
    composite_axis.imshow(
        rgb_composite([images[selection.keys[i]] for i in rgb_indices], gamma=0.5)
    )
    composite_axis.set_title("RGB composite", fontsize=10)
    composite_axis.axis("off")
    panel_label(composite_axis, f"({chr(97 + n_channels)})")

    key_axis = fig.add_subplot(grid[1, 1])
    key_lines = []
    for colour, index in zip(["Red  ", "Green", "Blue "], rgb_indices):
        annotation = channel_annotation(selection.keys[index], config)
        suffix = f" ({annotation})" if annotation else ""
        key_lines.append(f"{colour} = {selection.labels[index]}{suffix}")
    _text_panel(key_axis, "Composite colour key", key_lines)

    summary_axis = fig.add_subplot(grid[1, 2])
    _text_panel(summary_axis, "Acquisition summary", _acquisition_summary(selection, metadata))

    fig.suptitle("Figure 1 — Dataset overview", fontsize=16, fontweight="bold")
    return save_figure(fig, output_path, dpi=dpi)


def figure_nmf_decomposition(
    nmf: Dict[str, Any],
    labels: Sequence[str],
    output_path: Path,
    config,
    dpi: int = 300,
) -> Path:
    """Figure 2 — endmember abundance maps, composite and spectra."""
    abundance = np.asarray(nmf["abundance_maps"])
    abundance_norm = np.asarray(nmf["abundance_maps_norm"])
    H = np.asarray(nmf["H"])
    k = abundance.shape[-1]
    names = segment_names(k, config)

    columns = max(k, 3)
    fig = plt.figure(figsize=(4.4 * columns, 10.0))
    grid = fig.add_gridspec(2, columns, hspace=0.28, wspace=0.22)

    for index in range(k):
        axis = fig.add_subplot(grid[0, index])
        axis.imshow(abundance[:, :, index], cmap="hot", vmin=0)
        axis.set_title(names[index], fontsize=10)
        axis.axis("off")
        panel_label(axis, f"({chr(97 + index)})")

    composite_axis = fig.add_subplot(grid[1, 0])
    layers = [abundance_norm[:, :, i] for i in range(min(3, k))]
    while len(layers) < 3:
        layers.append(np.zeros_like(layers[0]))
    composite_axis.imshow(np.stack(layers, axis=-1))
    composite_axis.set_title("Endmember composite", fontsize=10)
    composite_axis.axis("off")
    panel_label(composite_axis, f"({chr(97 + k)})")

    spectra_axis = fig.add_subplot(grid[1, 1:])
    positions = np.arange(len(labels))
    width = 0.8 / k
    for index in range(k):
        spectra_axis.bar(
            positions + index * width - 0.4 + width / 2, H[index], width=width,
            label=names[index], color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        )
    spectra_axis.set_xticks(positions)
    spectra_axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    spectra_axis.set_ylabel("Endmember weight")
    spectra_axis.set_title("Endmember spectra", fontsize=10)
    spectra_axis.legend(fontsize=8)

    fig.suptitle(
        f"Figure 2 — NMF decomposition (k = {k}, "
        f"reconstruction error = {float(nmf['reconstruction_error']):.1f})",
        fontsize=16, fontweight="bold",
    )
    return save_figure(fig, output_path, dpi=dpi)


def figure_segmentation_comparison(
    label_images: Dict[str, np.ndarray],
    agreement: pd.DataFrame,
    output_path: Path,
    cmap: str = "Set1",
    dpi: int = 300,
) -> Path:
    """
    Figure 3 — the segmentations side by side, captioned with measured agreement.

    The caption reports the actual ARI range rather than asserting a conclusion, so
    the figure stays honest if a rerun produces different agreement.
    """
    panels = list(label_images.items())
    columns = min(2, len(panels))
    rows = int(np.ceil(len(panels) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 6.0 * rows), squeeze=False)
    flat_axes = list(np.asarray(axes).ravel())

    for index, (axis, (title, image)) in enumerate(zip(flat_axes, panels)):
        masked = np.ma.masked_where(image < 0, image)
        axis.imshow(masked, cmap=cmap, interpolation="nearest")
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")
        panel_label(axis, f"({chr(97 + index)})")

    for axis in flat_axes[len(panels):]:
        axis.axis("off")

    if not agreement.empty:
        caption = (
            f"Pairwise agreement: ARI {agreement['ARI'].min():.2f}-{agreement['ARI'].max():.2f}, "
            f"NMI {agreement['NMI'].min():.2f}-{agreement['NMI'].max():.2f}"
        )
    else:
        caption = "Pairwise agreement not computed."

    fig.suptitle("Figure 3 — Segmentation method comparison", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    return save_figure(fig, output_path, dpi=dpi)


def figure_statistical_validation(
    profiles: pd.DataFrame,
    results: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Figure 4 — chemistry profiles beside the omnibus test results."""
    fig = plt.figure(figsize=(17.0, 6.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9], wspace=0.32)

    composition_axis = fig.add_subplot(grid[0, 0])
    percentages = profiles.div(profiles.sum(axis=1), axis=0) * 100
    left = np.zeros(len(percentages))
    for index, channel in enumerate(percentages.columns):
        composition_axis.barh(
            percentages.index, percentages[channel], left=left,
            label=channel, color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        )
        left += percentages[channel].to_numpy()
    composition_axis.set_xlabel("Share of segment signal (%)")
    composition_axis.set_title("Segment composition", fontsize=11, fontweight="bold")
    composition_axis.legend(fontsize=7, loc="lower right")
    panel_label(composition_axis, "(a)")

    heatmap_axis = fig.add_subplot(grid[0, 1])
    z_scores = (profiles - profiles.mean()) / profiles.std().replace(0, 1)
    handle = heatmap_axis.imshow(z_scores.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    heatmap_axis.set_xticks(range(len(profiles.columns)))
    heatmap_axis.set_xticklabels(profiles.columns, rotation=45, ha="right", fontsize=8)
    heatmap_axis.set_yticks(range(len(profiles.index)))
    heatmap_axis.set_yticklabels(profiles.index, fontsize=8)
    heatmap_axis.set_title("Channel enrichment (z-score)", fontsize=11, fontweight="bold")
    fig.colorbar(handle, ax=heatmap_axis, fraction=0.046)
    panel_label(heatmap_axis, "(b)")

    table_axis = fig.add_subplot(grid[0, 2])
    table_axis.axis("off")
    columns = ["mass_label", "H_statistic", "eta_squared", "effect_size"]
    table = table_axis.table(
        cellText=results[columns].values,
        colLabels=["Channel", "H", "eps^2", "Effect"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    max_p = results["p_value"].max()
    table_axis.set_title(
        f"Kruskal-Wallis by channel\n(all p <= {max_p:.1e})",
        fontsize=11, fontweight="bold", pad=16,
    )
    panel_label(table_axis, "(c)")

    fig.suptitle("Figure 4 — Statistical validation", fontsize=16, fontweight="bold")
    return save_figure(fig, output_path, dpi=dpi)


def figure_cross_polarity(
    secondary_image: np.ndarray,
    endmember_map: np.ndarray,
    endmember_index: int,
    coloc: pd.DataFrame,
    output_path: Path,
    config,
    clip_percentile: float = 75,
    dpi: int = 300,
) -> Path:
    """Figure 5 — secondary-polarity signal against the endmember it tracks best."""
    nonzero = endmember_map[endmember_map > 0]
    ceiling = np.percentile(nonzero, clip_percentile) if nonzero.size else endmember_map.max()
    display = log_stretch(np.clip(endmember_map, 0, ceiling))
    names = segment_names(endmember_index + 1, config)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6))

    axes[0].imshow(log_stretch(secondary_image), cmap="viridis")
    axes[0].set_title("Secondary-polarity signal", fontsize=11)
    axes[0].axis("off")
    panel_label(axes[0], "(a)")

    axes[1].imshow(display, cmap="hot")
    axes[1].set_title(names.get(endmember_index, f"EM{endmember_index + 1}"), fontsize=11)
    axes[1].axis("off")
    panel_label(axes[1], "(b)")

    axes[2].imshow(display, cmap="hot", alpha=0.55)
    nonzero_secondary = secondary_image[secondary_image > 0]
    if nonzero_secondary.size:
        threshold = np.percentile(nonzero_secondary, 95)
        overlay = np.zeros((*secondary_image.shape, 4))
        overlay[secondary_image > threshold] = [1.0, 0.0, 0.0, 0.75]
        axes[2].imshow(overlay)
    axes[2].set_title("Top 5% secondary hotspots overlaid", fontsize=11)
    axes[2].axis("off")
    panel_label(axes[2], "(c)")

    if not coloc.empty:
        strongest = coloc.iloc[coloc["pearson_r"].abs().idxmax()]
        caption = (
            f"Strongest colocalization: {strongest['segment_name']}, "
            f"r = {strongest['pearson_r']:.3f} ({strongest['interpretation'].lower()})"
        )
    else:
        caption = "Colocalization not computed."

    fig.suptitle("Figure 5 — Cross-polarity comparison", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")
    return save_figure(fig, output_path, dpi=dpi)


def write_manifest(figures: Dict[str, Path], output_path: Path) -> Path:
    """Record which figures were produced, for the write-up."""
    lines = ["# Generated report figures", ""]
    for title, path in figures.items():
        status = "ok" if Path(path).exists() else "missing"
        lines.append(f"- **{title}** — `{Path(path).name}` ({status})")
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
