"""
style.py
Shared plotting helpers.

Ion-count images are extremely right-skewed — a few hot pixels span three orders of
magnitude more than the tissue signal — so essentially every display here applies a
log or percentile stretch. Centralising that keeps every figure comparable.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.formatting import extract_mass  # noqa: E402

DEFAULT_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999",
]


def save_figure(fig, path: Path, dpi: int = 250, close: bool = True) -> Path:
    """Write a figure and release it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def log_stretch(image: np.ndarray) -> np.ndarray:
    """log(1+x) — the standard display transform for ion counts."""
    return np.log1p(np.asarray(image, dtype=np.float64))


def normalise(image: np.ndarray) -> np.ndarray:
    """Min-max an array to [0, 1] for display."""
    image = np.asarray(image, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = finite.min(), finite.max()
    return (image - low) / (high - low + 1e-12)


def rgb_composite(
    channels: Sequence[np.ndarray], gamma: float = 1.0, log: bool = True
) -> np.ndarray:
    """Combine up to three maps into an RGB image, each stretched independently."""
    layers = []
    for channel in list(channels)[:3]:
        layer = normalise(log_stretch(channel) if log else channel)
        if gamma != 1.0:
            layer = layer ** gamma
        layers.append(layer)
    while len(layers) < 3:
        layers.append(np.zeros_like(layers[0]))
    return np.stack(layers, axis=-1)


def symmetric_limits(image: np.ndarray, percentile: float = 99) -> float:
    """A symmetric colour limit for diverging maps, robust to outliers."""
    value = np.nanpercentile(np.abs(image), percentile)
    return float(value) if value > 0 else 1.0


def segment_names(n_segments: int, config=None) -> Dict[int, str]:
    """
    Display names for segments.

    Defaults to generic names. Endmember ordering is not stable across datasets (or
    even across NMF re-fits), so chemical names are only used when a config
    explicitly supplies them for this dataset.
    """
    configured = (config.get("channel_semantics.segment_labels") if config else None) or []
    names = {}
    for index in range(n_segments):
        if index < len(configured) and configured[index]:
            names[index] = str(configured[index])
        else:
            names[index] = f"Segment {index + 1} (EM{index + 1})"
    return names


def channel_colour(label: str, index: int, config=None) -> str:
    """Colour for a channel, honouring `channel_semantics.masses` when it matches."""
    semantics = (config.get("channel_semantics.masses") if config else None) or {}
    mass = extract_mass(label)
    if mass is not None:
        for key, value in semantics.items():
            try:
                if abs(float(key) - mass) < 0.05 and isinstance(value, dict):
                    return value.get("color", DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])
            except (TypeError, ValueError):
                continue
    return DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]


def channel_annotation(label: str, config=None) -> Optional[str]:
    """Optional chemical annotation for a channel, if the config names one."""
    semantics = (config.get("channel_semantics.masses") if config else None) or {}
    mass = extract_mass(label)
    if mass is None:
        return None
    for key, value in semantics.items():
        try:
            if abs(float(key) - mass) < 0.05 and isinstance(value, dict):
                return value.get("label")
        except (TypeError, ValueError):
            continue
    return None


def grid_dimensions(n_items: int, max_columns: int = 4) -> tuple:
    """Rows and columns for a roughly square panel grid."""
    columns = min(max_columns, max(n_items, 1))
    rows = int(np.ceil(n_items / columns))
    return rows, columns


def flatten_axes(axes) -> List[Any]:
    """Always get a flat list of axes, whatever subplots() returned."""
    return list(np.atleast_1d(np.asarray(axes)).ravel())


def hide_unused(axes: Sequence, used: int) -> None:
    """Switch off any leftover panels in a grid."""
    for axis in list(axes)[used:]:
        axis.axis("off")


def panel_label(ax, text: str, **kwargs) -> None:
    """Draw an (a)/(b)/(c) style panel label."""
    ax.text(
        -0.05, 1.05, text, transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top", ha="right", **kwargs
    )
