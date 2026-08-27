"""
selection.py
Deciding which images form the analysis cube.

A dataset typically contains one coherent set of mass maps on a shared pixel grid,
plus a handful of images from a different acquisition (opposite polarity, wider
field of view). Earlier versions of this analysis identified the two groups by
testing `shape == (640, 640)` and slicing `keys[:5]`, which silently produced an
empty cube on any other dataset.

Selection is now derived from the data: the most common pixel grid becomes the
analysis grid, and the polarity dominating that grid becomes the analysis polarity.
Both can be pinned explicitly in the config when a dataset needs it.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.formatting import extract_mass, fmt_mass


@dataclass
class ChannelSelection:
    """The outcome of choosing an analysis grid and polarity."""

    keys: List[str]
    shape: Tuple[int, int]
    polarity: Optional[str]
    labels: List[str] = field(default_factory=list)
    masses: List[Optional[float]] = field(default_factory=list)
    secondary_keys: List[str] = field(default_factory=list)
    secondary_polarity: Optional[str] = None

    @property
    def n_channels(self) -> int:
        return len(self.keys)

    def describe(self) -> str:
        polarity = self.polarity or "mixed"
        return (
            f"{self.n_channels} channels @ {self.shape[0]}x{self.shape[1]} "
            f"({polarity} mode); {len(self.secondary_keys)} secondary image(s)"
        )


def _records_by_key(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {record["key"]: record for record in metadata}


def modal_shape(images: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """The pixel grid shared by the most images — the natural analysis grid."""
    counts = Counter(tuple(image.shape) for image in images.values())
    # Ties break toward the larger grid, which carries more information.
    best_count = max(counts.values())
    candidates = [shape for shape, count in counts.items() if count == best_count]
    return max(candidates, key=lambda shape: shape[0] * shape[1])


def dominant_polarity(
    keys: Sequence[str], metadata_by_key: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """The polarity most represented among the given keys, or None if unknown."""
    polarities = [
        metadata_by_key.get(key, {}).get("polarity")
        for key in keys
        if metadata_by_key.get(key, {}).get("polarity")
    ]
    if not polarities:
        return None
    return Counter(polarities).most_common(1)[0][0]


def channel_label(key: str, record: Dict[str, Any]) -> str:
    """A short, human-readable channel label for figures."""
    mass = record.get("mass")
    polarity = record.get("polarity")
    if mass:
        label = fmt_mass(mass)
        return f"{label} ({polarity})" if polarity else label
    return key


def select_analysis_channels(
    images: Dict[str, np.ndarray],
    metadata: Sequence[Dict[str, Any]],
    config,
) -> ChannelSelection:
    """
    Choose the images that form the analysis cube.

    Honours `analysis.target_shape` and `analysis.polarity` when they are set to
    something other than `auto`; otherwise derives both from the dataset.
    """
    metadata_by_key = _records_by_key(metadata)

    if config.is_auto("analysis.target_shape"):
        shape = modal_shape(images)
    else:
        shape = tuple(config.get("analysis.target_shape"))

    on_grid = [key for key, image in images.items() if tuple(image.shape) == tuple(shape)]
    if not on_grid:
        available = sorted({tuple(img.shape) for img in images.values()})
        raise ValueError(
            f"No images match the analysis grid {tuple(shape)}. "
            f"Shapes present: {available}. "
            f"Set analysis.target_shape in the config to one of these."
        )

    if config.is_auto("analysis.polarity"):
        polarity = dominant_polarity(on_grid, metadata_by_key)
    else:
        polarity = config.get("analysis.polarity")

    keys = [
        key
        for key in on_grid
        if polarity is None or metadata_by_key.get(key, {}).get("polarity") == polarity
    ]
    if not keys:
        keys = on_grid
        polarity = None

    if config.get("analysis.sort_channels_by_mass", True):
        keys = sorted(keys, key=lambda k: (extract_mass(k) is None, extract_mass(k) or 0.0, k))
    else:
        keys = sorted(keys)

    secondary_keys = sorted(key for key in images if key not in keys)
    secondary_polarity = dominant_polarity(secondary_keys, metadata_by_key)

    return ChannelSelection(
        keys=keys,
        shape=tuple(shape),
        polarity=polarity,
        labels=[channel_label(key, metadata_by_key.get(key, {})) for key in keys],
        masses=[extract_mass(key) for key in keys],
        secondary_keys=secondary_keys,
        secondary_polarity=secondary_polarity,
    )


def build_cube(images: Dict[str, np.ndarray], selection: ChannelSelection) -> np.ndarray:
    """Stack the selected channels into an (H, W, C) float64 cube."""
    return np.stack([images[key] for key in selection.keys], axis=-1).astype(np.float64)


def resample_to(image: np.ndarray, target_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    """
    Resample an image onto the analysis grid.

    Used to bring a secondary-polarity acquisition (a different field of view, hence
    a different pixel grid) onto the primary grid for colocalization. Works for any
    pair of shapes rather than the single 896x640 -> 640x640 case the original code
    handled.
    """
    from scipy.ndimage import zoom

    if tuple(image.shape) == tuple(target_shape):
        return image

    factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
    resampled = zoom(image, factors, order=order)

    # zoom can land a pixel off; trim or pad to land exactly on the target grid.
    result = np.zeros(target_shape, dtype=resampled.dtype)
    rows = min(target_shape[0], resampled.shape[0])
    cols = min(target_shape[1], resampled.shape[1])
    result[:rows, :cols] = resampled[:rows, :cols]
    return result


def rgb_channel_indices(selection: ChannelSelection, config) -> List[int]:
    """
    Pick three channels for an RGB composite.

    `channel_semantics.rgb_channels: auto` spreads the picks across the available
    channels instead of hardcoding indices 0/2/3, which only made sense for the
    original five-channel dataset.
    """
    configured = config.get("channel_semantics.rgb_channels", "auto")
    n = selection.n_channels
    if isinstance(configured, (list, tuple)) and len(configured) == 3:
        return [int(i) % max(n, 1) for i in configured]
    if n == 0:
        return []
    if n < 3:
        return [i % n for i in range(3)]
    return [0, n // 2, n - 1]
