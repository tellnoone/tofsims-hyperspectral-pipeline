"""
validation.py
Statistical validation of the segmentation.

Segment boundaries drawn by an unsupervised method are a hypothesis, not a result.
This module tests whether the segments actually differ in ion intensity, per mass
channel, and how large the difference is:

  * Kruskal-Wallis per channel - a non-parametric omnibus test, chosen because
    ion-count distributions are heavily zero-inflated and non-normal.
  * Dunn's post-hoc with multiple-comparison correction - which specific pairs of
    segments differ.
  * Epsilon-squared effect size - because with tens of thousands of pixels almost
    any difference is "significant"; the effect size is what carries meaning.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.constants import COLOCALIZATION_FILE, POSTHOC_FILE, STATISTICAL_RESULTS_FILE
from src.utils.formatting import effect_size_label, fmt_mass, fmt_p, significance_stars


def segment_chemistry_profiles(
    cube_fg: np.ndarray,
    labels: np.ndarray,
    channel_labels: Sequence[str],
    segment_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Mean intensity per channel for each segment."""
    segments = sorted(set(labels))
    rows = {}
    for segment in segments:
        name = (segment_names or {}).get(segment, f"Segment {segment + 1}")
        rows[name] = cube_fg[labels == segment].mean(axis=0)
    return pd.DataFrame(rows, index=[fmt_mass(label) for label in channel_labels]).T


def kruskal_dunn_by_channel(
    cube_fg: np.ndarray,
    labels: np.ndarray,
    channel_labels: Sequence[str],
    p_adjust: str = "bonferroni",
    large_effect: float = 0.14,
    medium_effect: float = 0.06,
    p_floor: float = 1e-50,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Test every mass channel for differences between segments.

    Returns the omnibus results table and a per-channel dict of Dunn's post-hoc
    p-value matrices.
    """
    import scikit_posthocs as sp

    segments = sorted(set(labels))
    if len(segments) < 2:
        raise ValueError("Statistical validation needs at least two segments.")

    n_pixels = len(labels)
    results: List[Dict[str, Any]] = []
    posthoc: Dict[str, pd.DataFrame] = {}

    for index, channel in enumerate(channel_labels):
        values = cube_fg[:, index]
        groups = [values[labels == segment] for segment in segments]

        h_statistic, p_value = stats.kruskal(*groups)
        # Epsilon-squared: the standard effect size for Kruskal-Wallis.
        effect = float(h_statistic / (n_pixels - 1))

        frame = pd.DataFrame({"value": values, "segment": labels})
        posthoc[str(channel)] = sp.posthoc_dunn(
            frame, val_col="value", group_col="segment", p_adjust=p_adjust
        )

        results.append(
            {
                "channel": str(channel),
                "mass_label": fmt_mass(channel),
                "H_statistic": round(float(h_statistic), 2),
                "p_value": float(p_value),
                "p_display": fmt_p(p_value, floor=p_floor),
                "eta_squared": round(effect, 4),
                "significant": significance_stars(p_value),
                "effect_size": effect_size_label(effect, large_effect, medium_effect),
                "n_pixels": int(n_pixels),
                "n_segments": len(segments),
            }
        )

    return pd.DataFrame(results), posthoc


def colocalization(
    secondary_flat: np.ndarray,
    labels: np.ndarray,
    strong: float = 0.5,
    moderate: float = 0.3,
    segment_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Correlate a secondary-polarity channel against each segment's spatial footprint.

    Each segment is encoded as a binary membership map and correlated with the
    resampled secondary-mode intensity: a point-biserial correlation testing whether
    that ion concentrates where the segment sits.
    """
    rows = []
    for segment in sorted(set(labels)):
        name = (segment_names or {}).get(segment, f"Segment {segment + 1}")
        membership = (labels == segment).astype(float)
        r, p_value = stats.pearsonr(secondary_flat, membership)
        magnitude = abs(r)
        rows.append(
            {
                "segment": int(segment),
                "segment_name": name,
                "pearson_r": round(float(r), 4),
                "p_value": float(p_value),
                "p_display": fmt_p(p_value),
                "interpretation": (
                    "Strong colocalization"
                    if magnitude > strong
                    else "Moderate colocalization"
                    if magnitude > moderate
                    else "Weak / no colocalization"
                ),
            }
        )
    return pd.DataFrame(rows)


def endmember_correlations(
    secondary_flat: np.ndarray, abundances_fg: np.ndarray
) -> pd.DataFrame:
    """Correlate the secondary channel against each endmember's abundance."""
    rows = []
    for index in range(abundances_fg.shape[1]):
        r, p_value = stats.pearsonr(secondary_flat, abundances_fg[:, index])
        rows.append(
            {
                "endmember": index,
                "pearson_r": round(float(r), 4),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def strongest_effect_channel(results: pd.DataFrame) -> str:
    """The channel that separates the segments most strongly."""
    return results.loc[results["eta_squared"].idxmax(), "channel"]


def posthoc_to_long(posthoc: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten the per-channel Dunn matrices into one tidy table for saving."""
    rows = []
    for channel, matrix in posthoc.items():
        segments = list(matrix.columns)
        for i, first in enumerate(segments):
            for second in segments[i + 1:]:
                rows.append(
                    {
                        "channel": channel,
                        "segment_a": first,
                        "segment_b": second,
                        "p_value": float(matrix.loc[first, second]),
                        "p_display": fmt_p(matrix.loc[first, second]),
                    }
                )
    return pd.DataFrame(rows)


def save_statistics(
    processed_dir: Path,
    results: pd.DataFrame,
    posthoc: Dict[str, pd.DataFrame],
    coloc: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """Persist the statistical tables, post-hoc matrices included."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    results_path = processed_dir / STATISTICAL_RESULTS_FILE
    results.to_csv(results_path, index=False)
    written["results"] = results_path

    posthoc_path = processed_dir / POSTHOC_FILE
    posthoc_to_long(posthoc).to_csv(posthoc_path, index=False)
    written["posthoc"] = posthoc_path

    if coloc is not None and not coloc.empty:
        coloc_path = processed_dir / COLOCALIZATION_FILE
        coloc.to_csv(coloc_path, index=False)
        written["colocalization"] = coloc_path

    return written
