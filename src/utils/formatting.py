"""
formatting.py
Small display helpers shared by the figure modules.

These were duplicated inline in several notebooks; they live here so every figure
labels masses and p-values the same way.
"""
import re
from typing import Optional

# "103.91326±0.02297u" -> mass 103.91326, uncertainty 0.02297
MASS_PATTERN = re.compile(r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*u")
# Any decimal number immediately followed by the ± sign.
MASS_ONLY_PATTERN = re.compile(r"(\d+\.\d+)\s*±")


def extract_mass(label: str) -> Optional[float]:
    """Return the numeric m/z from a mass label or key, or None if absent."""
    match = MASS_ONLY_PATTERN.search(str(label))
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+\.\d+)", str(label))
    return float(match.group(1)) if match else None


def fmt_mass(label: str, decimals: int = 2) -> str:
    """Format a mass label for a plot title: 'm/z 103.91'."""
    match = MASS_PATTERN.search(str(label))
    if match:
        return f"m/z {float(match.group(1)):.{decimals}f}"
    mass = extract_mass(label)
    if mass is not None:
        return f"m/z {mass:.{decimals}f}"
    return str(label)


def fmt_p(p_value: float, floor: float = 1e-50) -> str:
    """Format a p-value compactly, flooring unrepresentably small values."""
    if p_value is None:
        return "n/a"
    try:
        p_value = float(p_value)
    except (TypeError, ValueError):
        return str(p_value)
    if p_value != p_value:  # NaN
        return "n/a"
    if p_value < floor:
        return f"<{floor:.0e}"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def significance_stars(p_value: float) -> str:
    """Conventional significance markers."""
    if p_value is None or p_value != p_value:
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def effect_size_label(eta_squared: float, large: float = 0.14, medium: float = 0.06) -> str:
    """Cohen-style bucketing of an eta-squared effect size."""
    if eta_squared is None or eta_squared != eta_squared:
        return "n/a"
    if eta_squared > large:
        return "Large"
    if eta_squared > medium:
        return "Medium"
    return "Small"
