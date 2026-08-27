"""
metadata.py
Acquisition metadata recovered from filenames.

The instrument encodes polarity, beam current, raster geometry, shot count and m/z
into the filename. Rather than one brittle whole-filename regex, each field is
matched independently, so a file that omits a field (or a dataset that uses a
slightly different naming convention) still yields everything else instead of
failing outright. Patterns live in the `metadata.fields` config block.
"""
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_FIELD_PATTERNS: Dict[str, str] = {
    "polarity": r"(?:^|[\s_\-])(neg(?:ative)?|pos(?:itive)?)(?:[\s_\-]|$)",
    "mass": r"(\d+\.?\d*\s*±\s*\d+\.?\d*\s*u)",
    "shots": r"(?:^|[\s_\-])(\d+)sh(?:ots)?(?:[\s_\-]|$)",
    "current": r"(\d+(?:\.\d+)?pA)",
    "dimensions": r"(\d+x\d+x\d+pix)",
}

_INT_FIELDS = {"shots"}


def _normalise_polarity(value: Optional[str]) -> Optional[str]:
    """Collapse neg/negative/NEG to 'Neg' and pos/positive/POS to 'Pos'."""
    if not value:
        return None
    lowered = value.lower()
    if lowered.startswith("neg"):
        return "Neg"
    if lowered.startswith("pos"):
        return "Pos"
    return value


def extract_metadata(
    key: str,
    field_patterns: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Extract acquisition fields from a single filename/key.

    Fields that do not match are set to None. `overrides` maps a case-insensitive
    filename substring to values that replace whatever was parsed, which is how a
    dataset with an irregularly named file is corrected without touching code.
    """
    patterns = field_patterns or DEFAULT_FIELD_PATTERNS
    record: Dict[str, Any] = {"key": key}
    matched_fields = 0

    for field, pattern in patterns.items():
        match = re.search(pattern, key, flags=re.IGNORECASE)
        value: Any = match.group(1) if match else None
        if value is not None:
            matched_fields += 1
            if field in _INT_FIELDS:
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif field == "polarity":
                value = _normalise_polarity(value)
            elif isinstance(value, str):
                value = value.strip()
        record[field] = value

    for substring, values in (overrides or {}).items():
        if substring.lower() in key.lower():
            record.update(values)

    record["matched"] = matched_fields == len(patterns)
    return record


def build_metadata(
    keys: Iterable[str],
    field_patterns: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Extract metadata for every key, preserving input order."""
    return [extract_metadata(key, field_patterns, overrides) for key in keys]


def build_metadata_from_config(keys: Iterable[str], config) -> List[Dict[str, Any]]:
    """Extract metadata using the patterns and overrides declared in the config."""
    return build_metadata(
        keys,
        field_patterns=config.get("metadata.fields") or DEFAULT_FIELD_PATTERNS,
        overrides=config.get("metadata.overrides") or {},
    )


def metadata_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame indexed by image key."""
    frame = pd.DataFrame(records)
    if "key" in frame.columns:
        frame = frame.set_index("key", drop=False)
    return frame


def attach_shapes(
    records: List[Dict[str, Any]], images: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Record each image's pixel grid alongside its parsed metadata."""
    for record in records:
        image = images.get(record["key"])
        record["shape"] = tuple(image.shape) if image is not None else None
    return records
