"""
stack_io.py
Persistence for the stage-01 artefacts.

The clean stack is stored as a dict-style .npz keyed by filename stem rather than a
single cube, because a dataset can legitimately contain images on different pixel
grids (e.g. a wide-field positive-mode scan alongside square negative-mode maps).
Downstream stages select and stack the subset they need.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import CLEAN_STACK_FILE, METADATA_FILE, SUMMARY_STATS_FILE


def save_clean_stack(
    images: Dict[str, np.ndarray],
    metadata: List[Dict[str, Any]],
    processed_dir: Path,
) -> Dict[str, Path]:
    """Write the image stack, metadata JSON and per-image summary statistics."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stack_path = processed_dir / CLEAN_STACK_FILE
    np.savez_compressed(stack_path, **images)

    metadata_path = processed_dir / METADATA_FILE
    serialisable = [_jsonify(record) for record in metadata]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2, ensure_ascii=False)

    summary_path = processed_dir / SUMMARY_STATS_FILE
    summary_statistics(images).to_csv(summary_path, index=False)

    return {"stack": stack_path, "metadata": metadata_path, "summary": summary_path}


def load_clean_stack(processed_dir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Load the stage-01 artefacts written by `save_clean_stack`."""
    processed_dir = Path(processed_dir)
    stack_path = processed_dir / CLEAN_STACK_FILE
    metadata_path = processed_dir / METADATA_FILE

    if not stack_path.exists():
        raise FileNotFoundError(
            f"{stack_path} not found. Run the 'load' stage first "
            f"(python scripts/run_pipeline.py --stage load)."
        )

    with np.load(stack_path, allow_pickle=False) as archive:
        images = {key: archive[key] for key in archive.files}

    metadata: List[Dict[str, Any]] = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return images, metadata


def summary_statistics(images: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-image intensity summary used for the stage-01 sanity check table."""
    rows = []
    for key, image in images.items():
        finite = np.isfinite(image)
        rows.append(
            {
                "image": key,
                "shape": f"{image.shape[0]}x{image.shape[1]}",
                "min": float(np.min(image)),
                "max": float(np.max(image)),
                "mean": float(np.mean(image)),
                "median": float(np.median(image)),
                "std": float(np.std(image)),
                "zeros_pct": float(np.count_nonzero(image == 0) / image.size * 100),
                "nonzero_pixels": int(np.count_nonzero(image)),
                "all_finite": bool(finite.all()),
                "non_negative": bool(np.min(image) >= 0),
            }
        )
    return pd.DataFrame(rows)


def _jsonify(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy scalars and tuples so json.dump accepts the record."""
    clean: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            clean[key] = bool(value)
        elif isinstance(value, tuple):
            clean[key] = list(value)
        else:
            clean[key] = value
    return clean
