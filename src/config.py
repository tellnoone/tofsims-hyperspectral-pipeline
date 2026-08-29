"""
config.py
Loading and access for configs/pipeline_config.yaml.

The Config object is passed down through every stage. It resolves directories once,
and exposes `get("a.b.c")` so modules can read nested settings without defensive
dictionary juggling.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.utils.constants import DEFAULT_CONFIG_PATH
from src.utils.paths import PROJECT_ROOT, ensure_dir, resolve_dir

AUTO = "auto"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Thin wrapper around the parsed YAML with resolved paths."""

    def __init__(self, data: Dict[str, Any], config_path: Optional[Path] = None):
        self.data = data
        self.config_path = config_path
        self.project_root = PROJECT_ROOT

        self.raw_dir = resolve_dir(self.get("dataset.raw_dir", "data/raw"))
        self.processed_dir = resolve_dir(self.get("dataset.processed_dir", "data/processed"))
        figures = self.get("dataset.figures_dir") or self.get("dataset.processed_dir")
        self.figures_dir = resolve_dir(figures)

    @property
    def random_seed(self) -> int:
        """
        The one seed every stochastic estimator uses.

        Falls back to the older decomposition.random_state key so existing config
        files keep working.
        """
        value = self.get("random_seed")
        if value is None:
            value = self.get("decomposition.random_state", 42)
        return int(value)

    # ------------------------------------------------------------------ access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested value, e.g. get('decomposition.nmf.max_iter')."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def is_auto(self, dotted_key: str, default: Any = AUTO) -> bool:
        value = self.get(dotted_key, default)
        return isinstance(value, str) and value.lower() == AUTO

    def section(self, dotted_key: str) -> Dict[str, Any]:
        value = self.get(dotted_key, {})
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------ paths
    def prepare_output_dirs(self) -> None:
        ensure_dir(self.processed_dir)
        ensure_dir(self.figures_dir)

    def processed_path(self, filename: str) -> Path:
        return self.processed_dir / filename

    def figure_path(self, filename: str) -> Path:
        return self.figures_dir / filename

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(raw_dir={self.raw_dir}, processed_dir={self.processed_dir})"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load the pipeline configuration.

    `overrides` is a nested dict merged over the file contents, used by the CLI to
    apply flags such as --raw-dir without mutating the YAML on disk.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(data, config_path=path)
