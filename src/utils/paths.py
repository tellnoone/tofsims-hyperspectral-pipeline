"""
paths.py
Project path resolution.

Every path in the pipeline is derived from PROJECT_ROOT, which is located relative
to this file rather than the current working directory. That way the same code works
when called from a notebook, from the CLI, or from a test runner.
"""
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_dir(relative: Union[str, Path], root: Optional[Path] = None) -> Path:
    """
    Resolve a config-supplied directory against the project root.

    Absolute paths are returned unchanged, so a user can point the pipeline at data
    living outside the repository. Relative paths are resolved case-tolerantly: a
    config saying ``data/raw`` still finds a folder named ``Data/raw`` on disk (and
    vice versa), which keeps Windows checkouts and Linux checkouts interchangeable.
    """
    root = root or PROJECT_ROOT
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate

    direct = root / candidate
    if direct.exists():
        return direct

    # Walk the relative parts, matching each segment case-insensitively.
    current = root
    for part in candidate.parts:
        if (current / part).exists():
            current = current / part
            continue
        match = next(
            (child for child in current.iterdir() if child.name.lower() == part.lower()),
            None,
        ) if current.is_dir() else None
        if match is None:
            # Nothing on disk matches; fall back to the literal path so the caller
            # can create it.
            return direct
        current = match
    return current


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
