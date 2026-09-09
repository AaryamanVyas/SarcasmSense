"""Project path resolution.

All modules resolve paths through here so that nothing is hard-coded to a
developer's home directory. Relative paths in YAML configs are interpreted
relative to the project root (the directory containing ``src/``), unless the
``SARCASM_PROJECT_ROOT`` environment variable overrides it -- which is what the
Docker image does.
"""

from __future__ import annotations

import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def project_root() -> Path:
    """Return the repository root directory."""
    env = os.getenv("SARCASM_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/utils/paths.py -> src/utils -> src -> <root>
    return _THIS_FILE.parents[2]


def resolve(path: str | os.PathLike[str] | None) -> Path:
    """Resolve ``path`` against the project root when it is relative."""
    if path is None:
        raise ValueError("resolve() received None")
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (project_root() / candidate).resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return the resolved dir."""
    resolved = resolve(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    """Create the parent directory of ``path`` and return the resolved file path."""
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
