"""Small, dependency-light IO helpers (JSON, JSONL, environment capture)."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.utils.paths import ensure_parent, project_root, resolve

logger = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line of a JSONL file."""
    resolved = resolve(path)
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_no}: invalid JSON ({exc})") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write ``rows`` as JSONL; returns the number of records written."""
    resolved = ensure_parent(path)
    count = 0
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Any:
    """Load a JSON document."""
    with resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write a JSON document, creating parent directories as needed."""
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    return resolved


def git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def package_versions(packages: Iterable[str] | None = None) -> dict[str, str]:
    """Return installed versions of the packages that affect reproducibility."""
    import importlib.metadata as metadata

    names = list(
        packages
        or (
            "torch",
            "transformers",
            "tokenizers",
            "sentence-transformers",
            "faiss-cpu",
            "numpy",
            "scikit-learn",
            "pandas",
            "shap",
            "lime",
            "fastapi",
        )
    )
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def environment_snapshot() -> dict[str, Any]:
    """Capture everything needed to reproduce a run later."""
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "git_commit": git_commit(),
        "packages": package_versions(),
    }
    try:
        import torch

        snapshot["torch_cuda_available"] = bool(torch.cuda.is_available())
        snapshot["torch_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            snapshot["cuda_device_name"] = torch.cuda.get_device_name(0)
            snapshot["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        snapshot["torch_cuda_available"] = False
    return snapshot
