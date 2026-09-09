"""Dataset acquisition helpers.

Only *official* sources are ever contacted:

* FigLang 2020 -> ``raw.githubusercontent.com/EducationalTestingService/sarcasm``
* MUStARD      -> ``raw.githubusercontent.com/soujanyaporia/MUStARD``

News-Headlines and SARC are **not** downloaded automatically: they are
distributed through Kaggle / a Princeton host that require an account or a
multi-GB transfer, so this module refuses to guess at unofficial mirrors and
instead reports precise manual instructions (see
:attr:`~src.data.sources.DatasetSource.manual_instructions`).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from src.data.sources import DatasetSource, RemoteFile
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass
class DownloadResult:
    """Outcome of fetching one :class:`RemoteFile`."""

    dataset: str
    path: Path
    status: str  # "downloaded" | "cached" | "failed"
    bytes_written: int = 0
    error: str | None = None


def download_file(
    remote: RemoteFile,
    target_dir: Path,
    dataset_key: str,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> DownloadResult:
    """Fetch one remote file into ``target_dir`` atomically.

    The download goes to a temporary file first and is only moved into place on
    success, so an interrupted run can never leave a truncated dataset behind
    that a later validation would happily accept.
    """
    destination = target_dir / remote.relative_path
    if destination.exists() and not force:
        return DownloadResult(
            dataset=dataset_key,
            path=destination,
            status="cached",
            bytes_written=destination.stat().st_size,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", remote.url, destination)
    tmp_path: Path | None = None
    try:
        with requests.get(remote.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                delete=False, dir=destination.parent, suffix=".part"
            ) as tmp:
                tmp_path = Path(tmp.name)
                written = 0
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        tmp.write(chunk)
                        written += len(chunk)
        if written == 0:
            raise OSError("remote returned an empty body")
        shutil.move(str(tmp_path), destination)
        tmp_path = None
        return DownloadResult(
            dataset=dataset_key,
            path=destination,
            status="downloaded",
            bytes_written=written,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        logger.error("Failed to download %s: %s", remote.url, exc)
        return DownloadResult(
            dataset=dataset_key, path=destination, status="failed", error=str(exc)
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def download_dataset(
    source: DatasetSource,
    raw_dir: str | Path = "data/raw",
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[DownloadResult]:
    """Download every auto-fetchable file of ``source``.

    Returns an empty list for manual-only datasets; callers should print
    :attr:`DatasetSource.manual_instructions` in that case.
    """
    if not source.auto_downloadable:
        return []
    target_dir = ensure_dir(Path(raw_dir) / source.key)
    return [
        download_file(remote, target_dir, source.key, force=force, timeout=timeout)
        for remote in source.remote_files
    ]
