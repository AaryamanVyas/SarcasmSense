"""Dataset adapter registry.

Adding a new corpus to the project is a three-step operation:

1. declare its provenance in :mod:`src.data.sources`,
2. write an adapter subclassing :class:`~src.data.loaders.base.DatasetAdapter`,
3. register it in :data:`ADAPTERS` below.

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders.base import DatasetAdapter, LoadStats
from src.data.loaders.figlang import FigLangRedditAdapter, FigLangTwitterAdapter
from src.data.loaders.headlines import NewsHeadlinesAdapter
from src.data.loaders.mustard import MUStARDAdapter
from src.data.loaders.sarc import SARCAdapter
from src.data.preprocessing import PreprocessingOptions

#: Every registered adapter class, keyed by dataset key.
ADAPTERS: dict[str, type[DatasetAdapter]] = {
    FigLangTwitterAdapter.source.key: FigLangTwitterAdapter,
    FigLangRedditAdapter.source.key: FigLangRedditAdapter,
    MUStARDAdapter.source.key: MUStARDAdapter,
    NewsHeadlinesAdapter.source.key: NewsHeadlinesAdapter,
    SARCAdapter.source.key: SARCAdapter,
}

ALL_DATASET_KEYS: tuple[str, ...] = tuple(ADAPTERS)


def build_adapter(
    key: str,
    raw_dir: str | Path = "data/raw",
    options: PreprocessingOptions | None = None,
    max_context_turns: int = 3,
    max_rows: int | None = None,
    seed: int = 42,
) -> DatasetAdapter:
    """Instantiate the adapter registered under ``key``.

    Args:
        key: Registered dataset key.
        raw_dir: Root of the raw data tree.
        options: Preprocessing switches.
        max_context_turns: Truncate context to the most recent N turns.
        max_rows: Cap the corpus by seeded random subsampling (not a head slice).
        seed: Controls that subsample.

    Raises:
        KeyError: if ``key`` is not a registered dataset.
    """
    try:
        adapter_cls = ADAPTERS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset {key!r}. Registered datasets: {sorted(ADAPTERS)}"
        ) from exc
    return adapter_cls(
        raw_dir=raw_dir,
        options=options,
        max_context_turns=max_context_turns,
        max_rows=max_rows,
        seed=seed,
    )


__all__ = [
    "ADAPTERS",
    "ALL_DATASET_KEYS",
    "DatasetAdapter",
    "FigLangRedditAdapter",
    "FigLangTwitterAdapter",
    "LoadStats",
    "MUStARDAdapter",
    "NewsHeadlinesAdapter",
    "SARCAdapter",
    "build_adapter",
]
