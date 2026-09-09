"""Provenance metadata for every corpus the project supports.

This module is deliberately *only* declarative. It records, for each dataset:
the official public source, the licence, the citation, whether it can be
fetched automatically, and the exact filenames/directories the loader expects.

Keeping this separate from the loaders means ``scripts/download_datasets.py``
can print a complete, accurate acquisition report without importing any heavy
dependency, and the README can be generated from the same single source of
truth.

Nothing here downloads anything from an unofficial mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RemoteFile:
    """One file that can be fetched directly over HTTPS from an official repo."""

    url: str
    relative_path: str
    description: str = ""


@dataclass(frozen=True)
class DatasetSource:
    """Where a corpus comes from and what the loader expects on disk.

    Attributes:
        key: Stable dataset key, also used in ``SarcasmRecord.dataset``.
        name: Human-readable name.
        homepage: Canonical landing page for the dataset.
        licence: Licence as stated by the dataset authors.
        citation: BibTeX-ish citation string for the README/report.
        domain: Coarse genre tag applied to all its records.
        has_context: Whether the corpus carries conversational context.
        auto_downloadable: True when :attr:`remote_files` can be fetched
            directly from the authors' official repository without credentials.
        remote_files: Files fetched by ``scripts/download_datasets.py``.
        manual_files: Paths (relative to ``data/raw/<key>/``) the user must
            place by hand when :attr:`auto_downloadable` is False.
        manual_instructions: Precise, step-by-step acquisition instructions.
        expected_rows: Row count published by the dataset authors, used by the
            validator as a sanity check (``None`` when the authors do not
            publish an exact figure).
    """

    key: str
    name: str
    homepage: str
    licence: str
    citation: str
    domain: str
    has_context: bool
    auto_downloadable: bool
    remote_files: tuple[RemoteFile, ...] = ()
    manual_files: tuple[str, ...] = ()
    manual_instructions: str = ""
    expected_rows: int | None = None
    notes: str = ""


_FIGLANG_BASE = (
    "https://raw.githubusercontent.com/EducationalTestingService/sarcasm/master"
)

FIGLANG_TWITTER = DatasetSource(
    key="figlang2020_twitter",
    name="FigLang 2020 Shared Task -- Twitter",
    homepage="https://github.com/EducationalTestingService/sarcasm",
    licence="Released by the shared-task organisers for research use; see repository.",
    citation=(
        "Ghosh, D., Vajpayee, A., & Muresan, S. (2020). A Report on the 2020 "
        "Sarcasm Detection Shared Task. Proceedings of the Second Workshop on "
        "Figurative Language Processing (FigLang @ ACL 2020)."
    ),
    domain="twitter",
    has_context=True,
    auto_downloadable=True,
    remote_files=(
        RemoteFile(
            url=f"{_FIGLANG_BASE}/twitter/sarcasm_detection_shared_task_twitter_training.jsonl",
            relative_path="sarcasm_detection_shared_task_twitter_training.jsonl",
            description="5,000 labelled Twitter responses with conversation context.",
        ),
    ),
    expected_rows=5000,
    notes=(
        "The shared-task *test* files contain no gold labels (they were held out "
        "for the competition and never publicly released), so only the training "
        "split is usable as labelled data. This is why the FigLang contribution "
        "is ~9.4k rows rather than the ~13k quoted in the project proposal, "
        "which counted the unlabelled test rows."
    ),
)

FIGLANG_REDDIT = DatasetSource(
    key="figlang2020_reddit",
    name="FigLang 2020 Shared Task -- Reddit",
    homepage="https://github.com/EducationalTestingService/sarcasm",
    licence="Released by the shared-task organisers for research use; see repository.",
    citation=FIGLANG_TWITTER.citation,
    domain="reddit",
    has_context=True,
    auto_downloadable=True,
    remote_files=(
        RemoteFile(
            url=f"{_FIGLANG_BASE}/reddit/sarcasm_detection_shared_task_reddit_training.jsonl",
            relative_path="sarcasm_detection_shared_task_reddit_training.jsonl",
            description="4,400 labelled Reddit responses with conversation context.",
        ),
    ),
    expected_rows=4400,
    notes=FIGLANG_TWITTER.notes,
)

MUSTARD = DatasetSource(
    key="mustard",
    name="MUStARD (text-only subset)",
    homepage="https://github.com/soujanyaporia/MUStARD",
    licence="MIT (code and text annotations); video/audio not used by this project.",
    citation=(
        "Castro, S., Hazarika, D., Perez-Rosas, V., Zimmermann, R., Mihalcea, R., "
        "& Poria, S. (2019). Towards Multimodal Sarcasm Detection (An _Obviously_ "
        "Perfect Paper). ACL 2019."
    ),
    domain="dialogue",
    has_context=True,
    auto_downloadable=True,
    remote_files=(
        RemoteFile(
            url="https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json",
            relative_path="sarcasm_data.json",
            description="690 annotated sitcom utterances with speaker-tagged context.",
        ),
    ),
    expected_rows=690,
    notes=(
        "This project is TEXT ONLY: we consume the 'utterance' and 'context' "
        "fields and the speaker names, and ignore every audio/video artefact."
    ),
)

NEWS_HEADLINES = DatasetSource(
    key="news_headlines",
    name="News Headlines Dataset for Sarcasm Detection",
    homepage="https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection",
    licence="CC BY 4.0 (Rishabh Misra).",
    citation=(
        "Misra, R., & Arora, P. (2023). Sarcasm Detection using News Headlines "
        "Dataset. AI Open, 4, 13-18. "
        "Misra, R., & Grover, J. (2021). Sculpting Data for ML."
    ),
    domain="news",
    has_context=False,
    auto_downloadable=False,
    manual_files=("Sarcasm_Headlines_Dataset_v2.json",),
    manual_instructions=(
        "1. Open https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection\n"
        "2. Sign in to Kaggle and press 'Download' (Kaggle requires an account, so "
        "   this step cannot be automated without your credentials).\n"
        "3. Unzip the archive.\n"
        "4. Copy 'Sarcasm_Headlines_Dataset_v2.json' (28,619 rows, the recommended "
        "   version) into  data/raw/news_headlines/\n"
        "   The v1 file 'Sarcasm_Headlines_Dataset.json' (26,709 rows) is also "
        "   accepted if that is what you downloaded.\n"
        "5. Verify with:  python scripts/download_datasets.py --validate-only\n"
        "\n"
        "Alternatively, if you have the Kaggle CLI configured "
        "(pip install kaggle; ~/.kaggle/kaggle.json), run:\n"
        "   kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection "
        "-p data/raw/news_headlines --unzip"
    ),
    expected_rows=28619,
    notes=(
        "Headlines have no conversational context by construction, so every "
        "record is emitted with context=[]. This corpus is what makes the "
        "'gracefully handle missing context' requirement a real test rather "
        "than a hypothetical."
    ),
)

SARC = DatasetSource(
    key="sarc",
    name="SARC -- Self-Annotated Reddit Corpus",
    homepage="https://nlp.cs.princeton.edu/SARC/2.0/",
    licence=(
        "Released for research use by the authors; derived from public Reddit "
        "comments. Redistribution of the raw dump is not permitted here."
    ),
    citation=(
        "Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). A Large Self-Annotated "
        "Corpus for Sarcasm. LREC 2018."
    ),
    domain="reddit",
    has_context=True,
    auto_downloadable=False,
    manual_files=("train-balanced-sarcasm.csv",),
    manual_instructions=(
        "SARC is large (the full 2.0 dump is several GB) and is not redistributed "
        "by this repository. Two accepted layouts:\n"
        "\n"
        "OPTION A (recommended, 1.01M rows, single CSV with parent comments):\n"
        "  1. Open https://www.kaggle.com/datasets/danofer/sarcasm\n"
        "  2. Download and unzip.\n"
        "  3. Copy 'train-balanced-sarcasm.csv' into  data/raw/sarc/\n"
        "     Expected columns: label, comment, author, subreddit, score, ups, "
        "     downs, date, created_utc, parent_comment\n"
        "\n"
        "OPTION B (official Princeton release):\n"
        "  1. Open https://nlp.cs.princeton.edu/SARC/2.0/main/\n"
        "  2. Download 'train-balanced.csv.bz2' and 'comments.json.bz2'\n"
        "  3. Decompress both into  data/raw/sarc/\n"
        "     -> data/raw/sarc/train-balanced.csv\n"
        "     -> data/raw/sarc/comments.json\n"
        "\n"
        "Then run:  python scripts/download_datasets.py --validate-only\n"
        "SARC is balanced by construction; use --sarc-max-rows to subsample it "
        "so it does not dominate the ~50k target corpus."
    ),
    expected_rows=None,
    notes=(
        "SARC labels are self-annotations (the author's own '/s' marker), which "
        "makes them noisier than the expert annotations in MUStARD. Only the "
        "immediately preceding parent comment is available as context in "
        "Option A, so SARC records carry at most one context turn there."
    ),
)

#: Every supported corpus, keyed by :attr:`DatasetSource.key`.
DATASET_SOURCES: dict[str, DatasetSource] = {
    source.key: source
    for source in (
        FIGLANG_TWITTER,
        FIGLANG_REDDIT,
        MUSTARD,
        NEWS_HEADLINES,
        SARC,
    )
}

#: Corpora that this repository can fetch without credentials.
AUTO_DOWNLOADABLE: tuple[str, ...] = tuple(
    key for key, source in DATASET_SOURCES.items() if source.auto_downloadable
)

#: Corpora requiring a manual, licence-respecting download by the user.
MANUAL_ONLY: tuple[str, ...] = tuple(
    key for key, source in DATASET_SOURCES.items() if not source.auto_downloadable
)


def get_source(key: str) -> DatasetSource:
    """Look up a :class:`DatasetSource`, with a helpful error for typos."""
    try:
        return DATASET_SOURCES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset {key!r}. Known datasets: {sorted(DATASET_SOURCES)}"
        ) from exc
