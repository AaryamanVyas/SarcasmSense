"""Shared pytest fixtures.

Everything the test-suite needs is built **inside the test session**: a tiny
corpus, a tiny knowledge base, a tiny tokenizer, a tiny RoBERTa and a tiny
trained-shaped checkpoint. No research dataset, no HuggingFace download and no
network access are required, which is what lets the suite run in CI and on a
laptop in about a minute.

The fixture corpus is *synthetic test data*, clearly labelled as such. It exists
to exercise code paths (missing context, one/two/three turns, duplicates,
conflicting labels), not to measure anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import SarcasmRecord  # noqa: E402
from src.retrieval.knowledge_base import KnowledgeFact  # noqa: E402

# --------------------------------------------------------------------- corpus

#: Synthetic examples covering every structural case the pipeline must handle.
#: Labels are arbitrary; nothing is measured on this data.
FIXTURE_RECORDS: list[dict] = [
    {
        "id": "fixture:0", "text": "Oh great, another Monday morning meeting.",
        "context": [], "label": 1, "dataset": "fixture_news", "domain": "news",
        "conversation_id": "fixture:conv-a",
    },
    {
        "id": "fixture:1", "text": "The quarterly report was submitted on time.",
        "context": [], "label": 0, "dataset": "fixture_news", "domain": "news",
        "conversation_id": "fixture:conv-b",
    },
    {
        "id": "fixture:2", "text": "Perfect, exactly what I needed today.",
        "context": ["My car broke down on the motorway."], "label": 1,
        "dataset": "fixture_dialogue", "domain": "dialogue",
        "conversation_id": "fixture:conv-c",
    },
    {
        "id": "fixture:3", "text": "Thanks, that actually helps a lot.",
        "context": ["I sent you the fixed configuration file.", "Try it now."],
        "label": 0, "dataset": "fixture_dialogue", "domain": "dialogue",
        "conversation_id": "fixture:conv-d",
    },
    {
        "id": "fixture:4", "text": "Wonderful. Simply wonderful.",
        "context": ["The flight is delayed.", "By how long?", "Six hours."],
        "label": 1, "dataset": "fixture_reddit", "domain": "reddit",
        "conversation_id": "fixture:conv-e",
    },
    {
        "id": "fixture:5", "text": "I appreciate the quick turnaround on this.",
        "context": ["Finished the review this morning."], "label": 0,
        "dataset": "fixture_reddit", "domain": "reddit",
        "conversation_id": "fixture:conv-f",
    },
    {
        "id": "fixture:6", "text": "Yeah, brilliant plan. Nothing could go wrong.",
        "context": ["Let's deploy on Friday evening."], "label": 1,
        "dataset": "fixture_reddit", "domain": "reddit",
        "conversation_id": "fixture:conv-g",
    },
    {
        "id": "fixture:7", "text": "The results look consistent with last quarter.",
        "context": [], "label": 0, "dataset": "fixture_news", "domain": "news",
        "conversation_id": "fixture:conv-h",
    },
]


def _expand(records: list[dict], copies: int) -> list[dict]:
    """Repeat the fixture records so splits are non-degenerate.

    Both the target text *and* every context turn are made unique per copy.
    Sharing a context string across copies would (correctly) cause
    :func:`build_conversation_groups` to merge them into one cluster, which
    would make the fixture too coarse to test split proportions.
    """
    out: list[dict] = []
    for copy in range(copies):
        for base in records:
            row = dict(base)
            row["id"] = f"{base['id']}-{copy}"
            row["conversation_id"] = f"{base['conversation_id']}-{copy}"
            row["text"] = f"{base['text']} (variant {copy})" if copy else base["text"]
            row["context"] = [
                f"{turn} (variant {copy})" if copy else turn for turn in base["context"]
            ]
            out.append(row)
    return out


@pytest.fixture(scope="session")
def fixture_records() -> list[SarcasmRecord]:
    """A small, structurally varied corpus in the common schema."""
    return [SarcasmRecord.from_dict(row) for row in _expand(FIXTURE_RECORDS, copies=6)]


@pytest.fixture(scope="session")
def fixture_corpus_path(tmp_path_factory, fixture_records) -> Path:
    """Write the fixture corpus to a JSONL file and return its path."""
    directory = tmp_path_factory.mktemp("corpus")
    path = directory / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in fixture_records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    return path


# ----------------------------------------------------------------- raw inputs
@pytest.fixture()
def raw_dir(tmp_path) -> Path:
    """A ``data/raw``-shaped directory with one small file per adapter format."""
    root = tmp_path / "raw"

    figlang = root / "figlang2020_twitter"
    figlang.mkdir(parents=True)
    (figlang / "sarcasm_detection_shared_task_twitter_training.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"label": "SARCASM", "response": "Oh brilliant, more rain.",
                 "context": ["forecast says sun", "it is pouring"]},
                {"label": "NOT_SARCASM", "response": "The forecast was accurate today.",
                 "context": ["what is the weather like"]},
                {"label": "SARCASM", "response": "", "context": []},          # dropped: empty
                {"label": "UNKNOWN", "response": "something", "context": []},  # dropped: label
            ]
        ),
        encoding="utf-8",
    )

    mustard = root / "mustard"
    mustard.mkdir(parents=True)
    (mustard / "sarcasm_data.json").write_text(
        json.dumps(
            {
                "1_10": {
                    "utterance": "It's just a privilege to watch your mind at work.",
                    "speaker": "SHELDON",
                    "context": ["I never would have identified the fingerprints."],
                    "context_speakers": ["LEONARD"],
                    "sarcasm": True,
                    "show": "BBT",
                },
                "1_11": {
                    "utterance": "That is a reasonable interpretation.",
                    "speaker": "LEONARD",
                    "context": ["I never would have identified the fingerprints."],
                    "context_speakers": ["LEONARD"],
                    "sarcasm": False,
                    "show": "BBT",
                },
            }
        ),
        encoding="utf-8",
    )

    headlines = root / "news_headlines"
    headlines.mkdir(parents=True)
    (headlines / "Sarcasm_Headlines_Dataset_v2.json").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"is_sarcastic": 1, "headline": "man discovers new way to avoid eye contact",
                 "article_link": "https://www.theonion.com/x"},
                {"is_sarcastic": 0, "headline": "council approves library budget",
                 "article_link": "https://www.huffingtonpost.com/y"},
            ]
        ),
        encoding="utf-8",
    )

    sarc = root / "sarc"
    sarc.mkdir(parents=True)
    (sarc / "train-balanced-sarcasm.csv").write_text(
        "label,comment,author,subreddit,score,ups,downs,date,created_utc,parent_comment\n"
        '1,"Yes because that always works","u1","politics",3,3,0,2016-01,'
        '2016-01-01 00:00:00,"we should just ignore the problem"\n'
        '0,"That is a fair point","u2","science",5,5,0,2016-02,'
        '2016-02-01 00:00:00,"the study had a small sample"\n',
        encoding="utf-8",
    )
    return root


# ------------------------------------------------------------------ knowledge
@pytest.fixture(scope="session")
def fixture_facts() -> list[KnowledgeFact]:
    """A tiny knowledge base of structurally realistic facts.

    These strings imitate the *shape* of verbalised ConceptNet/WordNet tuples so
    the index and retriever can be tested; they are test fixtures, not a
    knowledge source, and the production index is always built from real dumps.
    """
    raw = [
        ("a meeting is a type of gathering", "IsA", "meeting", "gathering"),
        ("rain causes wetness", "Causes", "rain", "wetness"),
        ("a delay causes frustration", "Causes", "delay", "frustration"),
        ("delay carries a negative connotation", "HasSentiment", "delay", "negative"),
        ("great carries a positive connotation", "HasSentiment", "great", "positive"),
        ("wonderful is the opposite of terrible", "Antonym", "wonderful", "terrible"),
        ("a laptop is used for working", "UsedFor", "laptop", "working"),
        ("a crash causes data loss", "Causes", "crash", "data loss"),
        ("a presentation requires preparation", "HasPrerequisite", "presentation", "preparation"),
        ("monday is a type of weekday", "IsA", "monday", "weekday"),
        ("traffic jam causes lateness", "Causes", "traffic jam", "lateness"),
        ("deploying on friday can be prevented by planning", "ObstructedBy",
         "deploying on friday", "planning"),
    ]
    return [
        KnowledgeFact(
            id=f"fixture-{i}", text=text, source="FixtureKB",
            relation=relation, head=head, tail=tail,
        )
        for i, (text, relation, head, tail) in enumerate(raw)
    ]


@pytest.fixture(scope="session")
def sentence_encoder():
    """The deterministic hashing encoder -- no download, fully reproducible.

    The default 384-dimensional width is used so that a retriever rebuilt from a
    checkpoint's config (``encoder_backend: hashing``) matches this index exactly
    and the strict encoder check passes.
    """
    from src.retrieval.encoder import HashingEncoder

    return HashingEncoder(dimension=384)


@pytest.fixture(scope="session")
def index_dir(fixture_facts, sentence_encoder, tmp_path_factory) -> str:
    """Directory holding a built FAISS index over the fixture facts."""
    from src.retrieval.faiss_index import KnowledgeIndex

    index = KnowledgeIndex.build(
        fixture_facts, sentence_encoder, index_type="flat",
        show_progress=False, is_sample_index=True, build_notes="pytest fixture",
    )
    directory = tmp_path_factory.mktemp("index")
    index.save(directory)
    return str(directory)


@pytest.fixture(scope="session")
def knowledge_index(index_dir, sentence_encoder):
    """The index, loaded back from disk (exercising save/load in every test)."""
    from src.retrieval.faiss_index import KnowledgeIndex

    return KnowledgeIndex.load(index_dir, expected_encoder_name=sentence_encoder.name)


@pytest.fixture(scope="session")
def retriever(knowledge_index, sentence_encoder):
    """A retriever over the fixture index."""
    from src.retrieval.retriever import CommonsenseRetriever

    return CommonsenseRetriever(knowledge_index, sentence_encoder, top_k=3)


# ---------------------------------------------------------------- dev encoder
@pytest.fixture(scope="session")
def dev_encoder_dir(tmp_path_factory, fixture_records) -> str:
    """Build the tiny DEVELOPMENT/TEST-ONLY RoBERTa once per session."""
    from src.models.dev_assets import build_dev_assets

    directory = tmp_path_factory.mktemp("dev_encoder")
    build_dev_assets(
        output_dir=directory,
        corpus=[record.text for record in fixture_records] * 4,
        vocab_size=600,
        hidden_size=64,
    )
    return str(directory)


@pytest.fixture(scope="session")
def tokenizer(dev_encoder_dir):
    """A tokenizer backed by the dev encoder."""
    from src.models.tokenization import SarcasmTokenizer

    return SarcasmTokenizer(dev_encoder_dir, 32, 32, local_files_only=True)


@pytest.fixture(scope="session")
def model_config(dev_encoder_dir):
    """Full-model architecture config pointing at the dev encoder."""
    from src.utils.config import ModelConfig

    return ModelConfig(
        encoder_name=dev_encoder_dir,
        max_utterance_length=32,
        max_context_length=32,
        use_context=True,
        use_knowledge=True,
        fusion="cross_attention",
        num_attention_heads=4,
        classifier_hidden=32,
    )


@pytest.fixture(scope="session")
def model(model_config, knowledge_index):
    """A constructed (untrained) full model in eval mode."""
    from src.models.sarcasm_model import SarcasmModel

    built = SarcasmModel(
        model_config,
        knowledge_dim=knowledge_index.dimension,
        knowledge_slots=3,
        local_files_only=True,
    )
    built.eval()
    return built


@pytest.fixture(scope="session")
def experiment_config(dev_encoder_dir, fixture_corpus_path, index_dir, tmp_path_factory):
    """A complete :class:`Config` wired to the session fixtures."""
    from src.utils.config import (
        Config,
        DataConfig,
        ExperimentConfig,
        ExplainConfig,
        ModelConfig,
        RetrievalConfig,
        TrainingConfig,
    )

    output = tmp_path_factory.mktemp("experiment")
    return Config(
        experiment=ExperimentConfig(name="pytest", seed=42, output_dir=str(output)),
        model=ModelConfig(
            encoder_name=dev_encoder_dir,
            max_utterance_length=32,
            max_context_length=32,
            num_attention_heads=4,
            classifier_hidden=32,
        ),
        data=DataConfig(
            processed_path=str(fixture_corpus_path),
            splits_dir=str(output / "splits"),
            num_workers=0,
        ),
        retrieval=RetrievalConfig(index_dir=index_dir, top_k=3, encoder_backend="hashing"),
        training=TrainingConfig(epochs=1, batch_size=4, mixed_precision="off"),
        explain=ExplainConfig(lime_num_samples=40, shap_num_samples=24),
    )


@pytest.fixture(scope="session")
def checkpoint_dir(model, tokenizer, experiment_config, tmp_path_factory) -> str:
    """An untrained but structurally complete checkpoint directory."""
    directory = tmp_path_factory.mktemp("checkpoint")
    model.save_pretrained(directory, extra={"note": "pytest fixture -- untrained"})
    tokenizer.save_pretrained(directory)
    experiment_config.save(Path(directory) / "training_config.yaml")
    return str(directory)


@pytest.fixture(scope="session")
def predictor(model, tokenizer, retriever, experiment_config, checkpoint_dir):
    """An inference service built directly from the fixtures (no disk reload)."""
    from src.inference.predictor import SarcasmPredictor

    return SarcasmPredictor(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        config=experiment_config,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )


@pytest.fixture(scope="session")
def analysis_service(predictor, experiment_config):
    """The full analysis service (prediction + attribution + explanation)."""
    from src.inference.service import SarcasmAnalysisService

    return SarcasmAnalysisService(predictor, experiment_config)
