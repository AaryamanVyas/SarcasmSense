#!/usr/bin/env python
"""End-to-end smoke test: is this installation actually working?

Runs a series of independent checks and prints a pass/fail table. It never
trains anything, so it finishes in well under a minute on a laptop.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --checkpoint checkpoints/full_model/best_model
    python scripts/smoke_test.py --strict     # non-zero exit on any warning

Checks
------
 1. Python dependencies importable, versions reported
 2. Project layout and configuration files parse
 3. Processed corpus + saved split present and consistent
 4. Tokenizer loads and round-trips text
 5. Knowledge index loads; a query returns real facts
 6. Model checkpoint loads; forward pass produces the right shapes
 7. Inference service classifies a demo input
 8. Explanations run (LIME / SHAP / occlusion)
 9. FastAPI app starts and /health, /model/info, /predict respond
10. Frontend build artefacts / dependencies are present

Checks degrade to WARN (not FAIL) when an optional artefact is simply missing --
for example, no trained checkpoint yet. Missing *code* or broken behaviour is a
FAIL.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import project_root, resolve  # noqa: E402

RULE = "=" * 78
PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
SYMBOL = {PASS: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}


@dataclass
class CheckResult:
    """Outcome of one smoke check."""

    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class SmokeRunner:
    """Runs checks, catching exceptions so one failure does not stop the rest."""

    def __init__(self, verbose: bool = False) -> None:
        self.results: list[CheckResult] = []
        self.verbose = verbose

    def run(self, name: str, fn: Callable[[], tuple[str, str]]) -> CheckResult:
        """Execute one check and record its result."""
        started = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a failing check is data, not a crash
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()
        result = CheckResult(
            name=name, status=status, detail=detail, seconds=time.perf_counter() - started
        )
        self.results.append(result)
        print(f"  {SYMBOL[result.status]} {name:<44} {result.status:<5} {result.detail}")
        return result

    @property
    def failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self.results if r.status == FAIL)

    @property
    def warned(self) -> int:
        """Number of warnings."""
        return sum(1 for r in self.results if r.status == WARN)


# --------------------------------------------------------------------- checks
def check_dependencies() -> tuple[str, str]:
    """Every runtime dependency must import."""
    import importlib

    required = [
        "torch", "transformers", "tokenizers", "numpy", "sklearn",
        "faiss", "fastapi", "pydantic", "yaml", "matplotlib",
    ]
    optional = ["sentence_transformers", "shap", "lime", "nltk", "seaborn"]
    missing = [name for name in required if not _importable(name)]
    if missing:
        return FAIL, f"missing required packages: {missing} -- pip install -r requirements.txt"
    missing_optional = [name for name in optional if not _importable(name)]
    import torch

    version = f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
    if missing_optional:
        return WARN, f"{version}; optional packages missing: {missing_optional}"
    return PASS, version


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def check_layout() -> tuple[str, str]:
    """Core directories and configs must exist and parse."""
    from src.utils.config import Config

    root = project_root()
    required_dirs = ["src", "scripts", "configs", "backend", "frontend", "tests"]
    missing = [d for d in required_dirs if not (root / d).is_dir()]
    if missing:
        return FAIL, f"missing directories: {missing}"
    configs = sorted((root / "configs").glob("*.yaml"))
    if not configs:
        return FAIL, "no YAML configs found in configs/"
    for path in configs:
        Config.from_yaml(path)
    return PASS, f"{len(configs)} config(s) parsed"


def check_corpus() -> tuple[str, str]:
    """Processed corpus and split must be present and mutually consistent."""
    from src.data.schema import SarcasmRecord
    from src.data.splitting import apply_split, corpus_fingerprint, load_split
    from src.utils.io import read_jsonl

    corpus_path = resolve("data/processed/dataset.jsonl")
    if not corpus_path.exists():
        return WARN, "no processed corpus -- run scripts/prepare_datasets.py"
    records = [SarcasmRecord.from_dict(row) for row in read_jsonl(corpus_path)]
    if not records:
        return FAIL, "processed corpus is empty"
    try:
        assignment = load_split("data/splits", expected_fingerprint=corpus_fingerprint(records))
        splits = apply_split(records, assignment)
    except Exception as exc:  # noqa: BLE001
        return WARN, f"split unusable ({exc}); re-run scripts/prepare_datasets.py"
    sizes = " ".join(f"{k}={len(v):,}" for k, v in splits.items())
    return PASS, f"{len(records):,} records; {sizes}"


def check_preprocessing() -> tuple[str, str]:
    """Preprocessing must preserve sarcasm cues and drop noise."""
    from src.data.preprocessing import clean_text, coerce_label

    cleaned = clean_text("Oh GREAT!!!!!!!! visit https://x.co @bob &amp; co")
    problems = []
    if "GREAT" not in cleaned:
        problems.append("capitalisation was destroyed")
    if "!!!!" not in cleaned:
        problems.append("repeated punctuation was destroyed")
    if "<url>" not in cleaned:
        problems.append("URL was not replaced")
    if "@user" not in cleaned:
        problems.append("mention was not replaced")
    if "&" not in cleaned:
        problems.append("HTML entity was not decoded")
    if coerce_label("SARCASM") != 1 or coerce_label("NOT_SARCASM") != 0:
        problems.append("label coercion is wrong")
    if problems:
        return FAIL, "; ".join(problems)
    return PASS, f"'{cleaned}'"


def check_tokenizer(encoder: str) -> tuple[str, str]:
    """Tokenizer must load and produce correctly-shaped tensors."""
    from src.models.tokenization import SarcasmTokenizer

    tokenizer = SarcasmTokenizer(encoder, 64, 64, local_files_only=_is_local(encoder))
    batch = tokenizer.encode(
        ["Great, another Monday.", "The report is finished."],
        [["how was your weekend?"], []],
    )
    shape = tuple(batch["utterance_input_ids"].shape)
    if shape != (2, 64):
        return FAIL, f"unexpected utterance shape {shape}"
    if batch["has_context"].tolist() != [True, False]:
        return FAIL, "has_context flag is wrong"
    return PASS, f"utterance {shape}, context {tuple(batch['context_input_ids'].shape)}"


def check_knowledge_index() -> tuple[str, str]:
    """A knowledge index must load and return real facts for a query."""
    from src.retrieval.encoder import build_sentence_encoder
    from src.retrieval.faiss_index import KnowledgeIndex, index_exists
    from src.retrieval.retriever import CommonsenseRetriever

    for directory in ("knowledge/index", "knowledge/index_sample"):
        if index_exists(directory):
            break
    else:
        return WARN, "no index -- run scripts/build_knowledge_index.py --sample"

    metadata_encoder = KnowledgeIndex.load(directory).metadata.encoder_name
    encoder = build_sentence_encoder(
        backend="hashing" if metadata_encoder.startswith("hashing") else "auto",
        model_name=metadata_encoder,
    )
    index = KnowledgeIndex.load(directory, expected_encoder_name=encoder.name, strict=False)
    retriever = CommonsenseRetriever(index, encoder, top_k=3)
    result = retriever.retrieve("my laptop crashed before the presentation")
    if not result.evidence:
        return WARN, f"index loaded ({len(index):,} facts) but returned no evidence"
    note = " [SAMPLE INDEX]" if index.metadata.is_sample_index else ""
    return PASS, f"{len(index):,} facts, top score {result.evidence[0].score:.3f}{note}"


def check_model_forward(encoder: str) -> tuple[str, str]:
    """A model must build and produce finite logits, including in degraded mode."""
    import torch

    from src.models.sarcasm_model import SarcasmModel
    from src.models.tokenization import SarcasmTokenizer
    from src.utils.config import ModelConfig

    config = ModelConfig(
        encoder_name=encoder, max_utterance_length=32, max_context_length=32
    )
    local = _is_local(encoder)
    model = SarcasmModel(config, knowledge_dim=384, knowledge_slots=5, local_files_only=local)
    model.eval()
    tokenizer = SarcasmTokenizer(encoder, 32, 32, local_files_only=local)
    batch = tokenizer.encode(["Great, just great."], [["everything broke"]])

    with torch.no_grad():
        full = model(
            **batch,
            knowledge_vectors=torch.randn(1, 5, 384),
            knowledge_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        degraded = model(
            batch["utterance_input_ids"],
            batch["utterance_attention_mask"],
            batch["context_input_ids"],
            batch["context_attention_mask"],
            has_context=torch.zeros(1, dtype=torch.bool),
            knowledge_vectors=torch.zeros(1, 5, 384),
            knowledge_mask=torch.zeros(1, 5, dtype=torch.bool),
        )
    if tuple(full.logits.shape) != (1, 2):
        return FAIL, f"unexpected logits shape {tuple(full.logits.shape)}"
    if not torch.isfinite(full.logits).all() or not torch.isfinite(degraded.logits).all():
        return FAIL, "non-finite logits (NaN/inf) -- masking bug"
    if full.fusion_attention is None:
        return FAIL, "cross-attention returned no attention weights"
    return PASS, (
        f"logits {tuple(full.logits.shape)}, attention {tuple(full.fusion_attention.shape)}, "
        f"head input {model.classifier_input_size}"
    )


def check_service(checkpoint: str) -> tuple[str, str]:
    """The full inference service must classify a demo input."""
    from src.inference.service import SarcasmAnalysisService

    if not (resolve(checkpoint) / "model_config.json").exists():
        return SKIP, f"no checkpoint at {checkpoint}"
    service = SarcasmAnalysisService.from_checkpoint(checkpoint, device="cpu")
    result = service.analyze(
        "Great, my laptop crashed five minutes before the presentation.",
        ["I have to present the quarterly results at 3pm."],
        explanation_method="occlusion",
    )
    body = result.to_dict()
    if body["prediction"] not in ("Sarcastic", "Not Sarcastic"):
        return FAIL, f"unexpected prediction {body['prediction']!r}"
    if not 0.0 <= body["confidence"] <= 1.0:
        return FAIL, f"confidence out of range: {body['confidence']}"
    return PASS, (
        f"{body['prediction']} ({body['confidence']:.0%}), "
        f"{len(body['retrieved_evidence'])} evidence, "
        f"{body['timings_ms']['total']:.0f} ms"
    )


def check_explanations(checkpoint: str) -> tuple[str, str]:
    """Every explanation method must produce attributions or a clear reason."""
    from src.inference.service import SarcasmAnalysisService

    if not (resolve(checkpoint) / "model_config.json").exists():
        return SKIP, f"no checkpoint at {checkpoint}"
    service = SarcasmAnalysisService.from_checkpoint(checkpoint, device="cpu")
    notes = []
    for method in ("occlusion", "lime", "shap"):
        result = service.analyze("Oh perfect, another delay.", [], explanation_method=method)
        n = len(result.explanation.tokens)
        actual = result.explanation.method
        if n == 0:
            return FAIL, f"{method} produced no attributions"
        notes.append(f"{method}->{actual}:{n}tok/{result.explanation.latency_ms:.0f}ms")
    return PASS, " ".join(notes)


def check_api(checkpoint: str) -> tuple[str, str]:
    """The FastAPI app must start and answer its core endpoints."""
    from fastapi.testclient import TestClient

    from backend.main import create_app
    from backend.settings import Settings

    has_checkpoint = (resolve(checkpoint) / "model_config.json").exists()
    settings = Settings(
        model_path=checkpoint,
        load_model_on_startup=has_checkpoint,
        log_level="ERROR",  # create_app reconfigures logging; keep the table readable
    )
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            return FAIL, f"/health returned {health.status_code}"
        if not has_checkpoint:
            return WARN, "API starts, but no checkpoint to serve (train a model first)"
        info = client.get("/model/info")
        if info.status_code != 200:
            return FAIL, f"/model/info returned {info.status_code}"
        response = client.post(
            "/predict",
            json={"text": "Great, another meeting.", "context": [], "explanation_method": "occlusion"},
        )
        if response.status_code != 200:
            return FAIL, f"/predict returned {response.status_code}: {response.text[:180]}"
        body = response.json()
        for key in ("prediction", "confidence", "explanation", "retrieved_evidence", "token_importance"):
            if key not in body:
                return FAIL, f"/predict response is missing {key!r}"
        bad = client.post("/predict", json={"text": "  "})
        if bad.status_code != 422:
            return FAIL, f"empty text should be rejected with 422, got {bad.status_code}"
    return PASS, f"/health /model/info /predict ok; validation rejects empty input"


def check_frontend() -> tuple[str, str]:
    """Frontend sources must exist; report whether it is installed/built."""
    root = project_root() / "frontend"
    if not (root / "package.json").exists():
        return FAIL, "frontend/package.json is missing"
    if not (root / "src" / "App.jsx").exists():
        return FAIL, "frontend/src/App.jsx is missing"
    installed = (root / "node_modules").is_dir()
    built = (root / "dist" / "index.html").exists()
    if built:
        return PASS, "sources present, production build in frontend/dist"
    if installed:
        return PASS, "sources present, dependencies installed (run `npm run build` to bundle)"
    return WARN, "sources present but dependencies not installed -- run `npm install` in frontend/"


def _is_local(encoder: str) -> bool:
    """Whether ``encoder`` points at a directory on disk rather than a hub id."""
    return Path(encoder).exists() or (project_root() / encoder).exists()


# ----------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to exercise. Default: dev_smoke if present, else full_model.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="Encoder for the tokenizer/forward checks. Default: dev encoder if present.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings too.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 when everything passed."""
    args = parse_args(argv)
    configure_logging("ERROR")

    encoder = args.encoder
    if encoder is None:
        dev = project_root() / "checkpoints" / "dev_encoder"
        encoder = str(dev) if (dev / "config.json").exists() else "roberta-base"

    checkpoint = args.checkpoint
    if checkpoint is None:
        for candidate in ("checkpoints/full_model/best_model", "checkpoints/dev_smoke/best_model"):
            if (resolve(candidate) / "model_config.json").exists():
                checkpoint = candidate
                break
        checkpoint = checkpoint or "checkpoints/full_model/best_model"

    print(f"\n{RULE}\nSMOKE TEST\n{RULE}")
    print(f"  project   : {project_root()}")
    print(f"  encoder   : {encoder}")
    print(f"  checkpoint: {checkpoint}\n")

    runner = SmokeRunner(verbose=args.verbose)
    runner.run("1. dependencies", check_dependencies)
    runner.run("2. project layout & configs", check_layout)
    runner.run("3. corpus & splits", check_corpus)
    runner.run("4. preprocessing", check_preprocessing)
    runner.run("5. tokenizer", lambda: check_tokenizer(encoder))
    runner.run("6. knowledge index & retrieval", check_knowledge_index)
    runner.run("7. model forward pass", lambda: check_model_forward(encoder))
    runner.run("8. inference service", lambda: check_service(checkpoint))
    runner.run("9. explanations (LIME/SHAP/occlusion)", lambda: check_explanations(checkpoint))
    runner.run("10. FastAPI endpoints", lambda: check_api(checkpoint))
    runner.run("11. frontend", check_frontend)

    total_seconds = sum(r.seconds for r in runner.results)
    print(f"\n{RULE}")
    print(
        f"  {len(runner.results)} checks in {total_seconds:.1f}s | "
        f"{sum(1 for r in runner.results if r.status == PASS)} pass, "
        f"{runner.warned} warn, {runner.failed} fail, "
        f"{sum(1 for r in runner.results if r.status == SKIP)} skip"
    )
    print(RULE)

    if runner.failed:
        print("\n  SMOKE TEST FAILED. Fix the ✗ checks above before continuing.")
        return 1
    if runner.warned and args.strict:
        print("\n  Warnings present and --strict was set.")
        return 2
    print("\n  Smoke test passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
