"""HTTP routes.

Endpoints
---------
``GET  /health``      liveness + readiness + degradation reason
``GET  /model/info``  architecture, retrieval and explainability metadata
``POST /predict``     classify + explain (the main endpoint)
``POST /explain``     explanation-focused alias; defaults to the SHAP audit path
``POST /retrieve``    retrieval only, for inspecting the knowledge base
``GET  /examples``    demonstration inputs for the UI's demo mode
``GET  /results``     saved experiment results for the dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.schemas.api import (
    DemoExample,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from backend.services.model_service import ModelService
from backend.settings import Settings, get_settings
from src.utils.logging_utils import REQUEST_ID, log_event, redact
from src.utils.paths import resolve

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """FastAPI dependency returning the process-wide model service."""
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:  # pragma: no cover - wired in create_app
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not configured.",
        )
    return service


def _require_loaded(model_service: ModelService):
    """Return the analysis service or raise a 503 with the real reason."""
    if not model_service.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                model_service.error
                or "Model is still loading. Retry shortly, or check GET /health."
            ),
        )
    return model_service.service


# --------------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Report whether the API can serve predictions, and why not if it cannot."""
    payload = model_service.health()
    return HealthResponse(version=settings.api_version, **payload)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info(model_service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    """Describe the served architecture, retrieval stack and explainers."""
    service = _require_loaded(model_service)
    described = service.describe()
    return ModelInfoResponse(
        model=described["model"],
        retrieval=described["retrieval"],
        explanation=described["explanation"],
        device=described["device"],
        checkpoint=described["checkpoint"],
        experiment=described["experiment"],
        fusion=described["fusion"],
        task=described["task"],
        input_modality=described["input_modality"],
        max_context_turns=described["max_context_turns"],
        is_development_asset=described["is_development_asset"],
        startup_warnings=described["startup_warnings"],
    )


# -------------------------------------------------------------------- predict
def _analyse(
    payload: PredictRequest,
    model_service: ModelService,
    settings: Settings,
) -> PredictResponse:
    """Shared implementation for ``/predict`` and ``/explain``."""
    service = _require_loaded(model_service)
    request_id = REQUEST_ID.get()
    if settings.log_request_text:
        logger.debug("analyse text=%s", redact(payload.text))

    try:
        result = service.analyze(
            text=payload.text,
            context=payload.context,
            explanation_method=payload.explanation_method,
            explain=payload.explain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Prediction failed: {type(exc).__name__}"
                + (f": {exc}" if settings.debug else "")
            ),
        )

    body = result.to_dict()
    log_event(
        logger,
        "prediction",
        request_id=request_id,
        label=body["label"],
        confidence=body["confidence"],
        retrieval_status=body["retrieval_status"],
        n_evidence=len(body["retrieved_evidence"]),
        explanation_method=body["explanation_method"],
        n_context_turns=len(body["context_used"]),
        text_length=len(payload.text),
        latency_ms=body["timings_ms"].get("total"),
    )
    body["request_id"] = request_id
    return PredictResponse(**body)


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    payload: PredictRequest,
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Classify an utterance and return prediction, evidence and explanation."""
    return _analyse(payload, model_service, settings)


@router.post("/explain", response_model=PredictResponse, tags=["inference"])
def explain(
    payload: PredictRequest,
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Explanation-focused endpoint.

    Identical to ``/predict`` except that ``explain`` is forced on. Use it with
    ``explanation_method="shap"`` for the slower audit-mode attribution.
    """
    payload = payload.model_copy(update={"explain": True})
    return _analyse(payload, model_service, settings)


@router.post("/retrieve", response_model=RetrieveResponse, tags=["inference"])
def retrieve(
    payload: RetrieveRequest,
    model_service: ModelService = Depends(get_model_service),
) -> RetrieveResponse:
    """Return the top-k commonsense facts for a query, without classifying."""
    service = _require_loaded(model_service)
    retriever = service.predictor.retriever
    request_id = REQUEST_ID.get()
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Commonsense retrieval is disabled for the served model.",
        )
    cleaned, turns = service.predictor.prepare(payload.text, payload.context)
    result = retriever.retrieve(cleaned, turns, top_k=payload.top_k)
    log_event(
        logger,
        "retrieval",
        request_id=request_id,
        status=result.status.value,
        n_evidence=len(result.evidence),
        latency_ms=result.latency_ms,
    )
    return RetrieveResponse(
        query=result.query,
        status=result.status.value,
        evidence=[e.to_dict() for e in result.evidence],  # type: ignore[arg-type]
        warning=result.warning,
        latency_ms=round(result.latency_ms, 2),
        request_id=request_id,
    )


# ----------------------------------------------------------------- demo mode
#: Demonstration inputs for the UI. These are **illustrative inputs only** --
#: they are not drawn from any split of any dataset and are never used to
#: compute a reported metric.
DEMO_EXAMPLES: list[DemoExample] = [
    DemoExample(
        id="demo-1",
        title="Positive wording, negative situation",
        text="Great, my laptop crashed five minutes before the presentation.",
        context=[],
        note=(
            "The classic incongruity pattern: an approving word attached to an "
            "undesirable event. Solvable without context."
        ),
    ),
    DemoExample(
        id="demo-2",
        title="Exaggerated praise for an unwanted event",
        text="Wow, what an amazing traffic jam. I only needed to get to work.",
        context=[],
        note="Tests whether the model uses commonsense about traffic jams being undesirable.",
    ),
    DemoExample(
        id="demo-3",
        title="Context-dependent -- sarcastic reading",
        text="Perfect timing, as always.",
        context=[
            "The delivery was supposed to arrive on Monday.",
            "It finally turned up this morning, nine days late.",
        ],
        note=(
            "The target alone is ambiguous. Only the preceding turns make the "
            "sarcastic reading available -- compare with the next example."
        ),
    ),
    DemoExample(
        id="demo-4",
        title="Context-dependent -- literal reading",
        text="Perfect timing, as always.",
        context=[
            "I was just about to call you about the report.",
            "I finished it ten minutes ago and sent it over.",
        ],
        note=(
            "Identical target utterance, opposite context. A context-blind model "
            "must give the same answer to both; a context-aware one should not."
        ),
    ),
    DemoExample(
        id="demo-5",
        title="Literal, no sarcasm",
        text="The train arrived exactly on schedule this morning.",
        context=[],
        note="A plain factual statement; a well-calibrated model should be confident here.",
    ),
    DemoExample(
        id="demo-6",
        title="News headline (no conversational context)",
        text="Local man discovers new way to avoid making eye contact on the bus.",
        context=[],
        note=(
            "Headline-style input exercising the empty-context path: the model must "
            "fall back to its learned no-context representation."
        ),
    ),
]


@router.get("/examples", response_model=list[DemoExample], tags=["demo"])
def examples() -> list[DemoExample]:
    """Return demonstration inputs for the UI's demo mode.

    These are illustrative inputs written for the demo; they are **not** test-set
    examples and no reported metric is computed from them.
    """
    return DEMO_EXAMPLES


# ----------------------------------------------------------------- dashboard
@router.get("/results", tags=["dashboard"])
def results(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return saved experiment results for the performance dashboard.

    Reads only what is already on disk. When no experiment has been run the
    response says so explicitly -- the dashboard renders "No experiment results
    available yet" rather than any placeholder number.
    """
    results_dir = resolve(settings.results_dir)
    payload: dict[str, Any] = {
        "available": False,
        "experiments": [],
        "ablation": None,
        "results_dir": str(results_dir),
    }

    experiments_path = results_dir / "experiments.json"
    if experiments_path.exists():
        try:
            runs = json.loads(experiments_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            logger.error("Could not parse %s: %s", experiments_path, exc)
            runs = []
        payload["experiments"] = [
            {
                "run_id": run.get("run_id"),
                "experiment": run.get("experiment"),
                "model_type": run.get("model_type"),
                "timestamp": run.get("timestamp"),
                "seed": run.get("seed"),
                "datasets": run.get("datasets", []),
                "is_development_asset": run.get("is_development_asset", False),
                "warnings": run.get("warnings", []),
                "val_metrics": run.get("val_metrics", {}),
                "test_metrics": run.get("test_metrics", {}),
                "per_dataset_metrics": run.get("per_dataset_metrics", {}),
                "train_seconds": run.get("train_seconds"),
            }
            for run in runs
        ]
        payload["available"] = bool(payload["experiments"])

    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        try:
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            ablation = None
        if ablation:
            payload["ablation"] = {
                "base_config": ablation.get("base_config"),
                "variants": [
                    {
                        "variant": v.get("variant"),
                        "description": v.get("description"),
                        "test": v.get("test", {}),
                        "train_seconds": v.get("train_seconds"),
                        "is_development_asset": v.get("is_development_asset", False),
                        "error": v.get("error"),
                    }
                    for v in ablation.get("variants", [])
                ],
                "explanation_agreement": (ablation.get("explanation_agreement") or {}),
            }
            payload["available"] = True

    benchmark_path = results_dir / "benchmark.json"
    if benchmark_path.exists():
        try:
            payload["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
            payload["available"] = True
        except json.JSONDecodeError:  # pragma: no cover
            pass

    dataset_report = results_dir / "dataset_report.json"
    if dataset_report.exists():
        try:
            report = json.loads(dataset_report.read_text(encoding="utf-8"))
            payload["dataset"] = {
                "n_records": report.get("n_records"),
                "merged_report": report.get("merged_report", {}),
                "split": report.get("split", {}),
                "leakage": report.get("leakage", {}),
            }
        except json.JSONDecodeError:  # pragma: no cover
            pass

    if not payload["available"]:
        payload["message"] = (
            "No experiment results available yet. Train a model with "
            "`python scripts/train.py --config configs/full_model.yaml`."
        )
    return payload
