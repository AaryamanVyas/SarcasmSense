"""Pydantic request/response models -- the API contract.

Validation happens here rather than in the route handlers so that FastAPI can
reject malformed input before any model code runs, and so that ``/docs`` shows an
accurate, self-documenting schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.settings import get_settings

ExplanationMethod = Literal["lime", "shap", "occlusion"]


class PredictRequest(BaseModel):
    """Body of ``POST /predict`` and ``POST /explain``."""

    text: str = Field(
        ...,
        description="The target utterance to classify.",
        json_schema_extra={"example": "Great, my laptop crashed five minutes before the presentation."},
    )
    context: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 preceding conversational turns, oldest first. "
            "Send an empty list for headlines or standalone posts."
        ),
        json_schema_extra={"example": ["I have to present the quarterly results at 3pm."]},
    )
    explanation_method: ExplanationMethod = Field(
        default="lime",
        description=(
            "'lime' or 'occlusion' for the fast interactive path; 'shap' for the "
            "slower audit path."
        ),
    )
    explain: bool = Field(
        default=True,
        description="Set false to skip token attribution entirely (fastest path).",
    )
    top_k: int | None = Field(
        default=None, ge=0, le=20, description="Override the number of retrieved facts."
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        settings = get_settings()
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only")
        if len(value) > settings.max_text_length:
            raise ValueError(
                f"text must be at most {settings.max_text_length} characters "
                f"(received {len(value)})"
            )
        return stripped

    @model_validator(mode="after")
    def _validate_context(self) -> "PredictRequest":
        settings = get_settings()
        turns = [turn for turn in self.context if turn and turn.strip()]
        if len(turns) > settings.max_context_turns:
            raise ValueError(
                f"context may contain at most {settings.max_context_turns} turns "
                f"(received {len(turns)}); send the most recent turns, oldest first"
            )
        for turn in turns:
            if len(turn) > settings.max_context_turn_length:
                raise ValueError(
                    f"each context turn must be at most "
                    f"{settings.max_context_turn_length} characters"
                )
        object.__setattr__(self, "context", [turn.strip() for turn in turns])
        return self


class RetrieveRequest(BaseModel):
    """Body of ``POST /retrieve``."""

    text: str = Field(..., description="Query utterance.")
    context: list[str] = Field(default_factory=list, description="Optional preceding turns.")
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        if len(value) > get_settings().max_text_length:
            raise ValueError("text is too long")
        return value.strip()


class EvidenceOut(BaseModel):
    """One retrieved commonsense fact."""

    text: str
    source: str
    relation: str
    score: float
    fact_id: str
    embedding_id: int
    head: str = ""
    tail: str = ""


class TokenImportanceOut(BaseModel):
    """One token attribution."""

    token: str
    weight: float
    position: int


class ContextInfluenceOut(BaseModel):
    """Attention paid to one conversational turn."""

    turn_index: int
    text: str
    attention: float


class IncongruityOut(BaseModel):
    """The literal-vs-situational polarity check."""

    detected: bool
    utterance_polarity: float
    situation_polarity: float
    positive_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    situation_source: str = "none"
    description: str = ""


class ProbabilitiesOut(BaseModel):
    """Class probabilities."""

    sarcastic: float
    not_sarcastic: float


class PredictResponse(BaseModel):
    """Body of a successful ``POST /predict`` / ``POST /explain``."""

    prediction: str = Field(..., description="'Sarcastic' or 'Not Sarcastic'.")
    label: int = Field(..., description="1 = sarcastic, 0 = not sarcastic.")
    analysed_text: str = Field(
        ...,
        description=(
            "The cleaned utterance the model actually saw. Token attributions "
            "index into this string, not into the raw request text."
        ),
    )
    confidence: float
    probabilities: ProbabilitiesOut
    important_tokens: list[str]
    token_importance: list[TokenImportanceOut]
    context_used: list[str]
    context_influence: list[ContextInfluenceOut]
    retrieved_evidence: list[EvidenceOut]
    retrieval_status: str
    evidence_attention: list[float]
    explanation: str
    explanation_method: str
    incongruity: IncongruityOut
    warnings: list[str]
    timings_ms: dict[str, float]
    request_id: str


class RetrieveResponse(BaseModel):
    """Body of ``POST /retrieve``."""

    query: str
    status: str
    evidence: list[EvidenceOut]
    warning: str | None = None
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok", "degraded", "loading", "error"]
    model_loaded: bool
    retrieval_available: bool
    device: str | None = None
    version: str
    uptime_seconds: float
    detail: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    """Body of ``GET /model/info``."""

    model: dict[str, Any]
    retrieval: dict[str, Any]
    explanation: dict[str, Any]
    device: str
    checkpoint: str | None
    experiment: str
    fusion: str
    task: str
    input_modality: str
    max_context_turns: int
    is_development_asset: bool
    startup_warnings: list[str] = Field(default_factory=list)


class DemoExample(BaseModel):
    """One demonstration input shown in the UI."""

    id: str
    title: str
    text: str
    context: list[str]
    note: str


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: str
    detail: str
    request_id: str
