"""Typed experiment configuration loaded from YAML.

Every experiment (baselines, full model, ablations) is fully described by one
YAML file under ``configs/``. Using dataclasses rather than raw dicts gives us
IDE completion, ``mypy``-checkable access and a single place to document what
each hyper-parameter means -- which matters a lot when the project has to be
explained in a viva.

Dotted-key overrides (``model.fusion=concat``) are supported so that
``scripts/run_ablation.py`` can sweep variants without writing 8 extra YAML
files.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

FUSION_TYPES = ("cross_attention", "concat", "none")
CLASS_WEIGHT_MODES = ("none", "balanced")
MIXED_PRECISION_MODES = ("auto", "fp16", "bf16", "off")
RETRIEVAL_QUERY_MODES = ("utterance", "utterance_plus_context")


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically wrong."""


@dataclass
class ExperimentConfig:
    """Bookkeeping for one training run."""

    name: str = "experiment"
    seed: int = 42
    output_dir: str = "checkpoints/experiment"
    results_dir: str = "results"
    notes: str = ""
    deterministic: bool = True


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Attributes:
        encoder_name: HuggingFace id (or local path) of the shared text encoder.
            The research configuration uses ``roberta-base`` (hidden size 768).
        max_utterance_length: Token budget for the target utterance (spec: 128).
        max_context_length: Token budget for the flattened conversational
            context. Context is usually longer than the target, so it gets its
            own budget.
        use_context: Toggle for the conversational-context branch (baseline 1
            and 3 set this to ``false``).
        use_knowledge: Toggle for the retrieved-commonsense branch (baseline 1
            and 2 set this to ``false``).
        fusion: ``cross_attention`` (proposed model), ``concat`` (ablation) or
            ``none`` (utterance-only baseline).
        shared_encoder: When ``true`` the *same* RoBERTa weights encode both the
            utterance and the context, exactly as drawn in the architecture
            diagram. Setting it to ``false`` instantiates a second encoder and
            roughly doubles memory; kept only as an ablation knob.
        num_attention_heads: Heads in the fusion cross-attention block.
        fusion_layers: Number of stacked cross-attention blocks.
        knowledge_projection: When ``true`` the sentence-transformer evidence
            vectors are projected into the encoder hidden size before fusion.
            Required whenever the retrieval encoder dimension differs from the
            text encoder dimension (384 for MiniLM vs 768 for RoBERTa).
    """

    encoder_name: str = "roberta-base"
    max_utterance_length: int = 128
    max_context_length: int = 128
    use_context: bool = True
    use_knowledge: bool = True
    fusion: str = "cross_attention"
    shared_encoder: bool = True
    num_attention_heads: int = 8
    fusion_layers: int = 1
    fusion_dropout: float = 0.1
    knowledge_projection: bool = True
    classifier_hidden: int = 256
    classifier_dropout: float = 0.2
    freeze_encoder_layers: int = 0
    pooling: str = "cls"  # "cls" | "mean"

    def __post_init__(self) -> None:
        if self.fusion not in FUSION_TYPES:
            raise ConfigError(f"model.fusion must be one of {FUSION_TYPES}, got {self.fusion!r}")
        if self.pooling not in ("cls", "mean"):
            raise ConfigError("model.pooling must be 'cls' or 'mean'")
        if self.max_utterance_length < 8:
            raise ConfigError("model.max_utterance_length must be >= 8")
        if self.fusion == "cross_attention" and not (self.use_context or self.use_knowledge):
            raise ConfigError(
                "model.fusion='cross_attention' requires at least one of "
                "model.use_context / model.use_knowledge to be true; otherwise "
                "there is nothing for the utterance to attend to."
            )


@dataclass
class DataConfig:
    """Where the normalised corpus lives and which slices to use."""

    processed_path: str = "data/processed/dataset.jsonl"
    splits_dir: str = "data/splits"
    datasets: list[str] = field(default_factory=list)  # empty = use everything
    max_context_turns: int = 3
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_workers: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.max_context_turns <= 3:
            raise ConfigError("data.max_context_turns must be between 0 and 3 (spec limit)")


@dataclass
class RetrievalConfig:
    """Commonsense retrieval settings.

    ``enabled`` is deliberately separate from :attr:`ModelConfig.use_knowledge`:
    an experiment may retrieve evidence for *explanation* purposes while the
    classifier ignores it (that is exactly what baselines 1 and 2 do in the web
    app, so users still see evidence with a clear "not used by classifier" note).
    """

    enabled: bool = True
    index_dir: str = "knowledge/index"
    top_k: int = 5
    encoder_backend: str = "auto"  # auto | sentence-transformers | spacy | hashing
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_mode: str = "utterance_plus_context"
    min_score: float = 0.0
    cache_size: int = 2048
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ConfigError("retrieval.top_k must be >= 0")
        if self.query_mode not in RETRIEVAL_QUERY_MODES:
            raise ConfigError(f"retrieval.query_mode must be one of {RETRIEVAL_QUERY_MODES}")


@dataclass
class TrainingConfig:
    """Optimisation schedule and hardware-friendliness knobs."""

    epochs: int = 4
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    mixed_precision: str = "auto"
    early_stopping_patience: int = 2
    class_weights: str = "balanced"
    metric_for_best: str = "macro_f1"
    log_every: int = 50
    save_every_epoch: bool = False
    resume_from: str | None = None

    def __post_init__(self) -> None:
        # YAML 1.1 parses a bare ``off``/``no`` as boolean False; accept that
        # spelling rather than failing on a config a user would reasonably write.
        if isinstance(self.mixed_precision, bool):
            self.mixed_precision = "off" if not self.mixed_precision else "auto"
        if isinstance(self.class_weights, bool):
            self.class_weights = "balanced" if self.class_weights else "none"
        if self.class_weights not in CLASS_WEIGHT_MODES:
            raise ConfigError(f"training.class_weights must be one of {CLASS_WEIGHT_MODES}")
        if self.mixed_precision not in MIXED_PRECISION_MODES:
            raise ConfigError(f"training.mixed_precision must be one of {MIXED_PRECISION_MODES}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("training.gradient_accumulation_steps must be >= 1")


@dataclass
class ExplainConfig:
    """Explainability defaults (the API can override per request)."""

    default_method: str = "lime"
    lime_num_samples: int = 200
    lime_num_features: int = 10
    shap_num_samples: int = 64
    max_tokens_reported: int = 12


@dataclass
class Config:
    """Root configuration object."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Sequence[str] | None = None) -> "Config":
        """Load a config file and apply ``key.subkey=value`` overrides."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
        cfg = cls.from_dict(raw)
        if overrides:
            cfg = cfg.with_overrides(overrides)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        """Build a :class:`Config` from a nested mapping, rejecting unknown keys."""
        return _build_dataclass(cls, raw, path="")

    # -------------------------------------------------------------- overrides
    def with_overrides(self, overrides: Iterable[str]) -> "Config":
        """Return a copy with dotted ``key=value`` overrides applied."""
        raw = self.to_dict()
        for override in overrides:
            if "=" not in override:
                raise ConfigError(f"Override must look like key.sub=value, got {override!r}")
            key, _, value = override.partition("=")
            _set_dotted(raw, key.strip(), yaml.safe_load(value))
        new = Config.from_dict(raw)
        new.validate()
        return new

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to plain Python types (YAML/JSON friendly)."""
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the resolved config next to the checkpoint for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path

    # -------------------------------------------------------------- integrity
    def validate(self) -> None:
        """Cross-field validation that individual dataclasses cannot do alone."""
        if self.model.use_knowledge and not self.retrieval.enabled:
            raise ConfigError(
                "model.use_knowledge=true requires retrieval.enabled=true "
                "(the classifier cannot consume evidence that is never retrieved)."
            )
        if self.model.use_knowledge and self.retrieval.top_k == 0:
            raise ConfigError("model.use_knowledge=true requires retrieval.top_k >= 1")
        if self.data.max_context_turns == 0 and self.model.use_context:
            logger.warning(
                "model.use_context=true but data.max_context_turns=0; the context "
                "branch will always receive the empty-context placeholder."
            )
        if self.model.fusion == "none" and (self.model.use_context or self.model.use_knowledge):
            raise ConfigError(
                "model.fusion='none' is only valid for the utterance-only baseline; "
                "disable use_context and use_knowledge or pick another fusion."
            )


# --------------------------------------------------------------------- helpers
def _build_dataclass(cls: type[T], raw: Mapping[str, Any], path: str) -> T:
    """Instantiate ``cls`` from ``raw``, recursing into nested dataclasses."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Expected a mapping at {path or '<root>'}, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown configuration key(s) at {path or '<root>'}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them once against this module's namespace.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        annotation = hints[name]
        child_path = f"{path}.{name}".lstrip(".")
        if is_dataclass(annotation) and isinstance(annotation, type):
            kwargs[name] = _build_dataclass(annotation, value, child_path)
        else:
            kwargs[name] = _coerce(value, annotation, child_path)
    return cls(**kwargs)  # type: ignore[return-value]


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Best-effort scalar coercion (YAML gives us ``"2e-5"`` as a string)."""
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is not None:  # Optional[...] / list[...] etc.
        args = [a for a in get_args(annotation) if a is not type(None)]
        if origin in (list, Sequence) and isinstance(value, list):
            return value
        if len(args) == 1:
            return _coerce(value, args[0], path)
        return value
    if annotation is float and isinstance(value, (int, str)):
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: cannot parse {value!r} as float") from exc
    if annotation is int and isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: cannot parse {value!r} as int") from exc
    if annotation is bool and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return value


def _set_dotted(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``raw['a']['b'] = value`` for ``dotted == 'a.b'``."""
    parts = dotted.split(".")
    cursor: dict[str, Any] = raw
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            raise ConfigError(f"Cannot override {dotted!r}: {part!r} is not a section")
        cursor = nxt
    if parts[-1] not in cursor:
        raise ConfigError(f"Cannot override unknown key {dotted!r}")
    cursor[parts[-1]] = value
