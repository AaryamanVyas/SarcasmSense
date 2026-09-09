"""Backend configuration, entirely environment-driven.

No hostname, port, path or origin is hard-coded anywhere in the serving code:
everything comes from environment variables (see ``.env.example``), which is
what makes the same image runnable locally, in Docker Compose and on a PaaS.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the FastAPI application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Pydantic reserves the ``model_`` prefix; ``MODEL_PATH`` is the natural
        # environment-variable name for this project, so opt out of the guard.
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- model
    model_path: str = Field(
        default="checkpoints/full_model/best_model",
        description="Directory containing the trained checkpoint to serve.",
    )
    device: str | None = Field(
        default=None, description="Torch device override; None = auto-detect CUDA."
    )
    config_path: str | None = Field(
        default=None,
        description="Override the training_config.yaml bundled with the checkpoint.",
    )
    load_model_on_startup: bool = Field(
        default=True,
        description="Load the model during app startup (set false only for tests).",
    )

    # ------------------------------------------------------------------ api
    api_title: str = "Context-Aware Sarcasm Detection API"
    api_version: str = "1.0.0"
    root_path: str = Field(default="", description="Path prefix behind a reverse proxy.")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
        description="Comma-separated list of allowed browser origins.",
    )
    debug: bool = Field(default=False, description="Verbose errors. NEVER true in production.")

    # ------------------------------------------------------------- limits
    max_text_length: int = Field(default=2000, ge=1, le=20000)
    max_context_turns: int = Field(default=3, ge=0, le=3)
    max_context_turn_length: int = Field(default=2000, ge=1, le=20000)
    max_batch_size: int = Field(default=16, ge=1, le=128)
    request_timeout_seconds: int = Field(default=120, ge=1)

    # ------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_format: str = Field(default="console", description="'console' or 'json'.")
    log_request_text: bool = Field(
        default=False,
        description=(
            "Log a truncated preview of user text at DEBUG level. Off by default: "
            "user-submitted content is not written to logs in production."
        ),
    )

    # ------------------------------------------------------------- results
    results_dir: str = Field(
        default="results", description="Where the dashboard reads saved experiment results."
    )

    # ------------------------------------------------------------ frontend
    serve_frontend: bool = Field(
        default=False,
        description=(
            "Serve the pre-built React bundle from this process. Useful on "
            "single-service platforms (Render free tier, HF Spaces) where you "
            "cannot run a separate static host. With docker-compose the frontend "
            "gets its own nginx container instead."
        ),
    )
    frontend_dist: str = Field(
        default="frontend_dist",
        description="Directory holding the built frontend (index.html + assets).",
    )

    @field_validator("log_format")
    @classmethod
    def _check_log_format(cls, value: str) -> str:
        if value not in ("console", "json"):
            raise ValueError("log_format must be 'console' or 'json'")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed CORS origins. ``*`` is honoured but warned about at startup."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
