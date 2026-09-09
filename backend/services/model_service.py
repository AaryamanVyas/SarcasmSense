"""Singleton holder for the analysis service.

The model, tokenizer, FAISS index and sentence encoder are loaded **once**, at
application startup, and shared by every request. Loading them per request would
add seconds of latency and gigabytes of churn.

Load failure is not fatal to the process: the service records the error and
``GET /health`` reports ``status="error"`` with the reason, so an operator sees a
useful message instead of a container that crash-loops.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from backend.settings import Settings
from src.inference.service import SarcasmAnalysisService

logger = logging.getLogger(__name__)


class ModelService:
    """Owns the loaded :class:`SarcasmAnalysisService` for the process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service: SarcasmAnalysisService | None = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._loading = False

    # ---------------------------------------------------------------- state
    @property
    def loaded(self) -> bool:
        """Whether a model is ready to serve."""
        return self._service is not None

    @property
    def loading(self) -> bool:
        """Whether a load is currently in progress."""
        return self._loading

    @property
    def error(self) -> str | None:
        """The load error, if any."""
        return self._error

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the service object was created."""
        return time.time() - self._started_at

    @property
    def service(self) -> SarcasmAnalysisService:
        """The loaded service.

        Raises:
            RuntimeError: when the model is not loaded, carrying the underlying
                reason so the API can return an actionable 503.
        """
        if self._service is None:
            raise RuntimeError(
                self._error
                or "Model is not loaded yet. Check MODEL_PATH and GET /health."
            )
        return self._service

    # ----------------------------------------------------------------- load
    def load(self) -> bool:
        """Load the checkpoint. Returns ``True`` on success.

        Safe to call more than once; subsequent calls are no-ops while a model is
        already loaded.
        """
        with self._lock:
            if self._service is not None:
                return True
            self._loading = True
            self._error = None
            started = time.perf_counter()
            try:
                logger.info("Loading model from %s ...", self.settings.model_path)
                config = None
                if self.settings.config_path:
                    from src.utils.config import Config

                    config = Config.from_yaml(self.settings.config_path)
                self._service = SarcasmAnalysisService.from_checkpoint(
                    self.settings.model_path,
                    device=self.settings.device,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - reported through /health
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Model load failed for %s -- the API will start but /predict "
                    "will return 503 until this is fixed. Reason: %s",
                    self.settings.model_path,
                    self._error,
                )
                return False
            finally:
                self._loading = False
            logger.info(
                "Model loaded in %.2fs | %s",
                time.perf_counter() - started,
                self._service.predictor.describe()["model"]["encoder_name"],
            )
            return True

    def unload(self) -> None:
        """Release the model (used by tests and graceful shutdown)."""
        with self._lock:
            self._service = None
            self._error = None

    # ---------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        """Health payload for ``GET /health``."""
        if self._loading:
            return {
                "status": "loading",
                "model_loaded": False,
                "retrieval_available": False,
                "device": None,
                "uptime_seconds": round(self.uptime_seconds, 2),
                "detail": "Model is loading.",
                "warnings": [],
            }
        if self._service is None:
            return {
                "status": "error",
                "model_loaded": False,
                "retrieval_available": False,
                "device": None,
                "uptime_seconds": round(self.uptime_seconds, 2),
                "detail": self._error or "Model has not been loaded.",
                "warnings": [],
            }
        retriever = self._service.predictor.retriever
        retrieval_ok = bool(retriever is not None and retriever.available)
        warnings = list(self._service.predictor.startup_warnings)
        return {
            # "degraded" is honest: the classifier works, but the system is not
            # running in its full proposed configuration.
            "status": "ok" if retrieval_ok or not self._service.config.model.use_knowledge
            else "degraded",
            "model_loaded": True,
            "retrieval_available": retrieval_ok,
            "device": str(self._service.predictor.device),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "detail": None
            if retrieval_ok
            else "Commonsense retrieval is unavailable; predictions run without evidence.",
            "warnings": warnings,
        }
