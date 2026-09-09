"""FastAPI application factory.

Run locally::

    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Production (inside the Docker image)::

    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1

One worker is deliberate: the model is several hundred megabytes and each worker
would hold its own copy. Scale with replicas (and a shared model volume) rather
than with workers.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routes import router
from backend.services.model_service import ModelService
from backend.settings import Settings, get_settings
from src.utils.logging_utils import REQUEST_ID, configure_logging, log_event, new_request_id

logger = logging.getLogger("backend")

DESCRIPTION = """
Context-aware, explainable sarcasm detection.

**Pipeline:** RoBERTa encoder (shared between the target utterance and up to three
preceding conversational turns) → offline commonsense retrieval over a FAISS index
of ConceptNet / COMET / WordNet facts → trainable cross-attention fusion →
classification head → LIME / SHAP token attribution → evidence-grounded natural
language explanation.

**Text only.** No audio, video or image input is accepted or used.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup and release it at shutdown."""
    settings: Settings = app.state.settings
    service = ModelService(settings)
    app.state.model_service = service
    if settings.load_model_on_startup:
        service.load()
    else:
        logger.warning("LOAD_MODEL_ON_STARTUP=false; /predict will return 503")
    yield
    service.unload()
    logger.info("Model released; shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    if settings.debug:
        logger.warning(
            "DEBUG=true: internal error details will be returned to clients. "
            "Do not run this way in production."
        )
    if "*" in settings.cors_origin_list:
        logger.warning(
            "CORS_ORIGINS contains '*': any website may call this API. Set an "
            "explicit origin list before deploying publicly."
        )

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=DESCRIPTION,
        root_path=settings.root_path,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # no cookies are used; keeps '*' origins legal
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id, time the request and log the outcome."""
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = REQUEST_ID.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - converted to a 500 below
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception(
                "Unhandled error", extra={"path": request.url.path, "latency_ms": elapsed}
            )
            REQUEST_ID.reset(token)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "detail": "An unexpected error occurred.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        if request.url.path not in ("/health", "/openapi.json"):
            log_event(
                logger,
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(elapsed, 2),
            )
        REQUEST_ID.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(  # type: ignore[no-untyped-def]
        request: Request, exc: RequestValidationError
    ):
        """Return a readable message for schema violations."""
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
            messages.append(f"{location or 'body'}: {error.get('msg')}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": "; ".join(messages) or "Invalid request.",
                "request_id": REQUEST_ID.get(),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):  # type: ignore[no-untyped-def]
        """Uniform error envelope for HTTP errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"http_{exc.status_code}",
                "detail": str(exc.detail),
                "request_id": REQUEST_ID.get(),
            },
        )

    # The API is mounted twice on purpose. At the root it is the plain API; under
    # ``/api`` it matches the path the frontend uses by default, so a single
    # container can serve both without rebuilding the bundle for a new base URL.
    app.include_router(router)
    app.include_router(router, prefix="/api")

    @app.get("/api", tags=["system"], include_in_schema=False)
    def api_index() -> dict[str, str]:
        """Service descriptor under the /api prefix."""
        return _service_descriptor(settings)

    _mount_frontend(app, settings)

    if not getattr(app.state, "frontend_mounted", False):

        @app.get("/", tags=["system"])
        def index() -> dict[str, str]:
            """Point clients at the docs and the health endpoint."""
            return _service_descriptor(settings)

    return app


def _service_descriptor(settings: Settings) -> dict[str, str]:
    """Small JSON pointer to the docs and health endpoints."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Optionally serve the built React bundle from this process."""
    app.state.frontend_mounted = False
    if not settings.serve_frontend:
        return
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as _HTTPException
    from starlette.responses import FileResponse
    from starlette.types import Scope

    from src.utils.paths import resolve

    directory = resolve(settings.frontend_dist)
    index = directory / "index.html"
    if not index.exists():
        logger.warning(
            "SERVE_FRONTEND=true but no build found at %s. Build it with "
            "`npm run build` in frontend/, or leave SERVE_FRONTEND unset.",
            directory,
        )
        return

    class SinglePageApp(StaticFiles):
        """Static files with an SPA fallback, mirroring nginx's ``try_files``.

        A client-side route such as ``/dashboard`` has no file behind it, so
        plain StaticFiles would 404. Serving ``index.html`` instead lets the
        React app resolve the route itself -- the same behaviour the nginx
        frontend container provides in the docker-compose deployment.
        """

        async def get_response(self, path: str, scope: Scope):  # noqa: D102
            try:
                return await super().get_response(path, scope)
            except _HTTPException as exc:
                if exc.status_code == 404 and not path.startswith("assets/"):
                    return FileResponse(index)
                raise

    # Mounted last so that every API route registered above still wins the match.
    app.mount("/", SinglePageApp(directory=str(directory), html=True), name="frontend")
    app.state.frontend_mounted = True
    logger.info("Serving frontend from %s (SPA fallback enabled)", directory)


app = create_app()
