# =============================================================================
# Context-Aware Explainable Sarcasm Detection -- production image
# =============================================================================
# Multi-stage:
#   1. frontend-builder : Node builds the React bundle into /app/dist
#   2. runtime          : Python slim, CPU-only PyTorch, serves the API and
#                         (optionally) the pre-built frontend
#
# Model weights and the FAISS index are NOT baked in. They are large, they
# change independently of the code, and they must not live in git. Mount them:
#
#   docker run -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints:ro \
#     -v $(pwd)/knowledge:/app/knowledge:ro \
#     -e MODEL_PATH=/app/checkpoints/full_model/best_model \
#     sarcasm-api
# =============================================================================

# ----------------------------------------------------------------- stage 1
FROM node:22-alpine AS frontend-builder

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY frontend/ ./
# The bundle talks to the API through a relative /api path by default, so the
# same image works behind any reverse proxy without a rebuild.
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


# ----------------------------------------------------------------- stage 2
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Context-Aware Sarcasm Detection API" \
      org.opencontainers.image.description="RoBERTa + conversational context + retrieval-augmented commonsense reasoning + explainable AI" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SARCASM_PROJECT_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/knowledge/raw/nltk_data \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# curl is used by the container HEALTHCHECK; build tools are needed by a few
# wheels and are removed again in the same layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default PyPI wheel drags in ~2.5 GB of CUDA
# libraries that a CPU serving container will never use.
# The fallback matters: some networks mirror PyPI but not download.pytorch.org.
# The PyPI wheel also works on CPU, it just carries the unused CUDA libraries.
COPY requirements-inference.txt ./
RUN (pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu \
     || pip install --no-cache-dir torch==2.5.1) \
    && pip install --no-cache-dir -r requirements-inference.txt

# Application code (see .dockerignore for what is deliberately excluded).
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Pre-built frontend, served by the API container when SERVE_FRONTEND=true.
COPY --from=frontend-builder /app/dist ./frontend_dist/

# Mount points for artefacts that are too large for the image or for git.
RUN mkdir -p /app/checkpoints /app/knowledge /app/data /app/results /app/.cache

# Run unprivileged.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV MODEL_PATH=/app/checkpoints/full_model/best_model \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    DEBUG=false \
    RESULTS_DIR=/app/results

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker on purpose: each worker would hold its own copy of the model.
# Scale with replicas, not with --workers.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
