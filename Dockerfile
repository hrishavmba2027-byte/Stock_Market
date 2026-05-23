FROM python:3.11-slim

# TARGETARCH is populated automatically by Docker BuildKit (arm64 on Apple
# Silicon, amd64 on x86_64 hosts / CI).  We use it to pick the right torch
# wheel: pytorch.org's CPU-only +cpu builds for amd64, and the standard PyPI
# wheel (which is already CPU-only) for arm64.
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps:
#   - build-essential / gcc: native C wheels (pyarrow, sentencepiece, scipy)
#   - libgomp1: PyTorch OpenMP runtime
#   - libglib2.0-0, libgl1: pyarrow + some transformers backends
#   - curl, ca-certificates: healthcheck + HTTPS downloads
#   - git: needed by pip when any dependency is resolved from a VCS source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache pip install: copy only requirements.txt first so rebuilds
# triggered by code changes don't reinstall the world.
COPY requirements.txt .

# Install PyTorch CPU wheel.
#
# amd64 (x86_64): use pytorch.org's /whl/cpu index — the +cpu suffix wheel
#   avoids pulling down the multi-gigabyte CUDA build from PyPI.
#
# arm64 (aarch64 / Apple Silicon): the standard PyPI wheel IS already CPU-only
#   for linux/aarch64; the +cpu suffix wheels on pytorch.org do not have
#   reliable aarch64 builds for all minor releases. Using plain PyPI keeps the
#   install reproducible across arm64 hosts.
#
# Neither path installs torchvision or torchaudio — nothing in this project
# imports them, and their wheels carry a strict torch== requirement that
# conflicts with the +cpu local-version identifier on amd64.
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ "${TARGETARCH}" = "arm64" ] || [ "${TARGETARCH}" = "aarch64" ]; then \
        echo "ARM64 host detected — installing torch from standard PyPI (CPU-only wheel)" && \
        python -m pip install "torch==2.6.0"; \
    else \
        echo "AMD64 host detected — installing torch CPU wheel from pytorch.org" && \
        python -m pip install \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            "torch==2.6.0+cpu"; \
    fi && \
    python -m pip install -r requirements.txt

# Application code — everything we need at runtime.
# Top-level scripts:
COPY app.py app_data.py main.py Data_update.py Feature_Engineering.py ./
# Top-level Python packages added in P1:
COPY app ./app
COPY ingestion ./ingestion
COPY features ./features
COPY mlops ./mlops
# Tests are bundled so you can `docker compose run --rm api pytest` for parity
COPY tests ./tests
# Committed model artifacts (overridden at runtime when the GitHub Release
# bootstrap puts fresher weights here via the volume mount):
COPY outputs/Saved_Models ./outputs/Saved_Models

# Legacy back-compat symlink (some scripts referenced Data_Update.py with
# a capital U) + runtime dirs used by the FastAPI app & ingest modules.
RUN ln -s Data_update.py Data_Update.py \
    && mkdir -p /app/logs \
                /app/outputs/main_inference \
                /app/credentials \
                /app/state \
                /app/Data \
                /app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app_data:app", "--host", "0.0.0.0", "--port", "8000"]
