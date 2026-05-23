FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps:
#   - build-essential / gcc: native wheels (pyarrow, sentencepiece)
#   - libgomp1: lightgbm / pytorch runtime
#   - libglib2.0-0, libgl1: pyarrow + some transformers backends
#   - curl, ca-certificates: healthcheck + HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache pip install: copy only requirements.txt first so rebuilds
# triggered by code changes don't reinstall the world.
COPY requirements.txt .
# Install the CPU torch wheel first from pytorch.org's index so we don't drag
# in the multi-gig CUDA build. The subsequent `-r requirements.txt` reuses it
# (the version range in requirements.txt is satisfied).
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" && \
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
COPY outputs/pipeline_metadata.json ./outputs/pipeline_metadata.json

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
