FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY app.py app_data.py main.py Data_update.py Feature_Engineering.py ./
COPY outputs/Saved_Models ./outputs/Saved_Models
COPY outputs/pipeline_metadata.json ./outputs/pipeline_metadata.json

RUN ln -s Data_update.py Data_Update.py \
    && mkdir -p /app/logs /app/outputs/main_inference /app/credentials /app/state

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app_data:app", "--host", "0.0.0.0", "--port", "8000"]
