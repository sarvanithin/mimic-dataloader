FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY mimic_dataloader/ ./mimic_dataloader/
COPY api/ ./api/

RUN pip install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
