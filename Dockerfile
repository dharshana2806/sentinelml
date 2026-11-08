FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY sentinelml ./sentinelml
RUN pip install --no-cache-dir ".[api,tracking]"
COPY artifacts ./artifacts
RUN useradd -m app && chown -R app /app
USER app
ENV ARTIFACT_DIR=/app/artifacts FAILURE_PREDICTOR_PATH=/app/artifacts/failure_predictor.joblib FAILURE_TYPE_PATH=/app/artifacts/failure_type_clf.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "sentinelml.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
