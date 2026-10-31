# Python minor version must match the one used to pickle the model (3.13).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/baseline_model.pkl

# LightGBM needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY models/baseline_model.pkl models/baseline_model.pkl
COPY src/triage src/triage
COPY src/app src/app

RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run injects PORT (default 8080).
EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run src/app/main.py \
  --server.port=${PORT:-8080} --server.address=0.0.0.0 \
  --server.headless=true --browser.gatherUsageStats=false"]
