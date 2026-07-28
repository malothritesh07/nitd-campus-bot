# Single-stage: the image is dominated by torch + the embedding model, so a
# builder stage saves little and complicates the model bake-in below.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only torch: the GPU build is ~2 GB and this only ever encodes short queries
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the embedding model into the image (~90 MB) so a container start needs no
# network and no first-query stall. Without this the app is unusable offline.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')" \
 && chmod -R a+rX /models

COPY . .

# The model is present, so never reach out to HuggingFace at runtime.
ENV EMBED_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://localhost:8000/api/config >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
