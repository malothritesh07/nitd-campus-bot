# Dockerfile for Hugging Face Spaces.
#
# Differs from the root Dockerfile in three ways that Spaces requires:
#   - listens on 7860 (the port Spaces routes to)
#   - runs as uid 1000, not root — Spaces refuses root containers
#   - HF_HOME points somewhere that user owns, or the model cache fails to write
#
# Everything else is the same image: CPU-only torch, model baked in at build so
# the container needs no network at runtime.

FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    EMBED_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# CPU wheels only — the GPU build is ~2 GB and this only encodes short queries
RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Bake the embedding model into the image so a cold start needs no download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://localhost:7860/api/config >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
