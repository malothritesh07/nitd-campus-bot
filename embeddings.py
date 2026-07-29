"""Sentence-embedding model loader.

Dense retrieval is optional. When the model is unavailable — disabled by
configuration, or a failed download — callers receive None and fall back to
BM25 alone. Exact-lookup categories never use vectors and are unaffected.
"""
import logging
import os

log = logging.getLogger(__name__)

_model = None
_load_failed = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"
_TRUTHY = ("1", "true", "yes")


def _flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in _TRUTHY


def dense_enabled() -> bool:
    """Loading the model costs roughly 460 MB of RSS. Set ENABLE_DENSE=0 on
    memory-constrained hosts to skip it."""
    return os.getenv("ENABLE_DENSE", "1").strip().lower() not in ("0", "false", "no")


def get_model():
    """Return the loaded model, or None if dense retrieval is unavailable.

    A failed load is recorded so later calls skip the attempt rather than
    repeatedly paying the download timeout.
    """
    global _model, _load_failed
    if not dense_enabled() or _load_failed:
        return None
    if _model is not None:
        return _model

    # Opt-in only: forcing offline mode would stop a fresh install from ever
    # downloading the model.
    if _flag("EMBED_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(os.getenv("EMBED_MODEL", DEFAULT_MODEL))
    except Exception as exc:
        _load_failed = True
        log.warning("Embedding model unavailable, using BM25 only: %s", exc)
        return None
    return _model


def encode(text: str):
    """Return a normalised query vector, or None if no model is available."""
    model = get_model()
    if model is None:
        return None
    return model.encode([text], normalize_embeddings=True)[0]
