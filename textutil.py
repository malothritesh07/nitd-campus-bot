"""Text helpers shared by the retrieval modules."""
import re

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, as used by every BM25 index here."""
    return _WORD.findall((text or "").lower())


def format_inr(amount) -> str:
    return f"Rs {amount:,}"
