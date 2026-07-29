"""Academic calendar retrieval.

Pipeline: dense retrieval over the calendar chunks, slot extraction from the
question, then FlashRank cross-encoder reranking with a metadata boost.

Two-stage ranking matters here because calendar entries are short and lexically
similar to one another; a bi-encoder alone ranks "Mid Semester Examination"
and "End Semester Examination" almost identically. The cross-encoder sees the
query and passage together and separates them.

Slot extraction uses Groq when available and falls back to keyword rules, so the
category keeps working without a key.
"""
import json
import logging
import os
import re

import numpy as np

import db as D
from config import CFG
from embeddings import encode
from tracing import trace

log = logging.getLogger(__name__)

_cache = {"chunks": None, "vecs": None}
_ranker = None

CACHE_DIR = os.getenv("FLASHRANK_CACHE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flashrank_cache"))


def load_cache(force=False):
    if _cache["chunks"] is not None and not force:
        return
    chunks = list(D.db.chunks.find(
        {"domain": "calendar", "status": {"$ne": "archived"}}))
    for c in chunks:
        c.update({k: v for k, v in (c.get("meta") or {}).items() if k not in c})
    _cache["chunks"] = chunks
    _cache["vecs"] = (np.array([c["embedding"] for c in chunks], dtype="float32")
                      if chunks and all(c.get("embedding") for c in chunks) else None)


def get_ranker():
    """Returns None if the cross-encoder cannot be loaded; callers then use the
    dense ordering alone."""
    global _ranker
    if _ranker is not None:
        return _ranker if _ranker is not False else None
    try:
        from flashrank import Ranker
        _ranker = Ranker(model_name=CFG.get("retrieval.rerank_model"),
                         cache_dir=CACHE_DIR)
    except Exception as exc:
        log.warning("Reranker unavailable, using dense order: %s", exc)
        _ranker = False
        return None
    return _ranker


def allowed_values() -> dict:
    """Enum of filter values, derived from the data so it cannot drift."""
    load_cache()
    fields = ("event_type", "month", "category", "semester")
    out = {}
    for field in fields:
        out[field] = sorted({str(c.get(field)) for c in _cache["chunks"]
                             if c.get(field)}) + ["none"]
    return out


NONE = "none"


def _rule_filters(query: str, allowed: dict) -> dict:
    """Keyword slot extraction. Used when no model is available, and to check
    the model's answer."""
    ql = query.lower()
    slots = {k: NONE for k in allowed}

    for pattern, value in CFG.get("lexicon.calendar_event_types"):
        if re.search(pattern, ql):
            slots["event_type"] = value
            break

    for month in allowed.get("month", []):
        if month != NONE and month.lower()[:3] in ql:
            slots["month"] = month
            break
    return slots


def _model_filters(query: str, allowed: dict) -> dict | None:
    import rag_handlers as H
    if not H.llm_available():
        return None
    prompt = CFG.get("prompts.calendar_filter").format(
        event_type=allowed.get("event_type"), month=allowed.get("month"),
        category=allowed.get("category"), semester=allowed.get("semester"),
        query=query)
    try:
        # Temperature 0: this is extraction, not writing. At the default the
        # model intermittently filled in a month the question never stated,
        # inferring it from world knowledge rather than the text.
        raw = H.llm([{"role": "user", "content": prompt}], max_tokens=120,
                    temperature=0)
    except Exception as exc:
        log.warning("Filter extraction failed, using keyword rules: %s", exc)
        return None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    # Validate against the enum: anything hallucinated becomes "none" rather
    # than silently filtering on a value that exists nowhere in the data.
    clean = {}
    for field, values in allowed.items():
        candidate = str(parsed.get(field, NONE)).strip().lower().rstrip("s")
        clean[field] = next(
            (v for v in values if v.lower().rstrip("s") == candidate), NONE)
    return clean


@trace(name="calendar_filters", run_type="chain")
def extract_filters(query: str) -> tuple[dict, str]:
    """Returns (slots, method) where method records which path produced them."""
    allowed = allowed_values()
    slots = _model_filters(query, allowed)
    if slots is not None and any(v != NONE for v in slots.values()):
        return slots, "llm"
    rules = _rule_filters(query, allowed)
    return rules, "rules"


def _boost(meta: dict, slots: dict) -> float:
    """Metadata agreement nudges a passage up without overriding relevance."""
    weights = CFG.get("retrieval.calendar_boosts")
    total = 0.0
    for field, weight in weights.items():
        want = slots.get(field, NONE)
        have = meta.get(field)
        if want != NONE and have and str(have).strip().lower() == str(want).strip().lower():
            total += weight
    return total


@trace(name="calendar_search", run_type="retriever")
def search(query: str, top_k=None, final_k=None):
    """Returns [(chunk, score, boost)] best first."""
    top_k = top_k or CFG.get("retrieval.calendar_pool")
    final_k = final_k or CFG.get("retrieval.top_k_calendar")
    load_cache()
    chunks = _cache["chunks"]
    if not chunks:
        return [], {}, "no-data"

    slots, how = extract_filters(query)

    qv = encode(query)
    if qv is not None and _cache["vecs"] is not None:
        sims = _cache["vecs"] @ qv
        order = list(np.argsort(sims)[::-1][:top_k])
    else:
        # No embeddings: hand the whole set to the reranker, which is small
        # enough here that this stays fast.
        order = list(range(len(chunks)))[:top_k]

    candidates = [chunks[i] for i in order]
    ranker = get_ranker()
    if ranker is None:
        scored = [(c, 1.0 - n / max(1, len(candidates)), _boost(c, slots))
                  for n, c in enumerate(candidates)]
    else:
        from flashrank import RerankRequest
        passages = [{"id": n, "text": c["text"], "meta": c}
                    for n, c in enumerate(candidates)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        scored = [(r["meta"], float(r["score"]), _boost(r["meta"], slots))
                  for r in ranked]

    scored.sort(key=lambda x: x[1] + x[2], reverse=True)
    return scored[:final_k], slots, how


def render(hits) -> str:
    """One line per event, deduplicated, dates verbatim.

    Ranking decides which events are relevant; presentation is chronological.
    Listing "Re-Mid Examination" above the mid-semester exam it follows is
    confusing even when the reranker scores it marginally higher.
    """
    seen, rows = set(), []
    for chunk, _score, _boost in hits:
        text = chunk["text"].strip()
        if text in seen:
            continue
        seen.add(text)
        rows.append((chunk.get("date_iso") or "", text))
    # Undated entries (summary, totals) sort last rather than first.
    rows.sort(key=lambda r: (r[0] == "", r[0]))
    return "\n".join(text for _date, text in rows)


def coverage() -> str:
    load_cache()
    semesters = sorted({c.get("semester") for c in _cache["chunks"] if c.get("semester")})
    if not semesters:
        return "No academic calendar is loaded yet."
    return f"I hold the academic calendar for {', '.join(semesters)}."
