"""Answer cache for LLM-backed responses.

Only answers that cost a Groq call are cached. Everything else is already a
sub-100ms Mongo lookup, and caching it would add staleness risk for no gain.

Three things keep a cached answer from going stale:

1. **Corpus generation.** The key includes the latest `ingest_log` run id, so any
   `sync.py` run silently invalidates every entry — a fee or module list can
   never be served from before a data change.
2. **TTL.** Mongo expires entries after `cache.ttl_hours`, so an abandoned cache
   drains itself.
3. **Scope.** Volatile categories (shop status) and anything carrying a
   timestamp are never cached at all.
"""
import hashlib
import json
import re
from datetime import datetime, timezone

import db as D
import store
from config import CFG

_generation = {"id": None, "at": None}


NEVER_CACHE = {"shops", "feedback"}


def _now():
    return datetime.now(timezone.utc)


def generation() -> str:
    """Latest corpus sync id, cached briefly so this isn't a query per request."""
    if _generation["at"] and (_now() - _generation["at"]).total_seconds() < 60:
        return _generation["id"] or "none"
    doc = D.db.ingest_log.find_one({"mode": "sync"}, sort=[("ts", -1)])
    _generation["id"] = (doc or {}).get("run_id", "none")
    _generation["at"] = _now()
    return _generation["id"]


def _norm(text: str) -> str:
    """Whitespace, case and trailing punctuation shouldn't split the cache."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip("?!. ")


def key(message: str, category: str | None, state: dict | None) -> str:
    """State matters: 'list them' means different things in different contexts,
    so the resolved course/programme/semester is part of the identity."""
    ctx = {}
    for f in ("syl_course", "syl_program", "syl_semester", "syl_intent"):
        if (state or {}).get(f) is not None:
            ctx[f] = state[f]
    payload = json.dumps({"q": _norm(message), "c": category or "",
                          "ctx": ctx, "gen": generation()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def enabled() -> bool:
    return bool(CFG.get("cache.enabled", True))


def get(message, category, state):
    return get_by_key(key(message, category, state), category)


def get_by_key(k: str, category: str | None = None):
    if not enabled() or category in NEVER_CACHE:
        return None
    hit = store.cache_get(k)
    if not hit:
        return None
    out = dict(hit)
    out["method"] = out.get("method", "") + " · cached"
    out["cached"] = True
    return out


def put(message, category, state, response: dict) -> bool:
    return put_by_key(key(message, category, state), message, category, response)


def put_by_key(k: str, message: str, category, response: dict) -> bool:
    """Cache only if the answer actually cost an LLM call."""
    if not enabled() or category in NEVER_CACHE:
        return False
    if "+ LLM" not in (response.get("method") or ""):
        return False
    store.cache_put(
        k,
        {k: v for k, v in response.items() if k != "state"},
        ttl_seconds=int(CFG.get("cache.ttl_hours", 168)) * 3600,
        meta={"query": _norm(message), "category": category,
              "generation": generation()})
    return True


def ensure_indexes():
    store.ensure_indexes(int(CFG.get("cache.ttl_hours", 168)))


def stats() -> dict:
    s = store.cache_stats(generation())
    return {"enabled": enabled(), "generation": generation(),
            "llm_calls_saved": s["hits_served"], **s}


def clear(all_generations=False) -> int:
    return store.cache_clear(None if all_generations else generation())
