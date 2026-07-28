"""Rate limiting for the public endpoints.

Two separate budgets, because the two costs are different:

  requests  — protects the server. Generous, since 94% of answers are a single
              indexed Mongo read.
  llm_calls — protects the Groq quota. Much tighter, because those are the only
              requests that cost money and take seconds.

Counters live in MongoDB with a TTL index, so they expire themselves and survive
a restart. IPs are hashed with SERVER_PEPPER before storage — a rate-limit table
should not double as a log of who asked what.
"""
import hashlib
from datetime import datetime, timedelta, timezone

import db as D
import store
from config import CFG


def _now():
    return datetime.now(timezone.utc)


def client_id(request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown")
    return hashlib.sha256((ip + "|" + D.PEPPER.decode()).encode()).hexdigest()[:32]


def ensure_indexes():
    store.ensure_indexes(int(CFG.get("cache.ttl_hours", 168)))


def check(cid: str, kind: str = "request") -> tuple[bool, str, int]:
    """Returns (allowed, reason, retry_after_seconds). Never raises — if the
    counter store is unreachable the request is allowed through rather than
    taking the whole bot down over a rate limiter."""
    try:
        per_min  = int(CFG.get("ratelimit.requests_per_minute", 90))
        per_hour = int(CFG.get("ratelimit.requests_per_hour", 600))
        llm_hour = int(CFG.get("ratelimit.llm_calls_per_hour", 40))

        if kind == "llm":
            if store.rate_count(cid, "llm", 3600) >= llm_hour:
                return False, "llm_hourly", 3600
            return True, "", 0

        if store.rate_count(cid, "request", 60) >= per_min:
            return False, "per_minute", 60
        if store.rate_count(cid, "request", 3600) >= per_hour:
            return False, "per_hour", 3600
        return True, "", 0
    except Exception:
        return True, "", 0


def record(cid: str, kind: str = "request", path: str = ""):
    store.rate_record(cid, kind, (60, 3600), path)


def stats(cid: str | None = None) -> dict:
    now = _now()
    q = {"cid": cid} if cid else {}
    return {
        "backend": store.backend(),
        "limits": {
            "requests_per_minute": CFG.get("ratelimit.requests_per_minute", 90),
            "requests_per_hour":   CFG.get("ratelimit.requests_per_hour", 600),
            "llm_calls_per_hour":  CFG.get("ratelimit.llm_calls_per_hour", 40)},
        "last_minute": D.db.rate_events.count_documents(
            {**q, "ts": {"$gte": now - timedelta(minutes=1)}}),
        "last_hour":   D.db.rate_events.count_documents(
            {**q, "ts": {"$gte": now - timedelta(hours=1)}}),
        "llm_last_hour": D.db.rate_events.count_documents(
            {**q, "kind": "llm", "ts": {"$gte": now - timedelta(hours=1)}}),
        "distinct_clients_last_hour": len(D.db.rate_events.distinct(
            "cid", {"ts": {"$gte": now - timedelta(hours=1)}})),
    }
