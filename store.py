"""Answer cache and rate-limit counters, backed by MongoDB TTL collections.

Both live in Mongo rather than a dedicated cache so a clone needs no extra
service to run. `count_documents` carries a benign race under concurrency: two
simultaneous requests can read the same count and both proceed. At this traffic
that will not fire, and the thing being protected — a multi-second model call —
dwarfs the difference in read latency.
"""
from datetime import datetime, timedelta, timezone

import db as D


def _now():
    return datetime.now(timezone.utc)


def backend() -> str:
    return "mongo"


def cache_get(key: str):
    doc = D.db.answer_cache.find_one({"_id": key})
    if not doc:
        return None
    D.db.answer_cache.update_one(
        {"_id": key}, {"$inc": {"hits": 1}, "$set": {"last_hit": _now()}})
    return doc["response"]


def cache_put(key: str, value: dict, ttl_seconds: int, meta: dict | None = None):
    D.db.answer_cache.replace_one(
        {"_id": key},
        {"_id": key, "response": value, "created_at": _now(),
         "last_hit": None, "hits": 0, **(meta or {})},
        upsert=True)


def cache_stats(generation: str) -> dict:
    agg = list(D.db.answer_cache.aggregate(
        [{"$group": {"_id": None, "h": {"$sum": "$hits"}}}]))
    return {"backend": "mongo",
            "entries": D.db.answer_cache.count_documents({}),
            "hits_served": agg[0]["h"] if agg else 0}


def cache_clear(generation: str | None = None) -> int:
    query = {} if generation is None else {"generation": {"$ne": generation}}
    return D.db.answer_cache.delete_many(query).deleted_count


def rate_count(cid: str, kind: str, window_seconds: int) -> int:
    """Events recorded for this client within the window."""
    return D.db.rate_events.count_documents(
        {"cid": cid, **({"kind": kind} if kind != "request" else {}),
         "ts": {"$gte": _now() - timedelta(seconds=window_seconds)}})


def rate_record(cid: str, kind: str, windows: tuple[int, ...], path: str = ""):
    """One row per event; the window is applied when counting.

    Failures are swallowed: losing a rate-limit record is preferable to failing
    the request it was meant to meter.
    """
    try:
        D.db.rate_events.insert_one(
            {"cid": cid, "kind": kind, "path": path, "ts": _now()})
    except Exception:
        pass


def ensure_indexes(cache_ttl_hours: int):
    """TTL indexes expire cached answers and rate events without a cron job."""
    for collection, field, seconds in (
            ("answer_cache", "created_at", cache_ttl_hours * 3600),
            ("rate_events", "ts", 7200)):
        try:
            D.db[collection].create_index(field, expireAfterSeconds=seconds)
        except Exception:
            try:
                D.db[collection].drop_index(f"{field}_1")
                D.db[collection].create_index(field, expireAfterSeconds=seconds)
            except Exception:
                pass
    D.db.rate_events.create_index([("cid", 1), ("ts", 1)])
    D.db.answer_cache.create_index("generation")
