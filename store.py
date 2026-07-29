"""Key-value + counter backend, Redis if available, MongoDB otherwise.

Set REDIS_URL and the cache and rate limiter use Redis; leave it unset and they
use MongoDB TTL collections. Nothing else in the codebase changes, and a clone
still runs with `docker compose up` and no extra service.

Why offer both:

  Redis    counters are atomic (INCR), so two concurrent requests cannot both
           read 89 and both proceed. Sub-millisecond reads.
  MongoDB  no extra service to run, deploy or monitor. `count_documents` has a
           benign race under concurrency — at a few requests per second it will
           not fire, and the thing being protected (a 3-4s LLM call) dwarfs the
           difference in read latency.

The Redis path degrades to MongoDB automatically if the connection drops, so a
Redis outage slows things down rather than taking the bot offline.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import db as D

REDIS_URL = (os.getenv("REDIS_URL") or "").strip()
_r = None
_backend = "mongo"
_last_fail = 0.0


def _now():
    return datetime.now(timezone.utc)


def _redis():
    """Connect lazily. After a failure, don't retry for 30s — otherwise every
    request pays the connection timeout."""
    global _r, _backend, _last_fail
    if not REDIS_URL:
        return None
    if _r is not None:
        return _r
    if time.time() - _last_fail < 30:
        return None
    try:
        import redis
        c = redis.Redis.from_url(REDIS_URL, decode_responses=True,
                                 socket_connect_timeout=2, socket_timeout=2)
        c.ping()
        _r, _backend = c, "redis"
        return _r
    except Exception:
        _last_fail = time.time()
        return None


def backend() -> str:
    return "redis" if _redis() is not None else "mongo"


# ------------------------------------------------------------------ cache
def cache_get(key: str):
    r = _redis()
    if r is not None:
        try:
            raw = r.get(f"ans:{key}")
            if raw:
                r.hincrby("ans:hits", key, 1)
                return json.loads(raw)
            return None
        except Exception:
            pass                                   # fall through to Mongo
    doc = D.db.answer_cache.find_one({"_id": key})
    if not doc:
        return None
    D.db.answer_cache.update_one({"_id": key},
                                 {"$inc": {"hits": 1}, "$set": {"last_hit": _now()}})
    return doc["response"]


def cache_put(key: str, value: dict, ttl_seconds: int, meta: dict | None = None):
    r = _redis()
    if r is not None:
        try:
            r.setex(f"ans:{key}", ttl_seconds, json.dumps(value, default=str))
            if meta and meta.get("generation"):
                r.sadd(f"ans:gen:{meta['generation']}", key)
            return
        except Exception:
            pass
    D.db.answer_cache.replace_one(
        {"_id": key},
        {"_id": key, "response": value, "created_at": _now(),
         "last_hit": None, "hits": 0, **(meta or {})},
        upsert=True)


def cache_stats(generation: str) -> dict:
    r = _redis()
    if r is not None:
        try:
            keys = r.keys("ans:*")
            entries = len([k for k in keys if not k.startswith("ans:gen:")
                           and k != "ans:hits"])
            hits = sum(int(v) for v in (r.hgetall("ans:hits") or {}).values())
            return {"backend": "redis", "entries": entries, "hits_served": hits}
        except Exception:
            pass
    agg = list(D.db.answer_cache.aggregate(
        [{"$group": {"_id": None, "h": {"$sum": "$hits"}}}]))
    return {"backend": "mongo",
            "entries": D.db.answer_cache.count_documents({}),
            "hits_served": agg[0]["h"] if agg else 0}


def cache_clear(generation: str | None = None) -> int:
    r = _redis()
    if r is not None:
        try:
            keys = [k for k in r.keys("ans:*") if k != "ans:hits"]
            return r.delete(*keys) if keys else 0
        except Exception:
            pass
    q = {} if generation is None else {"generation": {"$ne": generation}}
    return D.db.answer_cache.delete_many(q).deleted_count


# ------------------------------------------------------------- rate limits
def rate_count(cid: str, kind: str, window_seconds: int) -> int:
    """How many events for this client in the window."""
    r = _redis()
    if r is not None:
        try:
            bucket = int(time.time() // window_seconds)
            return int(r.get(f"rl:{kind}:{cid}:{bucket}") or 0)
        except Exception:
            pass
    return D.db.rate_events.count_documents(
        {"cid": cid, **({"kind": kind} if kind != "request" else {}),
         "ts": {"$gte": _now() - timedelta(seconds=window_seconds)}})


def rate_record(cid: str, kind: str, windows: tuple[int, ...], path: str = ""):
    """Redis uses fixed windows (one counter per bucket, atomic INCR).
    Mongo stores one event row and counts on read."""
    r = _redis()
    if r is not None:
        try:
            pipe = r.pipeline()
            for w in windows:
                bucket = int(time.time() // w)
                k = f"rl:{kind}:{cid}:{bucket}"
                pipe.incr(k)
                pipe.expire(k, w * 2)
            pipe.execute()
            return
        except Exception:
            pass
    try:
        D.db.rate_events.insert_one(
            {"cid": cid, "kind": kind, "path": path, "ts": _now()})
    except Exception:
        pass


def ensure_indexes(cache_ttl_hours: int):
    """Mongo-only: TTL indexes. Redis expires keys itself."""
    for coll, field, secs in (("answer_cache", "created_at", cache_ttl_hours * 3600),
                              ("rate_events", "ts", 7200)):
        try:
            D.db[coll].create_index(field, expireAfterSeconds=secs)
        except Exception:
            try:
                D.db[coll].drop_index(f"{field}_1")
                D.db[coll].create_index(field, expireAfterSeconds=secs)
            except Exception:
                pass
    D.db.rate_events.create_index([("cid", 1), ("ts", 1)])
    D.db.answer_cache.create_index("generation")
