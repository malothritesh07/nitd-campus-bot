"""Shared DB access, config and status logic for the campus shop-status feature."""
import os, hmac, hashlib
from datetime import datetime, timedelta, timezone

import bcrypt
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI    = os.environ["MONGO_URI"]
DB_NAME      = os.getenv("DB_NAME", "nitd_campus")
PEPPER       = os.getenv("SERVER_PEPPER", "dev-pepper").encode()
AUTO_CLOSE_H = int(os.getenv("AUTO_CLOSE_HOUR", "23"))
STALE_HOURS  = int(os.getenv("STALE_HOURS", "6"))

IST = timezone(timedelta(hours=5, minutes=30))

# Rate limits. Layered deliberately: the whole campus sits behind one NAT address,
# so IP is a weak signal and must stay loose or one prankster locks out every owner.
# The precise signals are per-shop (brute force against a shop) and per-code
# (targeted guessing of one owner's code).
MAX_ATTEMPTS_PER_IP    = 40     # per hour — loose, only catches runaway scripts
MAX_ATTEMPTS_PER_SHOP  = 12     # per hour — main brute-force defence
MAX_FAILS_PER_CODE     = 5      # consecutive failures before that code locks
MAX_TOGGLES_PER_SHOP   = 6      # successful, per hour — caps damage from a leaked code
LOCKOUT_MINUTES        = 30

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)
db = _client[DB_NAME]

shops      = db.shops
status     = db.shop_status
staff      = db.shop_staff
audit      = db.shop_audit
attempts   = db.toggle_attempts


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_indexes() -> None:
    shops.create_index([("shop_id", ASCENDING)], unique=True)
    status.create_index([("shop_id", ASCENDING)], unique=True)
    staff.create_index([("code_lookup", ASCENDING)])
    staff.create_index([("shop_id", ASCENDING)])
    audit.create_index([("ts", ASCENDING)])
    audit.create_index([("shop_id", ASCENDING), ("ts", ASCENDING)])
    # attempts self-expire one hour after they are written
    attempts.create_index([("ts", ASCENDING)], expireAfterSeconds=3600)


# ---------------------------------------------------------------- codes
def code_lookup(code: str) -> str:
    """Deterministic, indexable. bcrypt is salted so it cannot be looked up directly."""
    return hmac.new(PEPPER, code.strip().upper().encode(), hashlib.sha256).hexdigest()


def code_hash(code: str) -> str:
    return bcrypt.hashpw(code.strip().upper().encode(), bcrypt.gensalt()).decode()


def code_verify(code: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(code.strip().upper().encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------- status
def last_reset_boundary(ref: datetime | None = None) -> datetime:
    """Most recent AUTO_CLOSE_HOUR (IST) before `ref`.

    Evaluated lazily at read time rather than by a nightly cron: a status written
    before this boundary is treated as 'no update today', so the morning never
    inherits yesterday's Open. Same effect as the scheduled job, nothing to babysit.
    """
    ref = (ref or now_utc()).astimezone(IST)
    boundary = ref.replace(hour=AUTO_CLOSE_H, minute=0, second=0, microsecond=0)
    if ref < boundary:
        boundary -= timedelta(days=1)
    return boundary.astimezone(timezone.utc)


def describe_age(delta: timedelta) -> str:
    mins = int(delta.total_seconds() // 60)
    if mins < 1:   return "just now"
    if mins < 60:  return f"{mins} minute{'s' if mins != 1 else ''} ago"
    hrs = mins // 60
    if hrs < 24:   return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    days = hrs // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def effective_status(shop: dict, st: dict | None) -> dict:
    """Render status by age + source. Never a bare yes/no — see plan §3."""
    if not shop.get("active", True):
        return {"state": "inactive", "is_open": None,
                "headline": "Not operating currently", "detail": "", "age": None}

    if not st or not st.get("updated_at"):
        return {"state": "no_update", "is_open": None,
                "headline": "No confirmed update today",
                "detail": "Please check directly.", "age": None}

    updated = st["updated_at"]
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)

    # written before tonight's reset boundary -> treat as unconfirmed, not as Closed.
    if updated < last_reset_boundary():
        return {"state": "no_update", "is_open": None,
                "headline": "No confirmed update today",
                "detail": "Please check directly.", "age": describe_age(now_utc() - updated)}

    age  = now_utc() - updated
    word = "Open" if st.get("is_open") else "Closed"
    ago  = describe_age(age)
    reason = f" ({st['reason']})" if st.get("reason") else ""

    if age > timedelta(hours=STALE_HOURS):
        return {"state": "stale", "is_open": bool(st.get("is_open")),
                "headline": word + reason,
                "detail": f"Last updated {ago}, so this may be out of date.", "age": ago}

    return {"state": "fresh", "is_open": bool(st.get("is_open")),
            "headline": word + reason,
            "detail": f"Updated {ago} by the shop.", "age": ago}


def shop_list() -> list[dict]:
    out = []
    for s in shops.find({}).sort("display_order", ASCENDING):
        st = status.find_one({"shop_id": s["shop_id"]})
        eff = effective_status(s, st)
        out.append({
            "shop_id": s["shop_id"], "name": s["name"],
            "aliases": s.get("aliases", []),
            "location": s.get("location", ""), "active": s.get("active", True),
            **eff,
        })
    return out


# ---------------------------------------------------------------- rate limiting
def log_attempt(ip_hash: str, shop_id: str, success: bool, lookup: str = "") -> None:
    attempts.insert_one({"ts": now_utc(), "ip_hash": ip_hash, "shop_id": shop_id,
                         "success": success, "lookup": lookup})


def is_rate_limited(ip_hash: str, shop_id: str, lookup: str = "") -> tuple[bool, str]:
    """Returns (blocked, reason). Reason is for the audit log only — the caller
    always replies with the same generic message so probing reveals nothing."""
    since = now_utc() - timedelta(hours=1)

    if attempts.count_documents({"shop_id": shop_id, "ts": {"$gte": since}}) >= MAX_ATTEMPTS_PER_SHOP:
        return True, "shop_attempt_limit"

    if attempts.count_documents({"shop_id": shop_id, "success": True,
                                 "ts": {"$gte": since}}) >= MAX_TOGGLES_PER_SHOP:
        return True, "shop_toggle_limit"

    # consecutive failures against this specific code
    if lookup:
        lock_since = now_utc() - timedelta(minutes=LOCKOUT_MINUTES)
        recent = list(attempts.find({"lookup": lookup, "ts": {"$gte": lock_since}})
                      .sort("ts", -1).limit(MAX_FAILS_PER_CODE))
        if len(recent) >= MAX_FAILS_PER_CODE and all(not a.get("success") for a in recent):
            return True, "code_locked"

    if attempts.count_documents({"ip_hash": ip_hash, "ts": {"$gte": since}}) >= MAX_ATTEMPTS_PER_IP:
        return True, "ip_limit"

    return False, ""


def write_audit(**kw) -> None:
    audit.insert_one({"ts": now_utc(), **kw})
