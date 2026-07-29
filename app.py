"""HTTP API for the chat widget and the campus shop-status endpoints.

    uvicorn app:app --reload --port 8000
"""
import hashlib
import logging
import re
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db as D

log = logging.getLogger(__name__)

app = FastAPI(title="NIT Delhi — Campus Status", version="1.0")


import os as _os
_origins = [o.strip() for o in _os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


GENERIC = {"ok": True, "message": "Thanks, your response has been recorded."}


class ToggleIn(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    is_open: bool
    reason: Optional[str] = Field(default=None, max_length=80)


def ip_hash(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    return hashlib.sha256((ip + "|" + D.PEPPER.decode()).encode()).hexdigest()[:32]


@app.on_event("startup")
def _startup():
    D.ensure_indexes()
    import cache
    cache.ensure_indexes()


@app.get("/api/shops")
def list_shops():
    return {"shops": D.shop_list(), "stale_hours": D.STALE_HOURS}


@app.get("/api/shops/{shop_id}/status")
def shop_status(shop_id: str):
    """Read path for the chatbot. Never cached — always recomputed."""
    s = D.shops.find_one({"shop_id": shop_id})
    if not s:
        return JSONResponse({"error": "unknown shop"}, status_code=404)
    eff = D.effective_status(s, D.status.find_one({"shop_id": shop_id}))
    return {"shop_id": shop_id, "name": s["name"], **eff}


@app.post("/api/shops/{shop_id}/toggle")
def toggle(shop_id: str, body: ToggleIn, request: Request):
    iph = ip_hash(request)
    lookup = D.code_lookup(body.code)

    blocked, why = D.is_rate_limited(iph, shop_id, lookup)
    if blocked:
        D.log_attempt(iph, shop_id, False, lookup)
        D.write_audit(shop_id=shop_id, staff_id=None, action="toggle",
                      success=False, ip_hash=iph, note=why)
        return GENERIC

    shop = D.shops.find_one({"shop_id": shop_id, "active": True})
    member = D.staff.find_one({"code_lookup": lookup, "is_active": True})


    valid = bool(shop and member
                 and member["shop_id"] == shop_id
                 and D.code_verify(body.code, member["code_hash"]))

    D.log_attempt(iph, shop_id, valid, lookup)
    if not valid:
        D.write_audit(shop_id=shop_id,
                      staff_id=member["staff_id"] if member else None,
                      action="toggle", success=False, ip_hash=iph,
                      note="bad_code_or_wrong_shop")
        return GENERIC

    prev = D.status.find_one({"shop_id": shop_id})
    D.status.update_one(
        {"shop_id": shop_id},
        {"$set": {"shop_id": shop_id, "is_open": body.is_open,
                  "updated_by": member["staff_id"], "updated_at": D.now_utc(),
                  "reason": (body.reason or "").strip() or None,
                  "source": "owner"}},
        upsert=True)

    D.write_audit(shop_id=shop_id, staff_id=member["staff_id"], action="toggle",
                  success=True, ip_hash=iph,
                  old_status=(prev or {}).get("is_open"), new_status=body.is_open,
                  reason=body.reason)


    return {"ok": True, "message": f"{shop['name']} marked "
                                   f"{'Open' if body.is_open else 'Closed'}."}


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    category: Optional[str] = None
    slots: Optional[dict] = None


def guess_category(q: str):
    """Used only when no chip is selected."""
    ql = q.lower()
    if re.search(r"canteen|nescafe|amul|health cent|hk cafe|\bcafe\b"
                 r"|(open|closed|shut)\s*(now|today|yet)?\??$", ql):
        return "shops"
    if re.search(r"calendar|holiday|vacation|convocation|mid[- ]?sem|end[- ]?sem"
                 r"|exam date|teaching days|working days|registration"
                 r"|diwali|dussehra|christmas|independence day|janmashtami"
                 r"|gandhi|guru nanak|milad|\beid\b"
                 r"|when (do|does|is|are).{0,20}(class|exam|registration|result)", ql):
        return "calendar"
    if re.search(r"document|checklist|reporting|bring|verificat|\badmission\b", ql):
        return "admission"
    if re.search(r"\bfee|fees|cost|tuition|tution|charge|how much|kitna|ifsc|caution", ql):
        return "fee"
    if re.search(r"\blabs?\b|\blaborator|room\s*(no|number)", ql):
        return "lab"
    if re.search(r"\b(who is|hod|head of|faculty|staff|professor|designation|coordinator)\b", ql):
        return "faculty"
    if re.search(r"\b(dr|prof|mr|ms|mrs)\.?\s*[a-z]", ql):
        return "faculty"
    if re.search(r"syllabus|module|unit|subject|course|semester", ql):
        return "syllabus"
    if re.search(r"vision|mission|goal|about", ql):
        return "about"
    return None


LLM_CATEGORIES = ("syllabus", "about", "any")

RATE_LIMITED_BODY = ("You're sending questions faster than I can answer. "
                     "Give it a minute and try again.")


def _rate_limited(reason: str, retry_after: int) -> JSONResponse:
    return JSONResponse(
        {"answer": RATE_LIMITED_BODY, "source": None,
         "method": f"rate-limited ({reason})", "retry_after": retry_after},
        status_code=429, headers={"Retry-After": str(retry_after)})


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    """Answer within a category.

    Fees, labs, faculty and admissions are template lookups over MongoDB, so a
    figure or room number cannot be invented. Only the syllabus and about
    categories reach a model, and they fall back to source text without one.
    """


    import cache
    import ratelimit as RL
    import rag_handlers as H
    from config import CFG

    question = body.message.strip()
    category = (body.category or "").lower()
    state = dict(body.slots or {})
    client = RL.client_id(request)
    limits_on = CFG.get("ratelimit.enabled", True)

    if limits_on:
        allowed, reason, retry_after = RL.check(client, "request")
        if not allowed:
            return _rate_limited(reason, retry_after)
        RL.record(client, "request", "/api/chat")

    if category not in H.HANDLERS:
        category = guess_category(question) or "any"


    cache_key = cache.key(question, category, state)
    cached = cache.get_by_key(cache_key)
    if cached:
        cached["category"] = category
        cached["state"] = state
        return cached


    if category in LLM_CATEGORIES and limits_on:
        allowed, _, _ = RL.check(client, "llm")
        if not allowed:
            state["llm_budget_exhausted"] = True

    try:
        answer = H.HANDLERS[category](question, state)
    except Exception as exc:
        log.exception("Handler %r failed for %r", category, question[:80])
        return {"answer": "Something went wrong answering that. Please try rephrasing.",
                "source": None, "method": f"error: {str(exc)[:80]}"}

    answer.setdefault("source", None)
    answer.setdefault("method", category)
    if "+ LLM" in answer["method"]:
        RL.record(client, "llm", "/api/chat")
        cache.put_by_key(cache_key, question, category, answer)

    answer["category"] = category
    answer["state"] = state
    return answer


@app.get("/api/ops")
def ops(request: Request):
    """Cache and rate-limit visibility — useful during a demo, and the thing
    you actually want when the bot starts feeling slow."""
    import cache, ratelimit as RL, store
    return {"store_backend": store.backend(),
            "cache": cache.stats(),
            "ratelimit": RL.stats(RL.client_id(request))}


@app.get("/api/config")
def ui_config():
    """The widget builds itself from this — chips, suggestions, greeting and
    footer all come from the `config` collection, nothing baked into the JS."""
    from config import CFG
    return {"categories": CFG.get("ui.categories"),
            "quick":      CFG.get("ui.quick"),
            "greeting":   CFG.get("ui.greeting"),
            "placeholder": CFG.get("ui.placeholder"),
            "footer":     CFG.get("ui.footer")}


@app.get("/api/rag/stats")
def rag_stats():
    import rag_core, rag_handlers
    return {**rag_core.stats(), "llm_configured": rag_handlers.llm_available()}


@app.post("/api/feedback")
def feedback(payload: dict):
    """Student feedback stub. Phase 2 gates this behind college-email verification."""
    D.db.feedback.insert_one({"ts": D.now_utc(),
                              "text": str(payload.get("text", ""))[:2000],
                              "email": str(payload.get("email", ""))[:120]})
    return GENERIC


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():


    return FileResponse("static/index.html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache", "Expires": "0"})
