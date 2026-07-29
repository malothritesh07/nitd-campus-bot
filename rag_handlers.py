"""Category handlers — one per chip in the chat widget.

Only Syllabus ever calls the LLM, and it degrades to the raw source text when
no Groq key is set. Everything else is template-rendered from MongoDB, so a
number or room can never be invented.
"""
import os, re

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import db as D
import rag_core as R
import rag_fee as F
from tracing import trace
from config import CFG

SRC_PDF = lambda url: {"label": "Official notice (PDF)", "url": url} if url else None

# ------------------------------------------------------------------- Groq
GROQ_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
def MODELS():
    return CFG.get("generation.models")
_working = None

# Per-session overrides so a visitor can supply their own key and model without
# touching the deployment's. A ContextVar rather than a module global: Streamlit
# runs every session in its own thread, and a global would let one visitor's key
# leak into another's request.
import contextvars
_override = contextvars.ContextVar("llm_override", default=None)

# Clients are cached per key — building one costs a models.list() round trip.
_clients: dict = {}


def set_llm_override(api_key: str | None = None, model: str | None = None):
    """Called once per request. Passing None for both restores the deployment
    default."""
    _override.set({"api_key": (api_key or "").strip() or None,
                   "model": (model or "").strip() or None})


def active_key() -> str:
    return ((_override.get() or {}).get("api_key") or GROQ_KEY or "").strip()


def llm_available():
    return bool(active_key())


@trace(name="groq", run_type="llm")
def llm(messages, max_tokens=None, temperature=None):
    """Raises if no key is configured — callers fall back to raw context."""
    max_tokens  = max_tokens  or CFG.get("generation.max_tokens")
    temperature = CFG.get("generation.temperature") if temperature is None else temperature
    global _working
    key = active_key()
    if not key:
        raise RuntimeError("no GROQ_API_KEY set")
    client = _clients.get(key)
    if client is None:
        from groq import Groq
        # This laptop sits behind TLS interception, so the default cert bundle
        # rejects api.groq.com (the same thing broke nitdelhi.ac.in and
        # HuggingFace). Try the normal verified client first and only fall back
        # to an unverified one if the handshake fails. On a machine without the
        # proxy the fallback never runs.
        try:
            c = Groq(api_key=key)
            c.models.list()
            client = c
        except Exception:
            import httpx
            client = Groq(api_key=key,
                          http_client=httpx.Client(verify=False, timeout=60.0))
        _clients[key] = client

    # An explicitly chosen model goes first; the rest stay as fallbacks so one
    # decommissioned model can't take the category down.
    chosen = (_override.get() or {}).get("model")
    head = [m for m in (chosen, _working) if m]
    order = head + [m for m in MODELS() if m not in head]
    errs = []
    for m in order:
        try:
            r = client.chat.completions.create(model=m, messages=messages,
                                                max_tokens=max_tokens, temperature=temperature)
            t = (r.choices[0].message.content or "").strip()
            # a one-character reply ('>' came back once from a prompt-injection
            # attempt) is not an answer — try the next model rather than ship it
            if len(t) >= CFG.get("generation.min_reply_chars"):
                _working = m
                return t
            errs.append(f"{m}: reply too short ({t!r})")
        except Exception as e:
            errs.append(f"{m}: {str(e)[:70]}")
            if _working == m: _working = None
    raise RuntimeError("all Groq models failed: " + "; ".join(errs))


# ================================================================ handlers
def h_fee(q, st):
    out = F.answer_fee(q, carried=st.get("slots") or {})
    st["slots"] = out.get("slots", {})
    return {"answer": out["answer"], "source": out.get("source"),
            "method": out.get("method", "metadata-filter"), "slots": st["slots"]}


LAB_LIST = r"\b(list|all|names? of|show|which labs|what labs|how many)\b"


def h_lab(q, st):
    R.load()
    ql = q.lower()

    # "list the ece labs" is a roster request, not a name lookup. Without this it
    # fell through to fuzzy matching and suggested two unrelated labs.
    dept = R.detect_dept(q)
    if dept and re.search(LAB_LIST, ql):
        sel = [c for c in R.LABS if c["meta"].get("department") == dept]
        if sel:
            if re.search(r"how many|count|number of", ql):
                caps = [c["meta"].get("capacity") for c in sel
                        if isinstance(c["meta"].get("capacity"), int)]
                body = (f"{dept}: {len(sel)} labs"
                        + (f", total listed capacity {sum(caps)}." if caps else "."))
            else:
                lines = []
                for i, c in enumerate(sorted(sel, key=lambda x: x["meta"].get("name") or ""), 1):
                    loc = c["meta"].get("raw")
                    lines.append(f"{i}. {c['meta']['name']}" + (f" — {loc}" if loc else ""))
                body = f"{dept} — {len(sel)} labs:\n" + "\n".join(lines)
            return {"answer": body,
                    "source": {"label": f"{dept} — Laboratories", "url": sel[0].get("source")},
                    "method": "structured-listing"}

    # "hardware in it?" -> reuse the lab from the previous turn
    if (re.search(R.PRONOUN, ql) or len(q.split()) <= 3) and st.get("lab"):
        if not R.retrieve_lab(q):
            q = f"{st['lab']} {q}"
    hits = R.retrieve_lab(q)
    if not hits:
        sug = R.lab_suggestions(q)
        return {"answer": ("No lab matches that name.\nDid you mean:\n" +
                           "\n".join(f"  - {s}" for s in sug)) if sug else
                          "No such lab. Try the full lab name or its room number.",
                "source": None, "method": "no-match"}
    st["lab"] = hits[0][0]["meta"]["name"]
    body = R.render_lab(q) or f"{st['lab']} — that detail isn't listed."
    return {"answer": body,
            "source": {"label": f"{hits[0][0]['meta'].get('department')} — Laboratories",
                       "url": hits[0][0].get("source")},
            "method": "exact-lookup"}


PRIVATE_ASK = r"\b(phone|mobile|whatsapp|home address|residence|personal number|salary|aadhaar)\b"


def h_faculty(q, st):
    body = R.render_entity(q)
    if not body:
        return {"answer": "No matching person. Try a department name to see the full list.",
                "source": None, "method": "no-match"}
    dept = R.detect_dept(q)
    if re.search(PRIVATE_ASK, q.lower()):
        # answering the name-lookup silently would look like we'd checked and
        # found nothing private; say plainly that we don't publish it
        return {"answer": body + "\n\nI only share what the institute publishes. "
                                 "Personal phone numbers and home addresses aren't "
                                 "available here — use the department office contact "
                                 "on nitdelhi.ac.in.",
                "source": {"label": f"{dept} — Faculty" if dept else "Department directory",
                           "url": None},
                "method": "fuzzy-lexical + privacy-note"}
    return {"answer": body,
            "source": {"label": f"{dept} — Faculty" if dept else "Department directory",
                       "url": None},
            "method": "fuzzy-lexical"}


def h_admission(q, st):
    if re.search(r"dasa|mca|nimcet|josaa|csab|b\.?tech|m\.?tech", q.lower()):
        st["adm"] = q
    res = R.render_admission(st.get("adm", q))
    body, url = res if isinstance(res, tuple) else (res, None)
    return {"answer": body, "source": SRC_PDF(url), "method": "whole-document"}


SEM_RE      = r"(\d)\s*(?:st|nd|rd|th)?\s*sem"
LIST_UNITS  = r"unit|module|topic|chapter|syllabus content"
LIST_COURSE = r"subject|course|paper|syllabus list|what.*study|what.*subjects"


def _course_chunks(doc_id):
    return [x for x in R.CHUNKS
            if x["domain"] == "syllabus" and x["meta"].get("doc_id") == doc_id]


def _modules_of(course):
    """Deterministic module list — no LLM, so unit names cannot be invented."""
    mods = [x for x in _course_chunks(course["meta"]["doc_id"])
            if x["meta"].get("granularity") == "module"]
    mods.sort(key=lambda x: str(x["meta"].get("module_label") or ""))
    out = []
    for m in mods:
        label = m["meta"].get("module_label") or "Module"
        body = re.sub(r"^.*?\n", "", m["text"], count=1).strip()
        body = re.sub(r"^Module[-\s]?[IVX0-9]+[:.]?\s*(\(\d+\s*Hours?\))?\s*", "", body)
        out.append(f"{label}: {body[:160]}")
    return out


def h_syllabus(q, st):
    R.load()
    ql = q.lower()

    # ---- resolve context: what this turn says beats what was carried ----
    prog_now = R.detect_syl_program(q)
    m_sem    = re.search(SEM_RE, ql)
    sem_now  = int(m_sem.group(1)) if m_sem else None
    code_m   = re.search(r"\b([A-Za-z]{3,4})\s?(\d{3})\b", q)

    prog = prog_now or st.get("syl_program")
    sem  = sem_now if sem_now is not None else st.get("syl_semester")
    if prog_now:            st["syl_program"] = prog_now
    if sem_now is not None: st["syl_semester"] = sem_now

    # ---- unknown course code: refuse instead of inventing modules ----
    if code_m:
        cc = (code_m.group(1) + code_m.group(2)).upper()
        if cc not in R.KNOWN_CODES:
            return {"answer": f"{code_m.group(1).upper()} {code_m.group(2)} is not in my syllabus "
                              f"data. {R.syllabus_coverage()}",
                    "source": None, "method": "guard"}

    # ---- which course is in play? ----
    course = R.match_course(q)
    if course is None and not code_m and st.get("syl_course"):
        # "list them" / "module names?" should stay on the last course discussed
        if len(q.split()) <= 4 or re.search(r"\b(them|it|that|those|this)\b", ql):
            course = next((c for c in R.COURSES
                           if c["meta"].get("doc_id") == st["syl_course"]), None)
    if course is not None:
        st["syl_course"] = course["meta"].get("doc_id")

    # ---- named course + "units/modules" -> deterministic list, no LLM ----
    if course is not None and re.search(LIST_UNITS, ql):
        mods = _modules_of(course)
        m = course["meta"]
        if mods:
            head = f"{m.get('course_code')} — {m.get('course_title')} ({len(mods)} modules):"
            lines = [f"{i}. {x}" for i, x in enumerate(mods, 1)]
            return {"answer": head + "\n" + "\n".join(lines),
                    "source": {"label": f"{m.get('course_title')} — curriculum", "url": None},
                    "method": "structured-listing"}

    # A bare "in cse?" after a semester listing means "same question, other
    # programme" — carry the intent, not just the slots.
    repeat_listing = (st.get("syl_intent") == "list_courses"
                      and (prog_now or sem_now is not None)
                      and len(q.split()) <= 4)

    # ---- "subjects / units in Nth semester [of PROG]" -> course listing ----
    if sem is not None and course is None and (repeat_listing
                                               or re.search(LIST_COURSE + "|" + LIST_UNITS, ql)):
        sel = [c for c in R.COURSES if c["meta"].get("semester") == sem
               and (not prog or c["meta"].get("program") == prog)]
        if sel:
            head = f"Semester {sem}" + (f" — {prog}" if prog else "") + f" ({len(sel)} courses):"
            lines = [f"{i}. {c['meta']['course_code']} — {c['meta']['course_title']}"
                     for i, c in enumerate(sorted(sel, key=lambda x: x["meta"]["course_code"] or ""), 1)]
            tail = "\n\nAsk about any one of these for its module list."
            st["syl_intent"] = "list_courses"
            return {"answer": head + "\n" + "\n".join(lines) + tail,
                    "source": {"label": "Curriculum", "url": None},
                    "method": "structured-listing"}
        have = sorted({c["meta"]["semester"] for c in R.COURSES
                       if (not prog or c["meta"].get("program") == prog)
                       and c["meta"].get("semester")})
        msg = f"No courses stored for semester {sem}" + (f" of {prog}" if prog else "")
        if have:
            msg += f". Available semesters: {', '.join(map(str, have))}."
        return {"answer": msg, "source": None, "method": "guard"}

    # ---- a specific course, open question -> that course only ----
    if course is not None:
        ctx = "\n".join(x["text"] for x in _course_chunks(course["meta"]["doc_id"]))[:CFG.get("generation.context_chars")]
        m = course["meta"]
        label = f"{m.get('course_title')} — curriculum"
        # budget exhausted is treated exactly like no key: answer from source
        if not llm_available() or st.get("llm_budget_exhausted"):
            why = "LLM budget reached" if st.get("llm_budget_exhausted") else "no LLM key"
            return {"answer": ctx[:CFG.get("generation.raw_fallback_chars")], "source": {"label": label, "url": None},
                    "method": f"exact-course · {why}"}
        try:
            body = llm([{"role": "system", "content":
                         CFG.get("prompts.syllabus_course")},
                        {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}],
                       max_tokens=CFG.get('generation.max_tokens'))
            return {"answer": body, "source": {"label": label, "url": None},
                    "method": "exact-course + LLM"}
        except Exception:
            return {"answer": ctx[:CFG.get("generation.raw_fallback_chars")], "source": {"label": label, "url": None},
                    "method": "exact-course · LLM unavailable"}

    # ---- fallback: hybrid, SCOPED to the programme/semester in play ----
    # Unscoped, "units in AI 4th semester" was answered with CSBB 252 — a CSE course.
    def keep(x):
        if x["domain"] != "syllabus":
            return False
        if prog and x["meta"].get("program") != prog:
            return False
        if sem is not None and x["meta"].get("semester") != sem:
            return False
        return True

    hits = R.retrieve_scoped(q, keep, k=3)
    if not hits:
        where = (" for " + prog) if prog else ""
        if sem is not None:
            where += f", semester {sem}"
        return {"answer": f"I don't have that{where}. {R.syllabus_coverage()}",
                "source": None, "method": "no-match"}

    ctx = "\n---\n".join(R.expand_parent(h)["text"] for h, _ in hits[:2])[:CFG.get("generation.context_chars")]
    scope = (prog if prog else "Curriculum") + (f" · semester {sem}" if sem is not None else "")
    if not llm_available() or st.get("llm_budget_exhausted"):
        why = "LLM budget reached" if st.get("llm_budget_exhausted") else "no LLM key"
        return {"answer": ctx[:CFG.get("generation.raw_fallback_chars")], "source": {"label": scope, "url": None},
                "method": f"hybrid(scoped) · {why}"}
    try:
        body = llm([{"role": "system", "content":
                     CFG.get("prompts.syllabus_scoped")},
                    {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}],
                   max_tokens=CFG.get('generation.max_tokens'))
        return {"answer": body, "source": {"label": scope, "url": None},
                "method": "hybrid(scoped) + LLM"}
    except Exception:
        return {"answer": ctx[:CFG.get("generation.raw_fallback_chars")], "source": {"label": scope, "url": None},
                "method": "hybrid(scoped) · LLM unavailable"}


def h_about(q, st):
    hits = R.retrieve_scoped(q, lambda x: x["domain"] == "about", k=2)
    if not hits:
        return {"answer": "Nothing found for that department section.",
                "source": None, "method": "no-match"}
    body = "\n\n".join(h["text"] for h, _ in hits)[:1200]
    return {"answer": body,
            "source": {"label": "Department page", "url": hits[0][0].get("source")},
            "method": "hybrid(bm25+vector)"}


def h_any(q, st):
    """Uncategorised: try the exact lookups first, then fee, then syllabus."""
    body = R.render_entity(q)
    if body:
        return {"answer": body, "source": None, "method": "fuzzy-lexical"}
    if R.retrieve_lab(q):
        return h_lab(q, st)
    slots = F.extract_slots(q)
    if any(slots.get(k) for k in ("program", "semester", "admission_type", "residence")):
        rows = F.filter_rows({k: v for k, v in slots.items() if v})
        if rows and len(rows) <= 6:
            return h_fee(q, st)
    return h_syllabus(q, st)


HANDLERS = {
    "fee":       h_fee,
    "lab":       h_lab,
    "faculty":   h_faculty,
    "syllabus":  h_syllabus,
    "admission": h_admission,
    "about":     h_about,
    "any":       h_any,
}

# Trace every category, not just the ones that call a model. Most answers here
# never touch an LLM, and those are exactly the paths where the interesting
# failures live — wrong slot extracted, wrong person matched, filter too broad.
for _cat, _fn in list(HANDLERS.items()):
    HANDLERS[_cat] = trace(name=f"category:{_cat}", run_type="chain",
                           category=_cat, uses_llm=(_cat == "syllabus"))(_fn)
