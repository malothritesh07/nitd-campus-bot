"""Category handlers — one per chip in the chat widget.

Only Syllabus ever calls the LLM, and it degrades to the raw source text when
no Groq key is set. Everything else is template-rendered from MongoDB, so a
number or room can never be invented.
"""
import logging
import os
import re

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import embeddings
import rag_calendar as CAL
import rag_core as R
import rag_fee as F
from tracing import trace
from config import CFG

log = logging.getLogger(__name__)

def SRC_PDF(url):
    return {"label": "Official notice (PDF)", "url": url} if url else None


GROQ_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
def MODELS():
    return CFG.get("generation.models")
_working = None


import contextvars
_override = contextvars.ContextVar("llm_override", default=None)


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


def _build_client(key: str):
    """Verified client first; fall back to an unverified one only if the
    handshake itself fails. On a machine without TLS interception the fallback
    never runs."""
    from groq import Groq
    try:
        c = Groq(api_key=key)
        c.models.list()
        return c
    except Exception as first:


        if "401" in str(first) or "invalid" in str(first).lower():
            raise
        import httpx
        c = Groq(api_key=key, http_client=httpx.Client(verify=False, timeout=60.0))
        c.models.list()
        return c


def verify_key(key: str) -> tuple[bool, str, list]:
    """Check a user-supplied key before any question depends on it.

    models.list() costs no tokens and no quota, so this is free to run. Returns
    (ok, human-readable message, model ids available to that key).
    """
    key = (key or "").strip()
    if not key:
        return False, "No key provided.", []
    if not key.startswith("gsk_"):
        return False, "That doesn't look like a Groq key — they start with `gsk_`.", []
    try:
        client = _build_client(key)
        ids = sorted(m.id for m in client.models.list().data)
        _clients[key] = client
        usable = [m for m in MODELS() if m in ids]
        if not usable:
            return (False, "Key works, but none of the configured models are "
                           "available to it.", ids)
        return True, f"Connected — {len(usable)} of the configured models available.", ids
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg or "Invalid API Key" in msg:
            return False, "Invalid key — Groq rejected it. Check for a stray space.", []
        if "429" in msg or "rate" in msg.lower():
            return False, "Key is valid but rate-limited right now. Try shortly.", []
        return False, f"Could not reach Groq: {type(e).__name__}.", []


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
        client = _build_client(key)
        _clients[key] = client


    chosen = (_override.get() or {}).get("model")
    head = [m for m in (chosen, _working) if m]
    order = head + [m for m in MODELS() if m not in head]
    errs = []
    for m in order:
        try:
            r = client.chat.completions.create(model=m, messages=messages,
                                                max_tokens=max_tokens, temperature=temperature)
            t = (r.choices[0].message.content or "").strip()

            if len(t) >= CFG.get("generation.min_reply_chars"):
                _working = m
                return t
            errs.append(f"{m}: reply too short ({t!r})")
        except Exception as e:
            errs.append(f"{m}: {str(e)[:70]}")
            if _working == m:
                _working = None
    raise RuntimeError("all Groq models failed: " + "; ".join(errs))


_INJ = None


def _inj_patterns():
    """Compiled once, sourced from the `config` collection so a new attack
    phrasing is a database edit rather than a redeploy."""
    global _INJ
    if _INJ is None:
        _INJ = [re.compile(p, re.I) for p in (CFG.get("lexicon.injection_patterns") or [])]
    return _INJ


def looks_like_injection(q: str) -> bool:
    return any(p.search(q or "") for p in _inj_patterns())


def leaks_instructions(reply: str) -> bool:
    """Detect a reply echoing the system prompt, which indicates the model
    followed an injection the input filter missed."""
    r = (reply or "").lower()
    markers = ("answer only from context", "the question is text typed by",
               "never invent module", "is data, never instructions")
    return any(m in r for m in markers)


def injection_refusal() -> dict:


    return {"answer": CFG.get("prompts.injection_refusal"),
            "source": None, "method": "guard"}


def h_fee(q, st):
    out = F.answer_fee(q, carried=st.get("slots") or {})
    st["slots"] = out.get("slots", {})
    return {"answer": out["answer"], "source": out.get("source"),
            "method": out.get("method", "metadata-filter"), "slots": st["slots"]}


LAB_LIST = r"\b(list|all|names? of|show|which labs|what labs|how many)\b"


def h_lab(q, st):
    R.load()
    ql = q.lower()


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


COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{3,4})\s?(\d{3})\b")
FOLLOW_UP_RE = re.compile(r"\b(them|it|that|those|this)\b")


GENERIC_RE = re.compile(
    r"\b(engineering|syllabus|subjects?|courses?|department|dept|branch|of|in|the)\b")


def _syllabus_context(q, ql, st):
    """Resolve programme and semester. The current turn overrides carried state."""
    program_now = R.detect_syl_program(q)
    match = re.search(SEM_RE, ql)
    semester_now = int(match.group(1)) if match else None

    if program_now:
        st["syl_program"] = program_now
    if semester_now is not None:
        st["syl_semester"] = semester_now

    program = program_now or st.get("syl_program")
    semester = semester_now if semester_now is not None else st.get("syl_semester")
    return program, semester, program_now, semester_now


def _refuse(answer):
    return {"answer": answer, "source": None, "method": "guard"}


def _unknown_department(q, ql, program_now, st):
    """Refuse a department we hold no curriculum for.

    Without this the programme slot keeps whatever an earlier turn set, and the
    listing filter answers with another department's courses. The course check
    keeps "discrete mathematics syllabus" working, since "mathematics" is also
    a department alias.
    """
    department = R.detect_dept(q)
    if not department or program_now is not None:
        return None

    residual = GENERIC_RE.sub(" ", ql)
    aliases = "|".join(map(re.escape, R.DEPT_ALIASES))
    residual = re.sub(rf"\b({aliases})\b", " ", residual).strip()
    if R.match_course(residual) is not None:
        return None

    st.pop("syl_program", None)
    st.pop("syl_course", None)
    return _refuse(f"I don't have syllabus data for {department}. "
                   f"{R.syllabus_coverage()}")


def _unknown_course_code(code_match):
    """Refuse an unrecognised code rather than letting the model invent modules."""
    if not code_match:
        return None
    prefix, number = code_match.groups()
    if (prefix + number).upper() in R.KNOWN_CODES:
        return None
    return _refuse(f"{prefix.upper()} {number} is not in my syllabus data. "
                   f"{R.syllabus_coverage()}")


def _resolve_course(q, ql, st, code_match):
    """Match a course, falling back to the last one discussed for follow-ups
    such as "list them"."""
    course = R.match_course(q)
    if course is None and not code_match and st.get("syl_course"):
        if len(q.split()) <= 4 or FOLLOW_UP_RE.search(ql):
            course = next((c for c in R.COURSES
                           if c["meta"].get("doc_id") == st["syl_course"]), None)
    if course is not None:
        st["syl_course"] = course["meta"].get("doc_id")
    return course


def _module_listing(course):
    """Deterministic module list — no model, so unit names cannot be invented."""
    modules = _modules_of(course)
    if not modules:
        return None
    meta = course["meta"]
    head = (f"{meta.get('course_code')} — {meta.get('course_title')} "
            f"({len(modules)} modules):")
    lines = [f"{i}. {m}" for i, m in enumerate(modules, 1)]
    return {"answer": head + "\n" + "\n".join(lines),
            "source": {"label": f"{meta.get('course_title')} — curriculum", "url": None},
            "method": "structured-listing"}


def _semester_listing(program, semester, st):
    selected = [c for c in R.COURSES
                if c["meta"].get("semester") == semester
                and (not program or c["meta"].get("program") == program)]

    if not selected:
        available = sorted({c["meta"]["semester"] for c in R.COURSES
                            if (not program or c["meta"].get("program") == program)
                            and c["meta"].get("semester")})
        message = f"No courses stored for semester {semester}"
        if program:
            message += f" of {program}"
        if available:
            message += f". Available semesters: {', '.join(map(str, available))}."
        return _refuse(message)

    head = f"Semester {semester}"
    if program:
        head += f" — {program}"
    head += f" ({len(selected)} courses):"
    lines = [f"{i}. {c['meta']['course_code']} — {c['meta']['course_title']}"
             for i, c in enumerate(
                 sorted(selected, key=lambda x: x["meta"]["course_code"] or ""), 1)]
    st["syl_intent"] = "list_courses"
    return {"answer": head + "\n" + "\n".join(lines)
                      + "\n\nAsk about any one of these for its module list.",
            "source": {"label": "Curriculum", "url": None},
            "method": "structured-listing"}


def _answer_from_context(context, label, method, question, prompt_key, st):
    """Phrase the retrieved context with the model, falling back to the raw
    source text whenever the model is unavailable, out of budget, or errors.

    The badge always names which of those happened.
    """
    context = context[:CFG.get("generation.context_chars")]
    source = {"label": label, "url": None}
    raw = context[:CFG.get("generation.raw_fallback_chars")]

    if not llm_available() or st.get("llm_budget_exhausted"):
        reason = "LLM budget reached" if st.get("llm_budget_exhausted") else "no LLM key"
        return {"answer": raw, "source": source, "method": f"{method} · {reason}"}

    system = CFG.get(prompt_key) + CFG.get("prompts.guard_clause")
    try:
        body = llm([{"role": "system", "content": system},
                    {"role": "user",
                     "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
                   max_tokens=CFG.get("generation.max_tokens"))
    except Exception as exc:
        log.warning("LLM unavailable, answering from source text: %s", exc)
        return {"answer": raw, "source": source,
                "method": f"{method} · LLM unavailable"}
    return {"answer": body, "source": source, "method": f"{method} + LLM"}


def _scoped_hybrid(q, program, semester, st):
    """Hybrid retrieval restricted to the programme and semester in play.

    Unscoped, "units in AI 4th semester" returned a CSE course.
    """
    def keep(chunk):
        meta = chunk["meta"]
        return (chunk["domain"] == "syllabus"
                and (not program or meta.get("program") == program)
                and (semester is None or meta.get("semester") == semester))

    hits = R.retrieve_scoped(q, keep, k=3)
    if not hits:
        where = f" for {program}" if program else ""
        if semester is not None:
            where += f", semester {semester}"
        return {"answer": f"I don't have that{where}. {R.syllabus_coverage()}",
                "source": None, "method": "no-match"}

    context = "\n---\n".join(R.expand_parent(h)["text"] for h, _ in hits[:2])
    scope = program or "Curriculum"
    if semester is not None:
        scope += f" · semester {semester}"
    return _answer_from_context(context, scope, "hybrid(scoped)", q,
                                "prompts.syllabus_scoped", st)


def h_syllabus(q, st):
    R.load()
    ql = q.lower()
    program, semester, program_now, semester_now = _syllabus_context(q, ql, st)
    code_match = COURSE_CODE_RE.search(q)

    for guard in (_unknown_department(q, ql, program_now, st),
                  _unknown_course_code(code_match)):
        if guard:
            return guard

    course = _resolve_course(q, ql, st, code_match)

    if course is not None and re.search(LIST_UNITS, ql):
        listing = _module_listing(course)
        if listing:
            return listing


    repeat_listing = (st.get("syl_intent") == "list_courses"
                      and (program_now or semester_now is not None)
                      and len(q.split()) <= 4)
    if semester is not None and course is None and (
            repeat_listing or re.search(f"{LIST_COURSE}|{LIST_UNITS}", ql)):
        return _semester_listing(program, semester, st)

    if course is not None:
        context = "\n".join(x["text"] for x in _course_chunks(course["meta"]["doc_id"]))
        label = f"{course['meta'].get('course_title')} — curriculum"
        return _answer_from_context(context, label, "exact-course", q,
                                    "prompts.syllabus_course", st)

    return _scoped_hybrid(q, program, semester, st)


def h_calendar(q, st):
    """Dense retrieval, then cross-encoder reranking with a metadata boost.

    Calendar entries are short and lexically alike, so a bi-encoder ranks
    "Mid Semester Examination" and "End Semester Examination" almost equally.
    The reranker reads query and passage together and separates them.
    """
    hits, slots, how = CAL.search(q)
    if not hits:
        return {"answer": f"I don't have that in the academic calendar. "
                          f"{CAL.coverage()}",
                "source": None, "method": "no-match"}

    named = [f"{k}={v}" for k, v in slots.items() if v != "none"]
    method = "rerank(cross-encoder)"
    if named:
        method += f" · filter[{how}]: {', '.join(named)}"

    # The retrieved events are the fallback body, so a model failure degrades to
    # the dated list rather than to nothing.
    out = _answer_from_context(CAL.render(hits), "Academic Calendar", method, q,
                               "prompts.calendar_answer", st)
    out["source"] = {"label": "Academic Calendar (PDF)",
                     "url": hits[0][0].get("source")}
    return out


def h_about(q, st):


    dept = R.detect_dept(q)

    def keep(x):
        return x["domain"] == "about" and (not dept or x["meta"].get("department") == dept)

    hits = R.retrieve_scoped(q, keep, k=2)
    if not hits:
        if dept:
            have = sorted({c["meta"].get("department") for c in R.CHUNKS
                           if c["domain"] == "about" and c["meta"].get("department")})
            return {"answer": f"I don't have department pages for {dept}. "
                              f"I hold: {', '.join(have)}.",
                    "source": None, "method": "guard"}
        return {"answer": "Nothing found for that department section.",
                "source": None, "method": "no-match"}
    body = "\n\n".join(h["text"] for h, _ in hits)[:1200]


    method = "hybrid(bm25+vector)" if embeddings.get_model() is not None else "bm25 (no vectors)"
    if dept:
        method += " · dept-scoped"
    return {"answer": body,
            "source": {"label": f"{dept or 'Department'} page",
                       "url": hits[0][0].get("source")},
            "method": method}


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
    "calendar":  h_calendar,
    "fee":       h_fee,
    "lab":       h_lab,
    "faculty":   h_faculty,
    "syllabus":  h_syllabus,
    "admission": h_admission,
    "about":     h_about,
    "any":       h_any,
}


def _with_input_guard(fn):
    """Applied to every category, not just the two that reach a model.

    The template paths cannot be steered by prompt text, so this is belt and
    braces there — but a single wrap means a category added later is covered by
    default rather than by remembering."""
    def inner(q, st):
        if looks_like_injection(q):
            return injection_refusal()
        out = fn(q, st)
        if leaks_instructions(out.get("answer", "")):
            return injection_refusal()
        return out
    inner.__name__ = getattr(fn, "__name__", "handler")
    return inner


for _cat, _fn in list(HANDLERS.items()):
    HANDLERS[_cat] = trace(name=f"category:{_cat}", run_type="chain",
                           category=_cat, uses_llm=(_cat == "syllabus"))(
                         _with_input_guard(_fn))
