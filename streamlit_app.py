"""Reviewer-facing demo UI.

Surfaces what a normal chat interface hides — the retrieval path behind each
answer, its latency, and whether a language model was involved — so the design
can be evaluated without reading the code.

The production interface is the embedded widget in static/index.html, served by
app.py. This module imports the handlers directly rather than calling them over
HTTP, so it runs as a single process with no separate backend.
"""
import logging
import os
import time

import streamlit as st

log = logging.getLogger(__name__)

st.set_page_config(page_title="NIT Delhi Campus Bot",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# Streamlit Cloud puts secrets in st.secrets; the rest of the codebase reads
# os.environ. Bridge them before anything imports db/config.
#
# `continue`, not `break`: with no secrets.toml at all, st.secrets raises on
# every lookup, and breaking out on the first one would silently skip the
# remaining keys even when some were readable.
for _k in ("MONGO_URI", "DB_NAME", "GROQ_API_KEY", "SERVER_PEPPER",
           "EMBED_MODEL", "ENABLE_DENSE", "LANGSMITH_TRACING",
           "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:      # no secrets configured — local runs use .env instead
        continue

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Fail with instructions rather than a KeyError from deep inside db.py, whose
# message Streamlit Cloud redacts — leaving a stack trace and no way to act on it.
if not os.getenv("MONGO_URI"):
    st.error("**MONGO_URI is not set**, so the app cannot reach the database.")
    st.markdown(
        "On **Streamlit Cloud**: *Manage app* (lower right) → **Settings** → "
        "**Secrets**, paste the block below, then **Save**. The app reboots "
        "automatically.\n\n"
        "Running **locally**: copy `.env.example` to `.env` and fill it in.")
    st.code('MONGO_URI = "mongodb+srv://USER:PASSWORD@cluster.mongodb.net/'
            '?appName=Cluster0"\n'
            'DB_NAME = "nitd_campus"\n'
            'SERVER_PEPPER = "a-long-random-string"\n'
            'GROQ_API_KEY = ""\n'
            'LANGSMITH_TRACING = "false"', language="toml")
    st.caption("Values go in TOML format — `KEY = \"value\"`, quoted, one per "
               "line. No `export`, no shell syntax.")
    st.info("Also required: MongoDB Atlas → **Network Access** → allow "
            "`0.0.0.0/0`. Streamlit Cloud connects from rotating IPs, so "
            "without this the app hangs on connect even with correct secrets.")
    st.stop()


# --------------------------------------------------------------- resources
# Imported at module scope rather than returned from a cached function, so a
# hot-reload can't leave the cache holding references to superseded modules.
import db as D            # noqa: E402
import embeddings         # noqa: E402
import rag_core           # noqa: E402
import rag_handlers as H  # noqa: E402


@st.cache_resource(show_spinner="Connecting to MongoDB and loading the corpus…")
def boot():
    """Load the corpus and warm the embedding model, once per container.

    Warming here moves the model download out of the first question. A failure
    is non-fatal: prose retrieval falls back to BM25.
    """
    rag_core.load()
    if embeddings.dense_enabled():
        try:
            embeddings.encode("warmup")
        except Exception as exc:
            log.warning("Embedding warm-up failed, using BM25 only: %s", exc)
    return True


def _fail(title: str, body: str, extra=None):
    st.error(title)
    st.markdown(body)
    if extra:
        st.code(extra, language="text")
    st.stop()


# Ping before loading the corpus. Otherwise the first failure surfaces from
# inside a cached function as a redacted pymongo traceback, which says nothing
# about which of the three usual causes it is.
try:
    D._client.admin.command("ping")
except Exception as _e:
    _n = type(_e).__name__
    if "OperationFailure" in _n or "Authentication" in _n:
        _fail(
            "**MongoDB rejected the username or password.**",
            "The app reached your cluster, so the network path and IP allowlist "
            "are fine — only the credentials are wrong.\n\n"
            "**1. Special characters must be percent-encoded.** This is the "
            "usual cause. If the password contains any of `@ : / ? # [ ] %`, it "
            "must be escaped in the URI, or the parser reads it as a "
            "delimiter:\n\n"
            "| char | write as | char | write as |\n"
            "|---|---|---|---|\n"
            "| `@` | `%40` | `/` | `%2F` |\n"
            "| `:` | `%3A` | `#` | `%23` |\n"
            "| `%` | `%25` | `?` | `%3F` |\n\n"
            "**2. Check the user exists** — Atlas → *Database Access*. It must "
            "be a **database user**, not your Atlas login, and the name must "
            "match exactly.\n\n"
            "**3. Check its role** — that user needs read/write on this "
            "database.\n\n"
            "If you rotated the password, confirm the secret holds the new one.")
    elif "ServerSelection" in _n or "Timeout" in _n:
        _fail(
            "**Could not reach the MongoDB cluster.**",
            "Atlas → **Network Access** → *Add IP Address* → **Allow access "
            "from anywhere** (`0.0.0.0/0`).\n\n"
            "Streamlit Cloud connects from rotating IPs, so an allowlist holding "
            "only your laptop's address fails here while still working locally.")
    else:
        _fail(
            "**Could not connect to MongoDB.**",
            "The connection string may be malformed — it should start with "
            "`mongodb+srv://` and contain no spaces or line breaks.",
            f"{_n}: {str(_e)[:400]}")


boot()

CATEGORIES = {
    "any":       "Auto-detect",
    "fee":       "Fees",
    "lab":       "Labs",
    "faculty":   "Faculty",
    "syllabus":  "Syllabus",
    "admission": "Admission",
    "about":     "About NIT Delhi",
    "shops":     "Open now",
}

# Chosen to exercise a different retrieval path each — and to include the
# refusals, which are the part worth grading.
SAMPLES = [
    ("B.Tech 3rd sem JOSAA fee",     "fee",       "Metadata filter — 5 slots, zero vectors"),
    ("Which lab has MATLAB?",        "lab",       "Equipment keyword over lab records"),
    ("HOD of Civil",                 "faculty",   "Role lookup, no name in the query"),
    ("dr haleem",                    "faculty",   "Typo → Dr. Halim, title-stripped fuzzy"),
    ("2nd semester subjects in CSE", "syllabus",  "Scoped prefilter, then hybrid"),
    ("Documents for MCA reporting",  "admission", "Whole document — never chunked"),
    ("fee for 9th semester",         "fee",       "REFUSAL — semester not published"),
    ("total fee for all 8 semesters", "fee",      "REFUSAL — no arithmetic on money"),
]

# Offered in the sidebar. Sourced from the `generation.models` config row so the
# picker can't drift from the fallback chain the backend actually uses.
def _groq_models():
    from config import CFG
    return list(CFG.get("generation.models") or [])


GROQ_MODELS = _groq_models()

BADGE_COLOR = {
    "METADATA-FILTER": "#0f766e", "EXACT-LOOKUP":   "#1d4ed8",
    "FUZZY-LEXICAL":   "#7c3aed", "WHOLE-DOCUMENT": "#0369a1",
    "GUARD":           "#b45309", "HYBRID":         "#be185d",
    "LIVE":            "#047857",
}


def badge_hue(method: str) -> str:
    m = (method or "").upper()
    for k, v in BADGE_COLOR.items():
        if k in m:
            return v
    return "#475569"


def md(text: str) -> str:
    """Handlers emit plain text, where a single newline separates a fee row or a
    course. Markdown collapses those into one paragraph, so every multi-row
    answer arrives as a run-on line. Promote them to hard breaks; leave blank
    lines alone so real paragraphs still work."""
    import re
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", text or "")


def source_line(src) -> str | None:
    """Sources are dicts of {label, url}; printing one raw puts a Python repr in
    front of a reviewer."""
    if not src:
        return None
    if isinstance(src, dict):
        label = src.get("label") or "Source"
        return f"Source: [{label}]({src['url']})" if src.get("url") else f"Source: {label}"
    return f"Source: {src}"


def render_badge(method: str, ms: int, cached: bool):
    hue = badge_hue(method)
    llm = "LLM" in (method or "").upper()
    chips = (
        f'<span style="background:{hue};color:#fff;padding:2px 9px;border-radius:11px;'
        f'font-size:11px;font-weight:600;letter-spacing:.3px">{method or "—"}</span>'
        f'<span style="color:#64748b;font-size:11px;margin-left:9px">{ms} ms</span>'
    )
    if not llm:
        chips += ('<span style="color:#047857;font-size:11px;margin-left:9px;'
                  'font-weight:600">NO LLM</span>')
    if cached:
        chips += '<span style="color:#7c3aed;font-size:11px;margin-left:9px">cached</span>'
    st.markdown(f'<div style="margin:-6px 0 4px">{chips}</div>', unsafe_allow_html=True)


def shops_answer() -> dict:
    """`shops` has no handler — it's live state, read straight from Mongo and
    never cached, so it can't go stale behind a cache TTL."""
    rows = D.shop_list()
    if not rows:
        return {"answer": "No outlets are configured yet.", "method": "LIVE-STATE"}
    lines = []
    for s in rows:
        dot = {"open": "🟢", "closed": "🔴"}.get(
            "open" if s.get("is_open") else ("closed" if s.get("is_open") is False else ""), "⚪")
        age = f"  ·  updated {s['age']}" if s.get("age") else ""
        lines.append(f"{dot}  **{s['name']}** — {s['headline']}{age}")
    return {"answer": "\n\n".join(lines),
            "method": "LIVE-STATE (never cached)", "source": None}


def ask(q: str, category: str) -> dict:
    t0 = time.perf_counter()
    if category == "shops":
        out = shops_answer()
    else:
        cat = category if category in H.HANDLERS else "any"
        try:
            out = H.HANDLERS[cat](q, st.session_state.slots)
        except Exception as e:
            # Full traceback to the server log — the one-line message a user sees
            # is rarely enough to find the cause.
            import traceback
            traceback.print_exc()
            out = {"answer": "Something went wrong answering that. Try rephrasing.",
                   "method": f"error: {str(e)[:90]}"}
    out["ms"] = int((time.perf_counter() - t0) * 1000)
    return out


# ------------------------------------------------------------------ state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("slots", {})
st.session_state.setdefault("pending", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### NIT Delhi Campus Bot")
    st.caption("Retrieval-first student assistant. Most answers never reach a language model.")

    stats = rag_core.stats()
    a, b = st.columns(2)
    a.metric("Chunks", stats["chunks"])
    b.metric("Embedded", stats["embedded"])
    a.metric("Faculty", stats["entity"])
    b.metric("Labs", stats["labs"])
    a.metric("Courses", stats["courses"])
    b.metric("Fee rows", stats["fee_rows"])

    st.divider()
    st.markdown("#### Why so little vector search?")
    st.markdown(
        "Faculty records are near-identical sentences with one name swapped. "
        "Measured similarity between **different people**: mean 0.787, "
        "top pairs **0.975**.\n\n"
        "An embedding model maps those to almost the same vector, so semantic "
        "search returns the wrong professor with high confidence. The only "
        "distinguishing signal is a proper noun — exactly what embeddings "
        "compress away.\n\n"
        "So retrieval is classified by **behaviour, not topic**:")
    st.markdown(
        "| Data | Technique | Vectors |\n|---|---|:--:|\n"
        "| Faculty | lexical + fuzzy | ✗ |\n"
        "| Labs | room → alias → keyword | ✗ |\n"
        "| Fees | metadata filter | ✗ |\n"
        "| Admission | whole document | ✗ |\n"
        "| Shops | live read | ✗ |\n"
        "| Syllabus | BM25 + dense, RRF | ✓ |")

    st.divider()
    st.markdown("#### Bring your own key")
    # Whether a key is configured on the deployment changes what "leave blank"
    # means, so say the true thing rather than a generic line that is wrong in
    # one of the two cases.
    if H.GROQ_KEY:
        st.caption("Optional — leave blank to use this deployment's key. "
                   "Only the Syllabus category calls a model at all.")
    else:
        st.caption("This demo ships **no** key, so answers come straight from "
                   "the source data. Add your own to have Syllabus answers "
                   "phrased conversationally — every other category is "
                   "template-rendered and needs no model.")

    user_key = st.text_input(
        "Groq API key", type="password", placeholder="gsk_…",
        help="Held in your browser session only — never written to the "
             "database, never logged, and gone when you close the tab.")
    # Verify once per distinct key, not on every rerun — Streamlit reruns the
    # whole script on each keystroke and widget change, and a network round trip
    # per rerun would make the sidebar feel broken.
    if user_key and st.session_state.get("checked_key") != user_key:
        with st.spinner("Checking key…"):
            st.session_state.key_result = H.verify_key(user_key)
        st.session_state.checked_key = user_key
    elif not user_key:
        st.session_state.pop("checked_key", None)
        st.session_state.pop("key_result", None)

    ok, msg, avail = st.session_state.get("key_result", (False, "", []))

    # Offer only models the key can actually call, so the picker can't select
    # one that fails on the first question.
    models = [m for m in GROQ_MODELS if m in avail] if (ok and avail) else GROQ_MODELS
    user_model = st.selectbox(
        "Model", models,
        help="Tried first. The others remain as fallbacks, so one decommissioned "
             "model can't take the category down.")

    if user_key:
        if ok:
            st.success(f"Connected successfully. {msg.split('—', 1)[-1].strip()}")
            st.caption("Requests are billed to your Groq account. The key stays "
                       "in this browser session — never stored or logged.")
        else:
            st.error(msg)
    st.markdown("[Get a free Groq key](https://console.groq.com/keys)")

    # A key that failed verification must not be used: falling through to it
    # would trade a clear error here for a confusing one mid-answer.
    H.set_llm_override(api_key=user_key if ok else None, model=user_model)

    st.divider()
    llm = H.llm_available()
    st.markdown(f"**LLM:** {'connected' if llm else 'not configured'}  \n"
                f"**Dense retrieval:** {'on' if embeddings.dense_enabled() else 'off (BM25 only)'}")
    if not llm:
        st.caption("Every category except Syllabus is unaffected — they render "
                   "from templates and need no model.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages, st.session_state.slots = [], {}
        st.rerun()
    st.caption("[Source on GitHub](https://github.com/malothritesh07/nitd-campus-bot)")

# ------------------------------------------------------------------- main
st.title("NIT Delhi Campus Bot")
st.caption("Fees · labs · faculty · syllabus · admissions · live campus status — "
           "with the retrieval path shown on every answer.")

if not st.session_state.messages:
    st.markdown("##### Try one of these")
    st.caption("Each runs a different retrieval path. The last two are refusals — "
               "the bot declining rather than guessing.")
    cols = st.columns(4)
    for i, (q, cat, why) in enumerate(SAMPLES):
        with cols[i % 4]:
            if st.button(q, key=f"s{i}", use_container_width=True, help=why):
                st.session_state.pending = (q, cat)
                st.rerun()
    st.divider()

choice = st.selectbox("Category", list(CATEGORIES),
                      format_func=lambda k: CATEGORIES[k],
                      help="Auto-detect infers the category from your wording. "
                           "Picking one scopes retrieval, exactly as the chips do in the real widget.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_badge(m.get("method"), m.get("ms", 0), m.get("cached", False))
        st.markdown(md(m["content"]))
        if m.get("source"):
            st.caption(source_line(m["source"]))
        if m.get("evidence"):
            with st.expander("Evidence used"):
                st.code(m["evidence"][:2500], language=None)

typed = st.chat_input("Ask about fees, labs, faculty, syllabus, admissions…")

pending = st.session_state.pending
st.session_state.pending = None
q, cat = pending if pending else ((typed, choice) if typed else (None, None))

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            out = ask(q, cat)
        render_badge(out.get("method"), out["ms"], out.get("cached", False))
        st.markdown(md(out.get("answer", "")))
        if out.get("source"):
            st.caption(source_line(out["source"]))
        ev = out.get("context") or out.get("evidence")
        if ev:
            with st.expander("Evidence used"):
                st.code(str(ev)[:2500], language=None)

    st.session_state.messages.append({
        "role": "assistant", "content": out.get("answer", ""),
        "method": out.get("method"), "ms": out["ms"],
        "source": out.get("source"), "cached": out.get("cached", False),
        "evidence": str(ev)[:2500] if ev else None})
