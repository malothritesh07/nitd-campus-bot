"""Fee retrieval: metadata filtering first, hybrid (lexical + vector) as fallback.

No LLM anywhere in this path. Numbers are read straight from MongoDB and
rendered by template, so a fee figure can never be invented or miscalculated.

Everything is read from Atlas at query time; only the query embedding is
computed on this machine.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

import db as D
from config import CFG
from embeddings import encode
from textutil import format_inr, tokenize
from tracing import trace

_cache = {"chunks": [], "bm25": None, "vecs": None}

INR = format_inr
_tok = tokenize


def load_cache(force=False):
    """Pull chunks + vectors out of Mongo into memory for ranking."""
    if _cache["bm25"] is not None and not force:
        return
    # fee chunks live in the shared `chunks` collection under domain="fee"
    chunks = list(D.db.chunks.find({"domain": "fee", "embedding": {"$exists": True},
                                   "status": {"$ne": "archived"}}))
    for c in chunks:                       # flatten meta so prefilter/render can read it
        c.update({k: v for k, v in (c.get("meta") or {}).items() if k not in c})
    _cache["chunks"] = chunks
    _cache["bm25"]   = BM25Okapi([_tok(c["text"]) for c in chunks]) if chunks else None
    _cache["vecs"]   = (np.array([c["embedding"] for c in chunks], dtype="float32")
                        if chunks else None)


# ------------------------------------------------------------------ slots
# Patterns come from the `config` collection — a new admission route or fee
# component is a database edit, not a code change.
PROGRAM_PAT = [(p, v) for p, v in CFG.get("lexicon.program_patterns")]
ADM_PAT     = [(p, v) for p, v in CFG.get("lexicon.admission_patterns")]
COMPONENTS  = CFG.get("lexicon.fee_components")


def extract_slots(q: str) -> dict:
    ql = q.lower()
    s = {"program": None, "semester": None, "admission_type": None,
         "residence": None, "category_bucket": None}
    for p, v in PROGRAM_PAT:
        if re.search(p, ql):
            s["program"] = v
            break
    for p, v in ADM_PAT:
        if re.search(p, ql):
            s["admission_type"] = v
            break

    # income first, so its digits can't be mistaken for a semester
    if   re.search(r"below 1|less than 1|under 1|<\s*1", ql):
        s["category_bucket"] = "below_1_lakh"
    elif re.search(r"1.{0,6}5\s*lakh|between\s*1", ql):
        s["category_bucket"] = "1_to_5_lakh"
    elif re.search(r"above 5|more than 5|over 5|>\s*5", ql):
        s["category_bucket"] = "above_5_lakh"
    elif re.search(r"\bsc\b|\bst\b|\bpwd\b|\bph\b|reserved", ql):
        s["category_bucket"] = "sc_st_pwd"
    elif re.search(r"\bgen\b|general|\bobc\b|\bews\b|\bur\b", ql):
        s["category_bucket"] = "gen_obc_ews"

    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*sem", ql)
    if m:
        s["semester"] = int(m.group(1))
    else:
        # a bare number can be the answer to "which semester?" — but only after
        # stripping income phrases, or "below 1 lakh" would read as semester 1
        cleaned = re.sub(r"\b(?:below|less than|under|above|more than|over|between)\s*\d+"
                         r"(?:\s*(?:to|-)\s*\d+)?\s*(?:lakhs?)?", " ", ql)
        cleaned = re.sub(r"\b\d+\s*(?:to|-)\s*\d+\s*lakhs?", " ", cleaned)
        cleaned = re.sub(r"\b\d+\s*lakhs?", " ", cleaned)
        cleaned = re.sub(r"\brs\.?\s*[\d,]+", " ", cleaned)
        m2 = re.search(r"\b([1-8])(?:st|nd|rd|th)?\b", cleaned)
        if m2:
            s["semester"] = int(m2.group(1))

    # typo-tolerant: hoostel / hostell / hostl, day scholer / dayscholar
    if   re.search(r"non[- ]?a\.?c", ql):
        s["residence"] = "hosteller_non_ac"
    elif re.search(r"\ba\.?c\b", ql):
        s["residence"] = "hosteller_ac"
    elif re.search(r"ho+ste?l+|hostl", ql):
        s["residence"] = "hosteller_ac"
    elif re.search(r"day\s*scho?l?a?e?r", ql):
        s["residence"] = "day_scholar"
    return s


def detect_component(q: str):
    ql = q.lower()
    for pat, key in COMPONENTS.items():
        if re.search(pat, ql):
            return key
    return None


# ------------------------------------------------------- metadata filtering
def filter_rows(slots: dict) -> list:
    """The primary path: a plain indexed MongoDB query. No vectors involved."""
    q = {k: v for k, v in slots.items() if v is not None}
    q["status"] = {"$ne": "archived"}   # never quote a fee that was withdrawn
    return list(D.db.fee_rows.find(q, {"_id": 0}))


# ------------------------------------------------------------------ hybrid
@trace(name="fee_hybrid", run_type="retriever")
def hybrid_chunks(query: str, k=None, pool=None, prefilter: dict | None = None):
    """Lexical (BM25) + vector (cosine), fused with Reciprocal Rank Fusion.
    Used only when metadata filtering can't pin the answer down."""
    k = k or CFG.get("retrieval.top_k_fee")
    pool = pool or CFG.get("retrieval.candidate_pool")
    load_cache()
    chunks = _cache["chunks"]
    if not chunks:
        return []

    idx = list(range(len(chunks)))
    if prefilter:
        idx = [i for i in idx
               if all(chunks[i].get(f) == v for f, v in prefilter.items() if v is not None)] or idx

    b_scores = _cache["bm25"].get_scores(_tok(query))
    b_rank = sorted(idx, key=lambda i: -b_scores[i])[:pool]

    qv = encode(query)
    if qv is None:
        sims, d_rank = np.zeros(len(chunks), dtype="float32"), []
    else:
        sims = _cache["vecs"] @ qv
        d_rank = sorted(idx, key=lambda i: -sims[i])[:pool]

    K, fused = CFG.get("retrieval.rrf_k"), {}
    for r, i in enumerate(b_rank):
        fused[i] = fused.get(i, 0) + 1 / (K + r + 1)
    for r, i in enumerate(d_rank):
        fused[i] = fused.get(i, 0) + 1 / (K + r + 1)
    top = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return [(chunks[i], round(s, 5), float(sims[i])) for i, s in top]


# ---------------------------------------------------------------- rendering
def _label(r):
    return (f"{r['program']} sem {r['semester']} · {r['admission_type']} · "
            f"{r['income_category'].replace('_', ' ')} · {r['residence'].replace('_', ' ')}")


def render_rows(rows: list) -> dict:
    lines = []
    for r in rows[:6]:
        excl = "  (tuition billed separately)" if r["excludes_tuition"] else ""
        lines.append(f"{_label(r)}\n   → {INR(r['amount'])}{excl}")
    note = ""
    if any(r["excludes_tuition"] and r.get("tuition_note") for r in rows[:6]):
        note = "\n\n" + next(r["tuition_note"] for r in rows[:6]
                             if r["excludes_tuition"] and r.get("tuition_note"))
    return {"answer": "\n".join(lines) + note,
            "source": {"label": "Official fee notice (PDF)", "url": rows[0]["source_url"]},
            "method": "metadata-filter", "rows": len(rows)}


def component_answer(q: str, slots: dict):
    key = detect_component(q)
    if not key:
        return None
    rows = filter_rows({k: v for k, v in slots.items()
                        if k in ("program", "semester", "admission_type") and v})
    if not rows:
        return {"answer": "Tell me the programme, semester and admission type first "
                          "(for example: B.Tech, 3rd semester, JOSAA).",
                "source": None, "method": "metadata-filter"}
    doc = D.db.fee_docs.find_one({"_id": rows[0]["doc_id"], "status": {"$ne": "archived"}})
    if not doc:
        return None
    hdr = f"{doc.get('program')} sem {doc.get('semester')} · {doc.get('admission_type')}"

    if key == "__bank__":
        b = doc.get("bank_details", {})
        return {"answer": f"{hdr}\nBank: {b.get('bank_name')}\nA/c: {b.get('account_name')}\n"
                          f"A/c no: {b.get('account_number')}\nIFSC: {b.get('ifsc_code')}\n\n{b.get('note','')}",
                "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                "method": "metadata-filter"}
    if key == "__hostel__":
        h = doc.get("hostel_fee_breakdown") or doc.get("hostel_one_time_breakdown") or {}
        body = "\n".join(f"{k.replace('_',' ')}: {INR(v)}" for k, v in h.items()
                         if isinstance(v, (int, float)))
        return {"answer": f"{hdr}\n{body}" if body else f"{hdr}\nNot listed separately.",
                "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                "method": "metadata-filter"}
    if key == "__caution__":
        vals = {**{k: v for k, v in (doc.get("hostel_one_time_breakdown") or {}).items() if "caution" in k},
                **(doc.get("one_time_admission_fee_breakdown_D") or {})}
        body = "\n".join(f"{k.replace('_',' ')}: {INR(v)}" for k, v in vals.items()
                         if isinstance(v, (int, float)))
        return {"answer": f"{hdr}\n{body}" if body else f"{hdr}\nNo caution money listed.",
                "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                "method": "metadata-filter"}

    if key in ("tuition_fee", "admission_fee", "total_B_institute_fees"):
        if isinstance(doc.get(key), (int, float)):
            return {"answer": f"{hdr}\n{key.replace('_',' ')}: {INR(doc[key])}",
                    "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                    "method": "metadata-filter"}
        if key == "tuition_fee" and doc.get("tuition_fee_note"):
            return {"answer": f"{hdr}\nTuition: {doc['tuition_fee_note']}",
                    "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                    "method": "metadata-filter"}
        lines = [f"{cn.replace('_',' ')}: {INR(blob[key])}"
                 for cn, blob in (doc.get("categories") or {}).items()
                 if isinstance(blob.get(key), (int, float))]
        if lines:
            return {"answer": f"{hdr}\n" + "\n".join(lines),
                    "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                    "method": "metadata-filter"}

    for sec in ("institute_fee_breakdown", "institute_fee_breakdown_B",
                "hostel_fee_breakdown", "annual_fee_breakdown_C"):
        blob = doc.get(sec) or {}
        if isinstance(blob.get(key), (int, float)):
            return {"answer": f"{hdr}\n{key.replace('_',' ')}: {INR(blob[key])}",
                    "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
                    "method": "metadata-filter"}
    return {"answer": f"{hdr}\nThat component isn't listed separately in this fee notice.",
            "source": {"label": "Official fee notice (PDF)", "url": doc.get("source_url")},
            "method": "metadata-filter"}


def COVERAGE_MSG():
    """Derived from the data. The old fixed string kept claiming semesters 3/5/7
    even after new semesters were ingested."""
    out = []
    live = {"status": {"$ne": "archived"}}
    for prog in sorted(D.db.fee_rows.distinct("program", live)):
        sems = sorted(D.db.fee_rows.distinct("semester", {**live, "program": prog}))
        out.append(f"{prog} semesters {', '.join(map(str, sems))}")
    return "I hold: " + "; ".join(out) + "." if out else "No fee data is loaded."


TOTAL_RE = re.compile(
    r"\btotal\b.*\b(years?|degree|course|all sem|entire|whole)\b"
    r"|\b(all|whole|entire)\s+(4|four|8|eight)?\s*years?\b")
COMPARE_RE = re.compile(
    r"difference|how much more|how much less|compare|cheaper|costlier|vs\b")

SLOT_ORDER = ("program", "semester", "admission_type", "residence")
FEE_PDF = "Official fee notice (PDF)"


def _merge_slots(current: dict, carried: dict | None) -> dict:
    """Carry unstated slots from earlier turns.

    A query naming two or more of programme, semester or admission type is a
    fresh question, so the rest of the carried state is dropped.
    """
    if not carried:
        return dict(current)
    named = sum(1 for k in ("program", "semester", "admission_type") if current[k])
    if named >= 2:
        return dict(current)
    return {k: (current[k] or carried.get(k)) for k in current}


def _guard_unpublished_semester(slots: dict):
    """Refuse a semester with no published figures.

    Falling through to hybrid here once answered a 4th-semester question with
    7th-semester amounts.
    """
    semester = slots.get("semester")
    if semester is None:
        return None

    probe = {"semester": semester, "status": {"$ne": "archived"}}
    if slots.get("program"):
        probe["program"] = slots["program"]
    if D.db.fee_rows.count_documents(probe):
        return None

    scope = {"program": slots["program"]} if slots.get("program") else {}
    available = sorted(D.db.fee_rows.distinct("semester", scope))
    programme = slots.get("program") or "that programme"
    # Clear the semester, or every later turn inherits it and hits this guard.
    return {"answer": f"Semester {semester} isn't published for {programme}. "
                      f"Available: {', '.join(str(s) for s in available)}.",
            "source": None, "method": "guard",
            "slots": {**slots, "semester": None}}


def _guard_total_across_semesters(query: str, slots: dict):
    """Published per-semester amounts are the only reliable figures; summing
    them would present an invented total as official."""
    if not TOTAL_RE.search(query.lower()):
        return None
    return {"answer": "I don't add figures across semesters — only the published "
                      f"per-semester amounts are reliable. {COVERAGE_MSG()}",
            "source": None, "method": "guard", "slots": slots}


def _guard_comparison(query: str, slots: dict):
    """Show both sides rather than computing a difference."""
    if not COMPARE_RE.search(query.lower()):
        return None

    # Residence is dropped so both sides of the comparison appear.
    rows = filter_rows({k: v for k, v in slots.items() if v and k != "residence"})
    if not rows:
        return None

    seen, lines = set(), []
    for row in rows:
        key = (row["income_category"], row["residence"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"   {row['residence'].replace('_', ' ')}"
                     f" ({row['income_category'].replace('_', ' ')}): {INR(row['amount'])}")
    return {"answer": "I don't calculate differences — here are the published figures:\n"
                      + "\n".join(lines[:8]),
            "source": {"label": FEE_PDF, "url": rows[0]["source_url"]},
            "method": "guard", "slots": slots}


def _ask_for_missing_slots(row_count: int, slots: dict) -> dict:
    missing = [k.replace("_", " ") for k in SLOT_ORDER if not slots.get(k)]
    return {"answer": "Which one do you need? Please tell me the "
                      + ", ".join(missing) + ".",
            "source": None, "method": "metadata-filter",
            "rows": row_count, "slots": slots}


def _hybrid_fallback(query: str, slots: dict):
    """Last resort when metadata filtering matched nothing."""
    hits = hybrid_chunks(query, k=2,
                         prefilter={"program": slots.get("program"),
                                    "semester": slots.get("semester")})
    if not hits or hits[0][2] < CFG.get("retrieval.fee_hybrid_min_cosine"):
        return None
    chunk = hits[0][0]
    return {"answer": chunk.get("quick_answer") or chunk["text"][:400],
            "source": {"label": chunk.get("title") or FEE_PDF,
                       "url": chunk.get("source_url")},
            "method": "hybrid(bm25+vector)", "score": hits[0][1], "slots": slots}


@trace(name="fee_answer", run_type="chain")
def answer_fee(q: str, carried: dict | None = None) -> dict:
    """Metadata filter first, component lookup next, hybrid only as a fallback.

    No arithmetic and no model anywhere in this path: every figure is read from
    the database and rendered by template.
    """
    slots = _merge_slots(extract_slots(q), carried)

    for guard in (_guard_unpublished_semester(slots),
                  _guard_total_across_semesters(q, slots),
                  _guard_comparison(q, slots)):
        if guard:
            return guard

    component = component_answer(q, slots)
    if component:
        return {**component, "slots": slots}

    rows = filter_rows({k: v for k, v in slots.items() if v})
    if rows and len(rows) <= 6:
        return {**render_rows(rows), "slots": slots}
    if rows:
        return _ask_for_missing_slots(len(rows), slots)

    hybrid = _hybrid_fallback(q, slots)
    if hybrid:
        return hybrid

    return {"answer": f"I don't have a fee record for that. {COVERAGE_MSG()}",
            "source": None, "method": "no-match", "slots": slots}
