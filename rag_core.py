"""Retrieval core — the Colab notebook's logic, reading from MongoDB.

Loaded once at startup: chunks and vectors come out of Atlas into memory for
ranking; only the query embedding is computed on this machine.

Design (unchanged from the notebook):
  entity  faculty/staff/HOD  -> lexical + fuzzy ONLY. Different people share
                                ~0.97 identical sentences, so vectors can't
                                separate them.
  lab     labs/rooms         -> room number -> alias fuzzy -> equipment keyword
  fee     fees               -> metadata filtering, hybrid only as fallback
  prose   syllabus/about     -> BM25 + dense fused with RRF, LLM to phrase it
  admission                  -> whole document, never chunked
"""
import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz, utils

import db as D
from config import CFG
from embeddings import encode
from textutil import format_inr, tokenize
from tracing import trace

INR = format_inr
tok = tokenize


CHUNKS: list = []
FAC: list = []
FAC_NAMES: list = []
LABS: list = []
COURSES: list = []
COURSE_KEYS: list = []
ADMISSION: list = []
LAB_ALIAS: list = []
LAB_ALIAS_STR: list = []
LAB_DF = Counter()
DF_MAX = 1
KNOWN_CODES: set = set()
SYL_PROGRAMS: list = []
bm25 = None
dense_vecs = None
DENSE_IDX: list = []
POS_IN_DENSE: dict = {}


DEPT_ALIASES = CFG.get("lexicon.dept_aliases")
QWORDS       = set(CFG.get("lexicon.question_words"))
STOP_LAB     = set(CFG.get("lexicon.lab_stopwords"))
LIST_PAT = r"\b(list|all|names of|show me|every|faculty|professors?|teachers?|staff|members)\b"
PRONOUN  = r"\b(it|its|that|this|there|same|above)\b"


def detect_dept(q):
    ql = q.lower()
    for k, v in DEPT_ALIASES.items():
        if re.search(rf"\b{k}\b", ql):
            return v
    return None


def norm_roman(s):
    s = s.lower()
    for rom, num in (("iii", "3"), ("ii", "2"), ("i", "1")):
        s = re.sub(rf"\blab[-\s]?{rom}\b", f"lab {num}", s)
    return s


def _lab_blob(c):
    return (" ".join(str(c["meta"].get(f) or "") for f in ("hardware", "software", "name"))
            + " " + c["text"]).lower()


def lab_aliases(c):
    n = (c["meta"].get("name") or "").strip()
    out = {n}
    for inner in re.findall(r"\(([^)]+)\)", n):
        out.add(inner.strip())
    base = re.sub(r"\([^)]*\)", "", n)
    out.add(base.strip())
    words = [w for w in re.findall(r"[A-Za-z]+", base) if len(w) > 2]
    if len(words) >= 2:
        out.add("".join(w[0] for w in words).upper())
    return [x for x in out if x]


def load(force=False):
    global CHUNKS, FAC, FAC_NAMES, LABS, COURSES, COURSE_KEYS, ADMISSION
    global LAB_ALIAS, LAB_ALIAS_STR, LAB_DF, DF_MAX, KNOWN_CODES, SYL_PROGRAMS
    global bm25, dense_vecs, DENSE_IDX, POS_IN_DENSE
    if CHUNKS and not force:
        return

    CHUNKS = list(D.db.chunks.find({"status": {"$ne": "archived"}}))
    ADMISSION = list(D.db.admission_docs.find({"status": {"$ne": "archived"}}))

    FAC       = [c for c in CHUNKS if c["rag_class"] == "entity"]
    FAC_NAMES = [c["meta"].get("name") or "" for c in FAC]
    LABS      = [c for c in CHUNKS if c["rag_class"] == "lab"]
    COURSES   = [c for c in CHUNKS if c["domain"] == "syllabus"
                 and c["meta"].get("granularity") == "course"]
    COURSE_KEYS = [f"{c['meta'].get('course_title')} {c['meta'].get('course_code')}"
                   for c in COURSES]
    KNOWN_CODES = {(c["meta"].get("course_code") or "").upper().replace(" ", "")
                   for c in COURSES}
    SYL_PROGRAMS = sorted({c["meta"]["program"] for c in COURSES if c["meta"].get("program")})

    LAB_DF = Counter()
    for c in LABS:
        LAB_DF.update(set(tok(_lab_blob(c))))
    DF_MAX = max(1, int(CFG.get("retrieval.lab_df_ratio") * max(1, len(LABS))))
    LAB_ALIAS     = [(c, a) for c in LABS for a in lab_aliases(c)]
    LAB_ALIAS_STR = [norm_roman(a) for _, a in LAB_ALIAS]

    bm25 = BM25Okapi([tok(c["text"] + " " + str(c.get("meta"))) for c in CHUNKS])
    DENSE_IDX = [i for i, c in enumerate(CHUNKS) if c.get("embedding")]
    dense_vecs = (np.array([CHUNKS[i]["embedding"] for i in DENSE_IDX], dtype="float32")
                  if DENSE_IDX else np.zeros((0, 384), dtype="float32"))
    POS_IN_DENSE = {ci: r for r, ci in enumerate(DENSE_IDX)}


def syllabus_coverage():
    """Describe the coverage actually loaded, so the statement cannot go stale
    as semesters are added."""
    load()
    if not COURSES:
        return "No syllabus data is loaded yet."
    parts = []
    for prog in SYL_PROGRAMS:
        sems = sorted({c["meta"]["semester"] for c in COURSES
                       if c["meta"].get("program") == prog and c["meta"].get("semester")})
        if sems:
            parts.append(f"{prog} (semesters {', '.join(map(str, sems))})")
    return "I hold: " + "; ".join(parts) + "." if parts else "No syllabus data is loaded yet."


def stats():
    load()
    return {"chunks": len(CHUNKS), "entity": len(FAC), "labs": len(LABS),
            "courses": len(COURSES), "embedded": len(DENSE_IDX),
            "fee_rows": D.db.fee_rows.count_documents({"status": {"$ne": "archived"}}),
            "admission": len(ADMISSION), "programs": SYL_PROGRAMS}


def strip_title(s):
    """Drop Dr./Prof./Mr. so the fuzzy score reflects the actual name."""
    return re.sub(r"^(?:prof\.?\s*)?(?:dr|mr|ms|mrs)?\.?\s*", "",
                  (s or "").strip().lower()).strip()


def extract_name(q):
    m = re.search(r"\b((?:dr|prof|mr|ms|mrs)\.?\s*[A-Za-z]+(?:\s+[A-Za-z]+){0,2})", q, re.I)
    if m:
        return m.group(1).strip()
    return " ".join(t for t in re.findall(r"[A-Za-z.]+", q)
                    if t.lower().strip(".") not in QWORDS).strip()


def best_name_score(q):
    load()
    if not FAC_NAMES:
        return 0
    r = process.extract(q, FAC_NAMES, scorer=fuzz.WRatio,
                        processor=utils.default_process, limit=1)
    return r[0][1] if r else 0


def _name_overlap(query_name, candidate, min_ratio=None):
    """A surname in the query must actually resemble a word in the candidate's
    name. Without this, 'dr rajesh mehta' (nobody) matched three real people."""
    min_ratio = CFG.get("retrieval.entity_name_overlap") if min_ratio is None else min_ratio
    qt = [w for w in tok(query_name) if w not in {"dr", "prof", "mr", "ms", "mrs"} and len(w) > 2]
    ct = [w for w in tok(candidate)  if w not in {"dr", "prof", "mr", "ms", "mrs"} and len(w) > 2]
    if not qt or not ct:
        return False
    return any(fuzz.ratio(a, b) >= min_ratio for a in qt for b in ct)


@trace(name="retrieve_entity", run_type="retriever")
def retrieve_entity(query, k=None, min_score=None):
    k = k or CFG.get("retrieval.top_k_entity")
    min_score = CFG.get("retrieval.entity_min_score") if min_score is None else min_score
    load()
    dept, ql = detect_dept(query), query.lower()
    pool = [(c, n) for c, n in zip(FAC, FAC_NAMES)]
    if dept:
        f = [(c, n) for c, n in pool if (c["meta"].get("department") or "") == dept]
        if f:
            pool = f
    if re.search(r"\bhod\b|head of (the )?dep", ql):
        hits = [c for c, _ in pool if re.search(
            r"head of department|hod",
            str(c["meta"].get("role") or "") + str(c["meta"].get("designation") or ""), re.I)]
        return [(h, 100) for h in hits[:k]]
    name = strip_title(extract_name(query))
    if not name:
        return []


    raw = process.extract(name, [strip_title(n) for _, n in pool], scorer=fuzz.WRatio,
                          processor=utils.default_process, limit=k)
    return [(pool[i][0], s) for _, s, i in raw
            if s >= min_score and _name_overlap(name, strip_title(pool[i][1]))]


def render_entity(q):
    load()
    dept = detect_dept(q)
    plural = bool(re.search(LIST_PAT, q.lower())) and not re.search(r"\bwho is\b", q.lower())
    if dept and plural:
        want_staff = bool(re.search(r"staff|non[- ]teach|technician", q.lower()))
        sel = [c for c in FAC if c["meta"].get("department") == dept
               and (("Staff" in str(c["meta"].get("staff_type"))) == want_staff)]
        if sel:
            return (f"{'Staff' if want_staff else 'Faculty'} of {dept} ({len(sel)}):\n" +
                    "\n".join(f"{i}. {c['meta']['name']} — {c['meta'].get('designation')}"
                              for i, c in enumerate(sel, 1)))
    hits = retrieve_entity(q)
    if not hits:
        return None


    top = hits[0][1]
    close = [h for h in hits if top - h[1] <= CFG.get("retrieval.entity_ambiguity_margin")]
    if len(close) > 1:
        return ("More than one match:\n" + "\n".join(
            f"- {c['meta']['name']} — {c['meta'].get('designation')} ({c['meta'].get('department')})"
            for c, _ in close) + "\nWhich one did you mean?")

    m = hits[0][0]["meta"]
    return f"{m['name']} — {m.get('designation') or m.get('role')}, {m.get('department')}."


def _tok_overlap(query, alias, min_ratio=None):
    min_ratio = CFG.get("retrieval.lab_token_overlap") if min_ratio is None else min_ratio
    """Fuzzy score alone isn't enough — a query token must resemble a token in the
    lab's name. Without this, 'airlib' confidently matched 'JEEVAN Lab'."""
    qt = [w for w in tok(query) if w not in STOP_LAB and w not in QWORDS and len(w) > 2]
    at = [w for w in tok(alias) if len(w) > 2]
    if not qt or not at:
        return False
    return any(fuzz.ratio(a, b) >= min_ratio for a in qt for b in at)


@trace(name="retrieve_lab", run_type="retriever")
def retrieve_lab(query, k=None):
    k = k or CFG.get("retrieval.top_k_lab")
    load()
    ql, dept = query.lower(), detect_dept(query)
    pool = LABS
    if dept:
        f = [c for c in pool if c["meta"].get("department") == dept]
        if f:
            pool = f

    m    = re.search(r"(?:lab|room)\s*(?:no\.?|number|#)\s*[-:.]?\s*(\d+)", ql)
    bare = re.search(r"\b(\d{2,3})\b", ql)
    num  = int(m.group(1)) if m else (int(bare.group(1)) if bare and int(bare.group(1)) >= 10 else None)
    if num is not None:
        hits = [c for c in pool if c["meta"].get("room_number") == num]
        if hits:
            return [(h, 100) for h in hits[:k]]

    equip = bool(re.search(r"\bwhich lab|\bhas\b|\bwith\b|contains|equipped", ql))
    if not equip:
        raw = process.extract(norm_roman(query), LAB_ALIAS_STR, scorer=fuzz.WRatio,
                              processor=utils.default_process, limit=10)
        seen, good = set(), []
        for alias, sc, i in raw:
            c = LAB_ALIAS[i][0]
            if id(c) in seen or (dept and c not in pool):
                continue
            if sc >= CFG.get("retrieval.lab_alias_strong") or (
                    sc >= CFG.get("retrieval.lab_alias_weak") and _tok_overlap(query, alias)):
                seen.add(id(c))
                good.append((c, sc))
        if good:
            return good[:k]

    content = [w for w in tok(query)
               if w not in STOP_LAB and w not in QWORDS and len(w) > 3]


    if any(LAB_DF[w] == 0 for w in content):
        return []
    kws = [w for w in content if 0 < LAB_DF[w] <= DF_MAX]
    if kws:


        scored = []
        for c in pool:
            name = (c["meta"].get("name") or "").lower()
            blob = _lab_blob(c)
            in_name = sum(1 for w in kws if w in name)
            in_blob = sum(1 for w in kws if w in blob)
            if not in_blob:
                continue
            scored.append((c, 60 + CFG.get("retrieval.lab_name_weight") * in_name
                              + CFG.get("retrieval.lab_blob_weight") * in_blob))
        scored.sort(key=lambda x: -x[1])
        if scored:
            return scored[:k]
    return []


def lab_suggestions(query, n=3):
    """Match on the distinctive words only — the full sentence dilutes the score
    so badly that 'airlib' failed to suggest AIRIL."""
    load()
    core = " ".join(w for w in tok(query)
                    if w not in STOP_LAB and w not in QWORDS and len(w) > 2) or query
    raw = process.extract(norm_roman(core), LAB_ALIAS_STR, scorer=fuzz.WRatio,
                          processor=utils.default_process, limit=12)
    seen, out = set(), []
    for _, sc, i in raw:
        nm = LAB_ALIAS[i][0]["meta"].get("name")
        if nm and nm not in seen and sc >= 55:
            seen.add(nm)
            out.append(nm)
    return out[:n]


def render_lab(q):
    hits = retrieve_lab(q)
    if not hits:
        return None
    m, ql = hits[0][0]["meta"], q.lower()
    if re.search(r"capacity|seats?|how many", ql):
        return f"{m['name']} — capacity {m.get('capacity')}." if m.get("capacity") else None
    if re.search(r"coordinator|incharge|in charge|who runs", ql):
        return f"{m['name']} — coordinator: {m.get('coordinators')}."
    if re.search(r"hardware|system|computer|equipment|machine", ql):
        return f"{m['name']} — hardware: {m.get('hardware')}"
    if re.search(r"software|matlab|tool", ql):
        return f"{m['name']} — software: {m.get('software') or 'not listed'}"
    if re.search(r"where|location|room|located|floor", ql):
        return (f"{m['name']} is at {m.get('raw')}." if m.get("raw")
                else f"{m['name']} — location not listed. Contact the department.")
    return (f"{m['name']}\nLocation: {m.get('raw')}\nCoordinator: {m.get('coordinators')}\n"
            f"Capacity: {m.get('capacity')}")


def courses_by_code(code_norm):
    """Exact code wins over fuzzy title matching. A student asking for CSBB 103
    was being handed ADBB 103 — a different subject entirely."""
    load()
    return [c for c in COURSES
            if (c["meta"].get("course_code") or "").upper().replace(" ", "") == code_norm]


def match_course(q, min_score=None):
    min_score = CFG.get("retrieval.course_match_min") if min_score is None else min_score
    load()
    if not COURSES:
        return None
    m = re.search(r"\b([A-Za-z]{3,4})\s?(\d{3})\b", q)
    if m:
        exact = courses_by_code((m.group(1) + m.group(2)).upper())
        if exact:
            return exact[0]
    r = process.extract(q, COURSE_KEYS, scorer=fuzz.WRatio,
                        processor=utils.default_process, limit=1)
    return COURSES[r[0][2]] if r and r[0][1] >= min_score else None


def expand_parent(chunk):
    if chunk["domain"] != "syllabus" or chunk["meta"].get("granularity") != "module":
        return chunk
    for c in CHUNKS:
        if (c["domain"] == "syllabus"
                and c["meta"].get("doc_id") == chunk["meta"]["doc_id"]
                and c["meta"].get("granularity") == "course"):
            return c
    return chunk


@trace(name="retrieve_scoped", run_type="retriever")
def retrieve_scoped(query, keep_fn, k=None, pool=None):
    k = k or CFG.get("retrieval.top_k_scoped")
    pool = pool or CFG.get("retrieval.candidate_pool")
    """Hybrid BM25 + dense restricted to one category — nothing outside can surface."""
    load()
    allowed = [i for i, c in enumerate(CHUNKS) if keep_fn(c)]
    if not allowed:
        return []
    aset = set(allowed)
    bs = bm25.get_scores(tok(query))
    b_rank = [int(i) for i in np.argsort(bs)[::-1] if int(i) in aset][:pool]
    drows = [(POS_IN_DENSE[i], i) for i in allowed if i in POS_IN_DENSE]
    d_rank = []
    qv = encode(query) if drows else None
    if qv is not None:
        sims = dense_vecs[[r for r, _ in drows]] @ qv
        d_rank = [drows[o][1] for o in np.argsort(sims)[::-1][:pool]]
    K, fused = CFG.get("retrieval.rrf_k"), {}
    for r, i in enumerate(b_rank):
        fused[i] = fused.get(i, 0) + 1 / (K + r + 1)
    for r, i in enumerate(d_rank):
        fused[i] = fused.get(i, 0) + 1 / (K + r + 1)
    return [(CHUNKS[i], s) for i, s in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def detect_syl_program(q):
    load()
    ql = q.lower()
    if re.search(r"\bai\b|artificial|data science|ai&ds|aids", ql):
        return next((p for p in SYL_PROGRAMS if "Artificial" in p), None)
    if re.search(r"mech|aero|m&ae|mae", ql):
        return next((p for p in SYL_PROGRAMS if "Mechanical" in p), None)
    if re.search(r"\bcse\b|computer", ql):
        return next((p for p in SYL_PROGRAMS if p.startswith("Computer")), None)
    return None


def render_admission(q):
    load()
    ql = q.lower()
    if   re.search(r"\bdasa\b", ql):
        want = "admission-docs-btech-dasa"
    elif re.search(r"\bmca\b|nimcet", ql):
        want = "admission-docs-mca-nimcet"
    elif re.search(r"\bjosaa\b|\bcsab\b|b\.?tech", ql):
        want = "admission-docs-btech-josaa-regular-status"
    else:
        want = "admission-overview"
    doc = next((d for d in ADMISSION if d["doc_id"] == want), ADMISSION[0] if ADMISSION else None)
    if not doc:
        return "Admission data isn't loaded."
    if doc.get("status") == "unavailable":
        return (f"{doc['quick_answer']}\n\nOfficial page: "
                "https://nitdelhi.ac.in/academics/services/admissions")
    parts = [doc.get("title", "")]
    if doc.get("reporting_schedule"):
        r = doc["reporting_schedule"]
        parts.append(f"When: {r.get('dates')}, {r.get('time')}\nWhere: {r.get('venue')}")
    items = doc.get("documents_required") or [i["item"] for i in doc.get("documents_checklist", [])]
    if items:
        parts.append(f"Documents ({len(items)}):\n" +
                     "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)))
    return "\n\n".join(p for p in parts if p), doc.get("source_url")
