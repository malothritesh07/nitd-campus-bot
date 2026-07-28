"""Builds the desired corpus from source files. Pure — touches no database.

Split out of ingest_all.py so the sync engine can compare "what should exist"
against "what does exist" without re-running any writes.
"""
import hashlib, json, os, re

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
DATA = os.getenv("DATA_DIR", "M:\\")

P      = lambda name: os.path.join(DATA, name)
jload  = lambda n: json.load(open(P(n), encoding="utf-8"))
jlload = lambda n: [json.loads(l) for l in open(P(n), encoding="utf-8") if l.strip()]

DEPT_FILES = ["rag_dataset_Ashm.json", "rag_dataset_CIVIL.json", "rag_dataset_CSE.json",
              "rag_dataset_ECE.json", "rag_dataset_MAE.json"]
CLASS_MAP = {"faculty": "entity", "staff": "entity", "leadership": "entity",
             "laboratories": "lab", "about": "prose"}

RES_KEYS = {
 "day_scholar_total": ("day_scholar", False),
 "day_scholar_total_excl_tuition": ("day_scholar", True),
 "hosteller_mess_ac_total": ("hosteller_ac", False),
 "hosteller_ac_total": ("hosteller_ac", False),
 "hosteller_ac_mess_total_excl_tuition": ("hosteller_ac", True),
 "hosteller_ac_total_excl_tuition": ("hosteller_ac", True),
 "hosteller_mess_non_ac_total": ("hosteller_non_ac", False),
 "hosteller_non_ac_total": ("hosteller_non_ac", False),
 "hosteller_non_ac_mess_total_excl_tuition": ("hosteller_non_ac", True),
 "hosteller_non_ac_total_excl_tuition": ("hosteller_non_ac", True)}


def content_hash(obj) -> str:
    """Stable hash of the parts that matter. Ordering-insensitive, so a
    reordered JSON file does not look like a change."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def parse_location(loc):
    if not loc:
        return {"room_number": None, "floor": None, "building": None, "raw": loc}
    m  = re.search(r"(?:lab|room)\s*(?:no\.?|number)?\s*[-:.]?\s*(\d+)", loc, re.I)
    fm = re.search(r"(basement|ground\s*floor|(\d+)\s*(?:st|nd|rd|th)\s*floor)", loc, re.I)
    bm = re.search(r"(mini\s*campus|admin(?:istrative)?\s*(?:building|block)|start-?up\s*building)",
                   loc, re.I)
    return {"room_number": int(m.group(1)) if m else None,
            "floor": fm.group(0).strip() if fm else None,
            "building": bm.group(0).strip() if bm else None, "raw": loc}


def bucket(cat):
    c = (cat or "").lower()
    if "less_than_1" in c: return "below_1_lakh"
    if "between_1"   in c: return "1_to_5_lakh"
    if "above_5"     in c: return "above_5_lakh"
    if re.search(r"\bsc|st|ph|pwd", c) and "gen" not in c: return "sc_st_pwd"
    if re.search(r"gen|obc|ews|ur", c): return "gen_obc_ews"
    return "all_categories"


def _fee_chunk_text(doc, rows):
    parts = [doc.get("title", ""),
             f"{doc.get('program')} semester {doc.get('semester')} "
             f"{doc.get('admission_type')} batch {doc.get('batch') or ''}".strip()]
    for r in rows:
        parts.append(f"{r['income_category'].replace('_',' ')} "
                     f"{r['residence'].replace('_',' ')} Rs {r['amount']}")
    if doc.get("tuition_fee_note"):
        parts.append("tuition: " + doc["tuition_fee_note"])
    for sec in ("institute_fee_breakdown", "institute_fee_breakdown_B", "hostel_fee_breakdown"):
        for k, v in (doc.get(sec) or {}).items():
            if isinstance(v, (int, float)):
                parts.append(f"{k.replace('_',' ')} Rs {v}")
    return " | ".join(p for p in parts if p)


def build():
    """Return the corpus the sources describe, each item carrying a content hash
    and the source file it came from (needed to detect whole-file removals)."""
    chunks, fee_rows, link_only = [], [], []
    seen = {}

    def add_chunk(c, src_file):
        cid = c["chunk_id"]
        if cid in seen:                       # real collisions: HHPB 150 is two subjects
            seen[cid] += 1
            cid = f"{cid}#{seen[cid]}"
            c["duplicate_code"] = True
        else:
            seen[cid] = 1
        c["_id"] = cid
        c["source_file"] = src_file
        c["content_hash"] = content_hash({"t": c["text"], "m": c.get("meta")})
        chunks.append(c)

    # ---- syllabus ----
    for r in jlload("rag_chunks_SYLLABUS.jsonl"):
        md = r.get("metadata", {})
        add_chunk({"chunk_id": r["chunk_id"], "rag_class": "prose", "domain": "syllabus",
                   "text": r["text"], "source": None,
                   "meta": {"doc_id": r.get("doc_id"), "granularity": r.get("granularity"),
                            "module_label": r.get("module_label"),
                            "course_code": md.get("course_code"),
                            "course_title": md.get("course_title"),
                            "program": md.get("program"), "semester": md.get("semester"),
                            "academic_year": md.get("academic_year")}},
                  "rag_chunks_SYLLABUS.jsonl")

    # ---- departments ----
    for fn in DEPT_FILES:
        if not os.path.exists(P(fn)):
            continue
        for r in jload(fn):
            md, cat = r.get("metadata", {}), r.get("metadata", {}).get("category")
            if md.get("status") == "incomplete_needs_scraping":
                dept_key = re.sub(r"[^a-z0-9]+", "_",
                                  (md.get("department") or fn).lower()).strip("_")
                link_only.append({"_id": f"{dept_key}__{r['id']}", "chunk_id": r["id"],
                                  "domain": cat, "dept": md.get("department"),
                                  "url": md.get("source"), "source_file": fn,
                                  "message": "That section isn't published on the website yet."})
                continue
            rc = CLASS_MAP.get(cat, "prose")
            e = {"chunk_id": r["id"], "rag_class": rc, "domain": cat, "text": r["text"],
                 "source": md.get("source"),
                 "meta": {"name": md.get("name"), "designation": md.get("designation"),
                          "department": md.get("department"), "staff_type": md.get("staff_type"),
                          "role": md.get("role"), "subcategory": md.get("subcategory")}}
            if rc == "lab":
                e["meta"].update(parse_location(md.get("location")))
                e["meta"]["coordinators"] = md.get("coordinators")
                e["meta"]["capacity"]     = md.get("capacity")
                e["meta"]["hardware"]     = md.get("hardware")
                e["meta"]["software"]     = md.get("software") or md.get("software_note")
            add_chunk(e, fn)

    # ---- fee ----
    fee_docs = jload("fee_structure.json")

    def emit(doc, cat, blob):
        for k, (res, excl) in RES_KEYS.items():
            if isinstance(blob.get(k), (int, float)):
                fee_rows.append({
                    "_id": f"{doc['doc_id']}::{cat}::{res}",
                    "doc_id": doc["doc_id"], "program": doc.get("program"),
                    "semester": doc.get("semester"),
                    "admission_type": doc.get("admission_type"),
                    "batch": doc.get("batch"), "income_category": cat,
                    "category_bucket": bucket(cat), "residence": res,
                    "amount": int(blob[k]), "excludes_tuition": excl,
                    "tuition_note": doc.get("tuition_fee_note"),
                    "academic_year": "2026-27", "source_url": doc.get("source_url"),
                    "source_file": "fee_structure.json"})

    for doc in fee_docs:
        if "categories" in doc:
            for cn, blob in doc["categories"].items():
                emit(doc, cn, blob)
        else:
            emit(doc, doc.get("category", "all_categories"), doc)

    for r in fee_rows:
        r["content_hash"] = content_hash({k: v for k, v in r.items()
                                          if k not in ("content_hash", "ingested_at")})

    for doc in fee_docs:
        rows = [r for r in fee_rows if r["doc_id"] == doc["doc_id"]]
        add_chunk({"chunk_id": f"fee::{doc['doc_id']}", "rag_class": "prose", "domain": "fee",
                   "text": _fee_chunk_text(doc, rows), "source": doc.get("source_url"),
                   "meta": {"doc_id": doc["doc_id"], "program": doc.get("program"),
                            "semester": doc.get("semester"),
                            "admission_type": doc.get("admission_type"),
                            "title": doc.get("title"),
                            "quick_answer": doc.get("quick_answer"),
                            "source_url": doc.get("source_url")}},
                  "fee_structure.json")

    admission = jload("admission_data.json")
    for a in admission:
        a["_id"] = a["doc_id"]
        a["source_file"] = "admission_data.json"
        a["content_hash"] = content_hash(
            {k: v for k, v in a.items() if k not in ("content_hash", "ingested_at")})
    for d in fee_docs:
        d["_id"] = d["doc_id"]
        d["source_file"] = "fee_structure.json"
        d["content_hash"] = content_hash(
            {k: v for k, v in d.items() if k not in ("content_hash", "ingested_at")})

    return {"chunks": chunks, "fee_rows": fee_rows, "fee_docs": fee_docs,
            "admission_docs": admission, "link_only": link_only}


if __name__ == "__main__":
    c = build()
    for k, v in c.items():
        print(f"{k:16s} {len(v)}")
