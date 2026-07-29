"""Builds the desired corpus from source files. Pure — touches no database.

Split out of ingest_all.py so the sync engine can compare "what should exist"
against "what does exist" without re-running any writes.
"""
import hashlib
import json
import os
import re

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
DATA = os.getenv("DATA_DIR", "M:\\")

def P(name):
    """Resolve a source filename against DATA_DIR."""
    return os.path.join(DATA, name)


def jload(name):
    with open(P(name), encoding="utf-8") as fh:
        return json.load(fh)


def jlload(name):
    with open(P(name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

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
    if "less_than_1" in c:
        return "below_1_lakh"
    if "between_1"   in c:
        return "1_to_5_lakh"
    if "above_5"     in c:
        return "above_5_lakh"
    if re.search(r"\bsc|st|ph|pwd", c) and "gen" not in c:
        return "sc_st_pwd"
    if re.search(r"gen|obc|ews|ur", c):
        return "gen_obc_ews"
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


class _ChunkCollector:
    """Accumulates chunks, stamping each with an id, source file and content hash.

    Chunk ids are not unique in the sources — HHPB 150 is two different subjects —
    so genuine collisions are suffixed rather than silently overwritten.
    """

    def __init__(self):
        self.chunks = []
        self._counts = {}

    def add(self, chunk, source_file):
        chunk_id = chunk["chunk_id"]
        if chunk_id in self._counts:
            self._counts[chunk_id] += 1
            chunk_id = f"{chunk_id}#{self._counts[chunk_id]}"
            chunk["duplicate_code"] = True
        else:
            self._counts[chunk_id] = 1
        chunk["_id"] = chunk_id
        chunk["source_file"] = source_file
        chunk["content_hash"] = content_hash({"t": chunk["text"], "m": chunk.get("meta")})
        self.chunks.append(chunk)


def _stamp_documents(docs, source_file):
    """Give whole documents the same id, provenance and hash treatment as chunks."""
    for doc in docs:
        doc["_id"] = doc["doc_id"]
        doc["source_file"] = source_file
        doc["content_hash"] = content_hash(
            {k: v for k, v in doc.items() if k not in ("content_hash", "ingested_at")})
    return docs


def _add_syllabus(collector):
    source = "rag_chunks_SYLLABUS.jsonl"
    for row in jlload(source):
        md = row.get("metadata", {})
        collector.add({
            "chunk_id": row["chunk_id"], "rag_class": "prose", "domain": "syllabus",
            "text": row["text"], "source": None,
            "meta": {"doc_id": row.get("doc_id"), "granularity": row.get("granularity"),
                     "module_label": row.get("module_label"),
                     "course_code": md.get("course_code"),
                     "course_title": md.get("course_title"),
                     "program": md.get("program"), "semester": md.get("semester"),
                     "academic_year": md.get("academic_year")}}, source)


def _link_only_record(row, md, category, filename):
    """Record an unpublished section as a link only, rather than a stub that
    would read as though the content were held."""
    dept_key = re.sub(r"[^a-z0-9]+", "_",
                      (md.get("department") or filename).lower()).strip("_")
    return {"_id": f"{dept_key}__{row['id']}", "chunk_id": row["id"],
            "domain": category, "dept": md.get("department"),
            "url": md.get("source"), "source_file": filename,
            "message": "That section isn't published on the website yet."}


def _add_departments(collector):
    link_only = []
    for filename in DEPT_FILES:
        if not os.path.exists(P(filename)):
            continue
        for row in jload(filename):
            md = row.get("metadata", {})
            category = md.get("category")
            if md.get("status") == "incomplete_needs_scraping":
                link_only.append(_link_only_record(row, md, category, filename))
                continue

            rag_class = CLASS_MAP.get(category, "prose")
            entry = {
                "chunk_id": row["id"], "rag_class": rag_class, "domain": category,
                "text": row["text"], "source": md.get("source"),
                "meta": {"name": md.get("name"), "designation": md.get("designation"),
                         "department": md.get("department"),
                         "staff_type": md.get("staff_type"), "role": md.get("role"),
                         "subcategory": md.get("subcategory")}}
            if rag_class == "lab":
                entry["meta"].update(parse_location(md.get("location")))
                entry["meta"]["coordinators"] = md.get("coordinators")
                entry["meta"]["capacity"] = md.get("capacity")
                entry["meta"]["hardware"] = md.get("hardware")
                entry["meta"]["software"] = md.get("software") or md.get("software_note")
            collector.add(entry, filename)
    return link_only


def _fee_rows_for(doc, category, blob):
    """One row per residence type, so a fee lookup is an indexed query rather
    than arithmetic over a document."""
    rows = []
    for key, (residence, excludes_tuition) in RES_KEYS.items():
        if not isinstance(blob.get(key), (int, float)):
            continue
        rows.append({
            "_id": f"{doc['doc_id']}::{category}::{residence}",
            "doc_id": doc["doc_id"], "program": doc.get("program"),
            "semester": doc.get("semester"),
            "admission_type": doc.get("admission_type"),
            "batch": doc.get("batch"), "income_category": category,
            "category_bucket": bucket(category), "residence": residence,
            "amount": int(blob[key]), "excludes_tuition": excludes_tuition,
            "tuition_note": doc.get("tuition_fee_note"),
            "academic_year": "2026-27", "source_url": doc.get("source_url"),
            "source_file": "fee_structure.json"})
    return rows


def _add_fees(collector):
    source = "fee_structure.json"
    fee_docs = jload(source)

    fee_rows = []
    for doc in fee_docs:
        if "categories" in doc:
            for category, blob in doc["categories"].items():
                fee_rows.extend(_fee_rows_for(doc, category, blob))
        else:
            fee_rows.extend(
                _fee_rows_for(doc, doc.get("category", "all_categories"), doc))

    for row in fee_rows:
        row["content_hash"] = content_hash(
            {k: v for k, v in row.items() if k not in ("content_hash", "ingested_at")})

    # A prose chunk per document keeps fees reachable by hybrid search when
    # metadata filtering cannot pin the answer down.
    for doc in fee_docs:
        rows = [r for r in fee_rows if r["doc_id"] == doc["doc_id"]]
        collector.add({
            "chunk_id": f"fee::{doc['doc_id']}", "rag_class": "prose", "domain": "fee",
            "text": _fee_chunk_text(doc, rows), "source": doc.get("source_url"),
            "meta": {"doc_id": doc["doc_id"], "program": doc.get("program"),
                     "semester": doc.get("semester"),
                     "admission_type": doc.get("admission_type"),
                     "title": doc.get("title"),
                     "quick_answer": doc.get("quick_answer"),
                     "source_url": doc.get("source_url")}}, source)

    return fee_rows, _stamp_documents(fee_docs, source)


def build():
    """Assemble the corpus from the source files.

    Every item carries a content hash and its originating file, which is what
    lets sync detect edits and whole-file removals.
    """
    collector = _ChunkCollector()
    _add_syllabus(collector)
    link_only = _add_departments(collector)
    fee_rows, fee_docs = _add_fees(collector)
    admission_docs = _stamp_documents(jload("admission_data.json"), "admission_data.json")

    return {"chunks": collector.chunks, "fee_rows": fee_rows, "fee_docs": fee_docs,
            "admission_docs": admission_docs, "link_only": link_only}


if __name__ == "__main__":
    c = build()
    for k, v in c.items():
        print(f"{k:16s} {len(v)}")
