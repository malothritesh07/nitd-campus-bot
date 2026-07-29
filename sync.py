"""Incremental corpus sync with soft deletes.

    python sync.py --dry-run       show what would change, touch nothing
    python sync.py                 apply
    python sync.py --purge 30      hard-delete records archived >30 days ago
    python sync.py --restore ID    bring one archived record back

Replaces the wipe-and-rebuild in ingest_all.py, which had three problems:
the collection was briefly empty mid-run, per-record state was destroyed, and
every chunk was re-embedded even when nothing about it had changed.

What this does instead, per collection:

    NEW        in source, not in DB          -> insert (embed if it needs a vector)
    CHANGED    content_hash differs          -> update  + re-embed
    UNCHANGED  hash matches                  -> touch last_verified only, no embed
    MISSING    in DB, no longer in source    -> ARCHIVE (status="archived"), never hard-delete

Archiving rather than deleting matters: retrieval filters on
status="approved", so an archived chunk stops being cited immediately, but it is
still recoverable and still auditable. A later --purge removes it for real.
"""
import argparse, os
from datetime import datetime, timedelta, timezone

from pymongo import UpdateOne

import db as D
import corpus

NEEDS_VECTOR = ("prose", "lab")
COLLECTIONS  = ("chunks", "fee_rows", "fee_docs", "admission_docs", "link_only")


def now():
    return datetime.now(timezone.utc)


def diff(coll_name, desired):
    """Compare desired records against what is live. Archived records are
    excluded from 'existing' so a re-appearing record is treated as NEW."""
    live = {d["_id"]: d for d in D.db[coll_name].find(
        {"status": {"$ne": "archived"}}, {"content_hash": 1})}
    want = {d["_id"]: d for d in desired}

    new       = [want[i] for i in want.keys() - live.keys()]
    gone      = sorted(live.keys() - want.keys())
    common    = want.keys() & live.keys()
    changed   = [want[i] for i in common if want[i].get("content_hash") != live[i].get("content_hash")]
    unchanged = [i for i in common if want[i].get("content_hash") == live[i].get("content_hash")]
    return new, changed, unchanged, gone


def embed(records):
    """Only new/changed prose+lab records reach here — the whole point of the diff."""
    todo = [r for r in records if r.get("rag_class") in NEEDS_VECTOR and r.get("text")]
    if not todo:
        return 0
    if os.getenv("EMBED_OFFLINE", "").strip() in ("1", "true", "True"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    vecs = model.encode([r["text"] for r in todo], normalize_embeddings=True,
                        show_progress_bar=len(todo) > 40, batch_size=32)
    for r, v in zip(todo, vecs):
        r["embedding"] = [float(x) for x in v]
        r["embed_model"] = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    return len(todo)


def apply(coll_name, new, changed, unchanged, gone, ts, run_id):
    ops = []
    embedded = embed(new + changed)

    for r in new + changed:
        r["ingested_at"] = ts
        r["status"] = "approved"
        r["archived_at"] = None
        r["sync_run"] = run_id
        ops.append(UpdateOne({"_id": r["_id"]}, {"$set": r}, upsert=True))

    if unchanged:
        ops.append(UpdateOne({"_id": {"$in": unchanged}},
                             {"$set": {"last_verified": ts}}, upsert=False))


    if gone:
        D.db[coll_name].update_many(
            {"_id": {"$in": gone}},
            {"$set": {"status": "archived", "archived_at": ts, "sync_run": run_id}})

    if ops:

        singles = [o for o in ops if not isinstance(o._filter.get("_id"), dict)]
        if singles:
            D.db[coll_name].bulk_write(singles, ordered=False)
        if unchanged:
            D.db[coll_name].update_many({"_id": {"$in": unchanged}},
                                        {"$set": {"last_verified": ts}})
    return embedded


def run(dry=False):
    built = corpus.build()
    ts, run_id = now(), now().strftime("%Y%m%dT%H%M%SZ")
    report, total_embed = {}, 0

    print(f"{'collection':16s} {'new':>5s} {'chg':>5s} {'same':>6s} {'archived':>9s}")
    print("-" * 48)
    for name in COLLECTIONS:
        new, changed, unchanged, gone = diff(name, built[name])
        report[name] = {"new": len(new), "changed": len(changed),
                        "unchanged": len(unchanged), "archived": len(gone),
                        "archived_ids": gone[:20]}
        print(f"{name:16s} {len(new):5d} {len(changed):5d} {len(unchanged):6d} {len(gone):9d}")
        if gone:
            for g in gone[:5]:
                print(f"                 - would archive: {g}")
            if len(gone) > 5:
                print(f"                   ... and {len(gone)-5} more")
        if not dry:
            total_embed += apply(name, new, changed, unchanged, gone, ts, run_id)

    print("-" * 48)
    if dry:
        print("DRY RUN — nothing written.")
        return report

    D.db.ingest_log.insert_one({"run_id": run_id, "ts": ts, "report": report,
                                "embedded": total_embed, "mode": "sync"})
    print(f"embedded this run: {total_embed}  (unchanged records were not re-embedded)")
    print(f"logged as ingest_log run_id={run_id}")
    return report


def purge(days):
    cutoff = now() - timedelta(days=days)
    total = 0
    for name in COLLECTIONS:
        res = D.db[name].delete_many({"status": "archived", "archived_at": {"$lt": cutoff}})
        if res.deleted_count:
            print(f"  {name:16s} hard-deleted {res.deleted_count}")
        total += res.deleted_count
    D.db.ingest_log.insert_one({"ts": now(), "mode": "purge",
                                "older_than_days": days, "deleted": total})
    print(f"purged {total} record(s) archived before {cutoff:%Y-%m-%d}")


def restore(rec_id):
    for name in COLLECTIONS:
        r = D.db[name].update_one({"_id": rec_id, "status": "archived"},
                                  {"$set": {"status": "approved", "archived_at": None}})
        if r.modified_count:
            print(f"restored {rec_id} in {name}")
            return
    print(f"no archived record with _id={rec_id}")


def status():
    print(f"{'collection':16s} {'approved':>9s} {'archived':>9s}")
    for name in COLLECTIONS:
        print(f"{name:16s} {D.db[name].count_documents({'status': {'$ne': 'archived'}}):9d} "
              f"{D.db[name].count_documents({'status': 'archived'}):9d}")
    last = D.db.ingest_log.find_one(sort=[("ts", -1)])
    if last:
        print(f"\nlast run: {last.get('run_id') or last['mode']} at {last['ts']:%Y-%m-%d %H:%M}Z")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", type=int, metavar="DAYS")
    ap.add_argument("--restore", metavar="ID")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        status()
    elif a.purge:
        purge(a.purge)
    elif a.restore:
        restore(a.restore)
    else:
        run(dry=a.dry_run)
