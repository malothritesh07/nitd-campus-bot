"""Write the default configuration into MongoDB.

    python seed_config.py          # only fills in sections that are missing
    python seed_config.py --force  # overwrite every section with the defaults

After this, thresholds/prompts/lexicon/UI live in the `config` collection and can
be changed without touching code.
"""
import sys
from datetime import datetime, timezone

import db as D
from config import DEFAULTS, CFG

force = "--force" in sys.argv
now = datetime.now(timezone.utc)

written, skipped = [], []
for section, value in DEFAULTS.items():
    existing = D.db.config.find_one({"_id": section})
    if existing and not force:
        skipped.append(section)
        continue
    D.db.config.replace_one(
        {"_id": section},
        {"_id": section, "value": value, "updated_at": now,
         "note": "edit `value` here; the API picks it up within CONFIG_RELOAD_SECONDS"},
        upsert=True)
    written.append(section)

print("written:", written or "(none)")
print("kept   :", skipped or "(none)")
print()
CFG.refresh()
print("read-back check:")
for probe in ("retrieval.entity_min_score", "generation.models",
              "ui.categories", "lexicon.dept_aliases"):
    v = CFG.get(probe)
    print(f"   {probe:28s} {str(v)[:70]}")
