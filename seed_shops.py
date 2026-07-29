"""Create the shop-status collections and issue owner codes.

Run once:  python seed.py
Re-runs are safe: shops are upserted, codes are only issued to staff who have none.
Codes are printed ONCE and never stored in plaintext.
"""
import secrets, string
from db import (db, shops, staff, ensure_indexes, code_lookup, code_hash, now_utc)

ALPHABET = string.ascii_uppercase + string.digits
AMBIGUOUS = set("O0I1")


def new_code(n: int = 8) -> str:
    return "".join(secrets.choice([c for c in ALPHABET if c not in AMBIGUOUS]) for _ in range(n))


SHOPS = [
    {"shop_id": "main_canteen", "name": "Main Canteen",
     "aliases": ["canteen", "main canteen", "mess canteen"],
     "location": "Near Academic Block", "display_order": 1,
     "owner_name": "TBD", "owner_email": ""},
    {"shop_id": "hk_cafe", "name": "HK Cafe",
     "aliases": ["hk", "hk cafe", "h k cafe", "hk canteen"],
     "location": "Near Admin Building", "display_order": 2,
     "owner_name": "TBD", "owner_email": ""},
    {"shop_id": "nescafe", "name": "Nescafe",
     "aliases": ["nescafe", "nes cafe", "coffee shop"],
     "location": "Campus Plaza", "display_order": 3,
     "owner_name": "TBD", "owner_email": ""},
    {"shop_id": "amul", "name": "Amul Parlour",
     "aliases": ["amul", "amul parlour", "ice cream"],
     "location": "Campus Plaza", "display_order": 4,
     "owner_name": "TBD", "owner_email": ""},
    {"shop_id": "health_centre", "name": "Health Centre",
     "aliases": ["health centre", "health center", "dispensary", "doctor", "clinic"],
     "location": "Institute Health Centre", "display_order": 5,
     "owner_name": "TBD", "owner_email": ""},
]


STAFF = [
    ("main_canteen",  "Canteen Manager"),
    ("hk_cafe",       "HK Cafe Owner"),
    ("nescafe",       "Nescafe Owner"),
    ("amul",          "Amul Parlour Owner"),
    ("health_centre", "Health Centre Attendant"),
]


def main() -> None:
    ensure_indexes()

    for s in SHOPS:
        shops.update_one({"shop_id": s["shop_id"]},
                         {"$set": {**s, "active": True}}, upsert=True)
    print(f"shops upserted: {shops.count_documents({})}")

    issued = []
    for shop_id, person in STAFF:
        existing = staff.find_one({"shop_id": shop_id, "name": person, "is_active": True})
        if existing:
            print(f"  code already issued for {person} — skipping (revoke to reissue)")
            continue
        code = new_code()
        staff.insert_one({
            "staff_id":    f"{shop_id}__{person.lower().replace(' ', '_')}",
            "shop_id":     shop_id,
            "name":        person,
            "phone":       "",
            "code_lookup": code_lookup(code),
            "code_hash":   code_hash(code),
            "is_active":   True,
            "issued_at":   now_utc(),
            "revoked_at":  None,
        })
        issued.append((shop_id, person, code))

    if issued:
        print("\n" + "=" * 58)
        print("  OWNER CODES — shown once, not recoverable later")
        print("=" * 58)
        for shop_id, person, code in issued:
            print(f"  {shop_id:15s} {person:26s} {code}")
        print("=" * 58)
        print("  Hand these to the named person only. To rotate: set")
        print("  is_active=false on their row and re-run this script.\n")
    else:
        print("\nno new codes issued")

    print("collections:", {c: db[c].estimated_document_count()
                           for c in ["shops", "shop_status", "shop_staff",
                                     "shop_audit", "toggle_attempts"]})


if __name__ == "__main__":
    main()
