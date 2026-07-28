"""End-to-end check of every category through the running API."""
import json, sys, urllib.request

API = "http://127.0.0.1:8000/api/chat"

TESTS = [
 ("fee",       "btech 3rd sem josaa fee below 1 lakh day scholar", "20,300"),
 ("fee",       "mtech sfs 3rd sem day scholar fee",                "120,300"),
 ("fee",       "tuition fee for btech 3rd sem josaa",              "62,500"),
 ("fee",       "8th semester btech fee",                           None),
 ("lab",       "where is programming lab 1",                       "03"),
 ("lab",       "AIRIL lab capacity",                               "35"),
 ("lab",       "which lab has MATLAB",                             "MATLAB"),
 ("lab",       "location of airlib lab",                           "AIRIL"),
 ("lab",       "which lab has a quantum computer",                 None),
 ("faculty",   "who is dr halim",                                  "Assistant Professor"),
 ("faculty",   "hod of civil",                                     "Kapil"),
 ("faculty",   "faculty of cse",                                   "Geeta"),
 ("faculty",   "who is gautam kumar",                              "More than one"),
 ("faculty",   "who is dr rajesh mehta",                           None),
 ("syllabus",  "2nd semester subjects in cse",                     "Semester 2"),
 ("syllabus",  "units in csbb 103",                                "CSBB"),
 ("syllabus",  "discrete mathematics syllabus",                    "Graph"),
 ("admission", "what documents for MCA reporting",                 "Migration"),
 ("admission", "dasa documents needed",                            "assport"),
 ("about",     "vision of cse department",                         "ision"),
 (None,        "who is dr karan verma",                            "Karan"),
 (None,        "btech 5th sem josaa hostel ac above 5 lakh",       "129,800"),
]

NEG = ("not in my syllabus", "no lab matches", "no such lab", "no matching person",
       "isn't published", "not available", "don't have", "no fee record",
       "i have semesters", "no courses stored")

def call(cat, msg, state=None):
    body = json.dumps({"message": msg, "category": cat, "slots": state or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())

passed, fails = 0, []
for cat, msg, want in TESTS:
    try:
        d = call(cat, msg)
        ans = d.get("answer", "")
    except Exception as e:
        ans, d = f"ERROR {e}", {}
    ok = (any(n in ans.lower() for n in NEG) if want is None
          else want.lower() in ans.lower())
    passed += ok
    tag = "PASS" if ok else "FAIL"
    print(f"{tag} [{(cat or 'auto'):9s}/{d.get('method','-'):26s}] {msg}")
    if not ok:
        fails.append((msg, ans))
        print(f"       -> {ans[:170]}")

print(f"\nSCORE {passed}/{len(TESTS)}")

print("\n--- multi-turn fee ---")
st = {}
for m in ["btech 3rd sem josaa", "day scholar below 1 lakh", "hostel fee"]:
    d = call("fee", m, st); st = d.get("state", {}).get("slots") or d.get("slots") or st
    print(f"  YOU {m}\n  BOT {d['answer'].splitlines()[0][:100]}   [{d['method']}]")

print("\n--- multi-turn lab (pronoun) ---")
st = {}
d = call("lab", "AIRIL lab location", st); st = d.get("state", st)
print(f"  BOT {d['answer'][:90]}")
d = call("lab", "hardware in it", st)
print(f"  BOT {d['answer'][:110]}")
