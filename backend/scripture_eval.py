"""Score the Scripture checker on data/scripture_eval.csv: python backend/scripture_eval.py [baseline|full]
Good items (exact, faithful) must pass; bad items (distorted, not_in_bible, wrong_ref) must be flagged."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import bible

PASS = {"exact", "paraphrase"}

def run_one(ref, quote, mode):
    if mode == "baseline":   # what the pipeline did before: only quotes that carry a reference are looked at
        return bible.check(ref, quote)["result"] if ref else "exact"
    from app import scripture
    return scripture.verify(ref, quote)["result"]

def main(mode="baseline"):
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "data", "scripture_eval.csv"))))
    good_bad = good_n = bad_miss = bad_n = 0
    if mode == "full":
        from app import scripture
        verdicts = scripture.verify_many([(r["reference"] or None, r["quote"]) for r in rows])
    for k, r in enumerate(rows):
        got = verdicts[k]["result"] if mode == "full" else run_one(r["reference"], r["quote"], mode)
        good = r["expected"] in ("exact", "faithful")
        ok = (got in PASS) == good
        good_n += good; bad_n += not good
        if not ok:
            if good: good_bad += 1
            else: bad_miss += 1
            note = (verdicts[k].get("note") or "")[:110] if mode == "full" else ""
            print(f"XX #{r['id']:>2} {r['expected']:<12} got={got:<12} {r['reference']:<16} {r['quote'][:50]}\n      {note}")
    print(f"\n[{mode}] bad items flagged: {bad_n - bad_miss}/{bad_n} (missed {bad_miss});  good items wrongly flagged: {good_bad}/{good_n}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
