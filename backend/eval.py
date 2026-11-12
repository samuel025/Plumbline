"""Run the labeled test set through the pipeline: python backend/eval.py [data/eval.csv]"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import db, pipeline, seed

def main(path):
    seed.seed()
    rows = list(csv.DictReader(open(path)))
    tp = fn = fp = ok = 0
    for r in rows:
        with db.session() as c:
            sid = c.execute("INSERT INTO submission(church_id,content_type,raw_text,source_tool,ai_disclosed) VALUES(?,?,?,?,1)",
                            (int(r["church_id"]), r["category"], r["text"], "eval")).lastrowid
        got = pipeline.run(sid)["status"]
        exp = r["expected"]
        ok += got == exp
        if exp != "approved":
            tp += got != "approved"; fn += got == "approved"   # bad item let through = the costly miss
        else:
            fp += got != "approved"                             # good item wrongly flagged
        print(f"{'OK ' if got == exp else 'XX '} #{r['id']:>2} {r['category']:<17} expected={exp:<12} got={got}")
    bad = sum(r["expected"] != "approved" for r in rows); good = len(rows) - bad
    print(f"\nExact verdict match: {ok}/{len(rows)}")
    print(f"Bad items caught: {tp}/{bad}   (missed: {fn})")
    print(f"Good items wrongly flagged: {fp}/{good}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "eval.csv"))
