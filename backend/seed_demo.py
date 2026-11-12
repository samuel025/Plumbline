"""Optional: fill the database with clearly-labeled SYNTHETIC history so dashboards have something to show.
Rows are tagged source_tool='demo-seed'. Run: cd backend && ../.venv/bin/python seed_demo.py [--clear]"""
import csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import db, seed

REASONS = {
 "approved": ["All claims aligned with the church's commitments; all Scripture verified."],
 "needs_review": ["Statement of faith is silent on one claim, so a pastor should look.", "Scripture reference could not be verified."],
 "blocked": ["Crisis or abuse-related content. Route to a pastor; do not answer with AI."]}

def main():
    seed.seed(); db.init()
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "data", "eval.csv"))))
    rnd = random.Random(7)
    with db.session() as c:
        if "--clear" in sys.argv:
            ids = [r[0] for r in c.execute("SELECT id FROM submission WHERE source_tool='demo-seed'")]
            marks = ",".join("?" * len(ids)) or "NULL"
            for t in ("claim", "scripture_check", "finding", "verdict"):
                c.execute(f"DELETE FROM {t} WHERE submission_id IN ({marks})", ids)
            c.execute("DELETE FROM submission WHERE source_tool='demo-seed'")
            print("cleared", len(ids), "synthetic rows"); return
        for day in range(13, -1, -1):
            for _ in range(rnd.randint(2, 7)):
                r = rnd.choice(rows)
                st = r["expected"]
                ts = f"-{day} days"
                sid = c.execute("""INSERT INTO submission(church_id,content_type,raw_text,source_tool,ai_disclosed,created_at)
                    VALUES(?,?,?,?,1,datetime('now',?,?))""", (int(r["church_id"]), r["category"].replace("_", " "), r["text"], "demo-seed",
                    ts, f"-{rnd.randint(0, 600)} minutes")).lastrowid
                decided = st != "approved" and rnd.random() < .6
                c.execute("INSERT INTO verdict(submission_id,status,reasons,reviewer,decided_at,final_status) VALUES(?,?,?,?,?,?)",
                          (sid, st, json.dumps(REASONS[st]), "pastor" if decided else None,
                           __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if decided else None, rnd.choice(["approved", "rejected", "edited"]) if decided else None))
                c.execute("INSERT INTO claim(submission_id,text,kind,risk) VALUES(?,?,?,?)",
                          (sid, r["text"][:140], "pastoral_advice" if st == "blocked" else "doctrinal", "crisis" if st == "blocked" else "none"))
    print("seeded synthetic history")

main()
