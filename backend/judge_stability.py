"""How steady is the church-specific judge? python backend/judge_stability.py [runs] [claims_cache.json]
Extracts each judge-driven eval item's claims ONCE, then reruns only the judge N times in two modes (plain vs steady),
so the numbers reflect the judge's own run-to-run variation. Needs a scratch DB: PLUMBLINE_DB=/tmp/x.db"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import db, llm, pipeline, seed
from app.verdict import CONF_MIN

CATS = {"clean", "drift", "church_dependent", "stance"}

def flagged(claims, findings):
    by = {}
    for f in findings:
        by.setdefault(f.get("claim_index"), f)
    for i, cl in enumerate(claims):
        if cl["kind"] not in ("doctrinal", "pastoral_advice") or cl.get("stance") == "refutes":
            continue
        f = by.get(i)
        if f is None or f["label"] != "aligned" or f.get("confidence", 0) < CONF_MIN:
            return True
    return False

def main(runs=3, cache=None):
    seed.seed()
    rows = [r for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "data", "eval.csv"))) if r["category"] in CATS]
    data = json.load(open(cache)) if cache and os.path.exists(cache) else {}
    with db.session() as c:
        churches = {r["id"]: dict(r) for r in c.execute("SELECT * FROM church").fetchall()}
        comms = {cid: db.rows(c.execute("SELECT * FROM commitment WHERE church_id=? AND status='approved'", (cid,))) for cid in churches}
        rul = {cid: db.rows(c.execute("SELECT * FROM ruling WHERE church_id=?", (cid,))) for cid in churches}
    for r in rows:
        if r["id"] in data:
            continue
        claims = []
        for sec in pipeline.split_sections(r["text"]):
            for cl in llm.call_json(pipeline.extract_system(r["category"]), sec, max_tokens=6000).get("claims", []):
                pipeline.clean_stance(cl, sec); claims.append(cl)
        data[r["id"]] = claims
    if cache:
        json.dump(data, open(cache, "w"))
    for mode in ("plain", "steady"):
        pipeline.STEADY = mode == "steady"
        res = {r["id"]: [] for r in rows}
        for _ in range(runs):
            for r in rows:
                cid = int(r["church_id"])
                try:
                    f = pipeline.judge_claims(churches[cid], comms[cid], rul[cid], data[r["id"]]) if data[r["id"]] else []
                    res[r["id"]].append(flagged(data[r["id"]], f))
                except llm.LLMUnavailable:
                    pass
        good = [r for r in rows if r["expected"] == "approved"]; bad = [r for r in rows if r["expected"] != "approved"]
        ff = [x for r in good for x in res[r["id"]]]; mm = [not x for r in bad for x in res[r["id"]]]
        print(f"[{mode:6}] good items wrongly flagged: {sum(ff)}/{len(ff)} judge-runs;  bad items missed: {sum(mm)}/{len(mm)} judge-runs")
        for r in rows:
            v = res[r["id"]]
            if v and len(set(v)) > 1:
                print(f"    unstable #{r['id']} ({r['expected']}): {['flag' if x else 'pass' for x in v]}")
            elif v and ((r["expected"] == "approved") == v[0]):
                print(f"    consistently wrong #{r['id']} ({r['expected']}): {['flag' if x else 'pass' for x in v]}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2] if len(sys.argv) > 2 else None)
