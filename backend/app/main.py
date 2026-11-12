import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import db, pipeline, commitments as commitments_mod, llm
import os

app = FastAPI(title="Plumbline")

@app.on_event("startup")
def _startup():
    db.init()

class SubmitIn(BaseModel):
    church_id: int
    text: str
    content_type: str = "devotional"
    source_tool: str = "paste"
    ai_disclosed: bool = False

class DecideIn(BaseModel):
    decision: str  # approved | rejected | edited
    reviewer: str
    note: str = ""
    save_ruling: bool = False
    claim_pattern: str = ""
    edited_text: str = ""

class CommitmentIn(BaseModel):
    church_id: int
    text: str
    topic: str = ""
    source_excerpt: str = ""

@app.get("/api/churches")
def churches():
    with db.session() as c:
        return db.rows(c.execute("SELECT * FROM church"))

class ChurchIn(BaseModel):
    name: str
    tradition: str = ""
    tone_guide: str = ""

class StatementIn(BaseModel):
    text: str

class CommitmentEdit(BaseModel):
    text: str
    topic: str = ""

@app.post("/api/churches")
def create_church(b: ChurchIn):
    with db.session() as c:
        cur = c.execute("INSERT INTO church(name,tradition,tone_guide) VALUES(?,?,?)", (b.name, b.tradition, b.tone_guide))
        return {"id": cur.lastrowid}

@app.post("/api/churches/{cid}/statement")
def upload_statement(cid: int, b: StatementIn):
    """Extract DRAFT commitments from a statement of faith for a pastor to edit and approve."""
    if len(b.text.strip()) < 40:
        raise HTTPException(400, "Statement of faith is too short")
    try:
        return commitments_mod.extract(cid, b.text)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, f"Model unavailable: {e}")

@app.put("/api/commitments/{id}")
def edit_commitment(id: int, b: CommitmentEdit):
    with db.session() as c:
        c.execute("UPDATE commitment SET text=?, topic=?, version=version+1, status='draft' WHERE id=?", (b.text, b.topic, id))
    return {"id": id, "status": "draft"}  # an edit sends it back to draft until re-approved

@app.delete("/api/commitments/{id}")
def delete_commitment(id: int):
    with db.session() as c:
        c.execute("DELETE FROM commitment WHERE id=?", (id,))
    return {"ok": True}

@app.get("/api/churches/{cid}/commitments")
def commitments(cid: int):
    with db.session() as c:
        return db.rows(c.execute("SELECT * FROM commitment WHERE church_id=?", (cid,)))

@app.post("/api/commitments")
def add_commitment(b: CommitmentIn):
    with db.session() as c:
        cur = c.execute("INSERT INTO commitment(church_id,text,topic,source_excerpt) VALUES(?,?,?,?)",
                        (b.church_id, b.text, b.topic, b.source_excerpt))
        return {"id": cur.lastrowid, "status": "draft"}

@app.post("/api/commitments/{id}/approve")
def approve_commitment(id: int, reviewer: str = "pastor"):
    with db.session() as c:
        c.execute("UPDATE commitment SET status='approved' WHERE id=?", (id,))
        db.audit(c, reviewer, "approve_commitment", f"commitment:{id}")
    return {"id": id, "status": "approved"}

@app.post("/api/check")
def check(b: SubmitIn):
    with db.session() as c:
        cur = c.execute("INSERT INTO submission(church_id,content_type,raw_text,source_tool,ai_disclosed) VALUES(?,?,?,?,?)",
                        (b.church_id, b.content_type, b.text, b.source_tool, int(b.ai_disclosed)))
        sid = cur.lastrowid
    return pipeline.run(sid)

@app.get("/api/churches/{cid}/rulings")
def rulings(cid: int):
    with db.session() as c:
        return db.rows(c.execute("SELECT * FROM ruling WHERE church_id=? ORDER BY id DESC", (cid,)))

@app.delete("/api/rulings/{id}")
def delete_ruling(id: int):
    with db.session() as c:
        c.execute("DELETE FROM ruling WHERE id=?", (id,))
        db.audit(c, "pastor", "delete_ruling", f"ruling:{id}")
    return {"ok": True}

def _detail(c, sid: int):
    sub = c.execute("SELECT * FROM submission WHERE id=?", (sid,)).fetchone()
    if not sub:
        return None
    v = c.execute("SELECT * FROM verdict WHERE submission_id=?", (sid,)).fetchone()
    claims = db.rows(c.execute("SELECT * FROM claim WHERE submission_id=? ORDER BY id", (sid,)))
    scripture = db.rows(c.execute("SELECT * FROM scripture_check WHERE submission_id=? ORDER BY id", (sid,)))
    findings = db.rows(c.execute(
        """SELECT f.*, cm.text AS commitment_text FROM finding f LEFT JOIN commitment cm ON cm.id=f.commitment_id
           WHERE f.submission_id=? ORDER BY f.id""", (sid,)))
    for cl in claims:
        cl["scripture"] = [x for x in scripture if x["claim_id"] == cl["id"]]
        cl["findings"] = [x for x in findings if x["claim_id"] == cl["id"]]
    vd = dict(v) if v else {}
    if vd.get("reasons"):
        vd["reasons"] = json.loads(vd["reasons"])
    return {"submission": dict(sub), "verdict": vd, "claims": claims,
            "loose_scripture": [x for x in scripture if x["claim_id"] is None],
            "loose_findings": [x for x in findings if x["claim_id"] is None],
            "ai_disclosure": "This verdict was produced by an AI system. A pastor makes the final decision."}

@app.get("/api/submissions/{sid}")
def submission_detail(sid: int):
    with db.session() as c:
        d = _detail(c, sid)
    if not d:
        raise HTTPException(404)
    return d

@app.get("/api/churches/{cid}/submissions")
def submissions(cid: int, status: str = "", limit: int = 100):
    q = """SELECT s.id, s.content_type, s.source_tool, substr(s.raw_text,1,160) AS preview, s.created_at,
                  v.status, v.final_status, v.reviewer, v.decided_at
           FROM submission s LEFT JOIN verdict v ON v.submission_id=s.id WHERE s.church_id=?"""
    args = [cid]
    if status:
        q += " AND v.status=?"; args.append(status)
    q += " ORDER BY s.id DESC LIMIT ?"; args.append(limit)
    with db.session() as c:
        return db.rows(c.execute(q, args))

@app.get("/api/churches/{cid}/stats")
def stats(cid: int):
    with db.session() as c:
        by = {r["status"]: r["n"] for r in c.execute(
            """SELECT v.status, COUNT(*) n FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? GROUP BY v.status""", (cid,))}
        pending = c.execute(
            """SELECT COUNT(*) FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? AND v.status IN ('needs_review','blocked') AND v.final_status IS NULL""", (cid,)).fetchone()[0]
        decided = c.execute(
            """SELECT COUNT(*) FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? AND v.final_status IS NOT NULL""", (cid,)).fetchone()[0]
        com = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM commitment WHERE church_id=? GROUP BY status", (cid,))}
        rul = c.execute("SELECT COUNT(*) FROM ruling WHERE church_id=?", (cid,)).fetchone()[0]
    return {"total": sum(by.values()), "approved": by.get("approved", 0), "needs_review": by.get("needs_review", 0),
            "blocked": by.get("blocked", 0), "pending": pending, "decided": decided,
            "commitments_approved": com.get("approved", 0), "commitments_draft": com.get("draft", 0), "rulings": rul}

@app.get("/api/churches/{cid}/timeline")
def timeline(cid: int, days: int = 14):
    """Checks per day by verdict, zero-filled, oldest first."""
    from datetime import date, timedelta
    start = date.today() - timedelta(days=days - 1)
    out = {(start + timedelta(d)).isoformat(): {"approved": 0, "needs_review": 0, "blocked": 0} for d in range(days)}
    with db.session() as c:
        for r in c.execute(
            """SELECT date(s.created_at) d, v.status st, COUNT(*) n FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? AND date(s.created_at)>=? GROUP BY d, st""", (cid, start.isoformat())):
            if r["d"] in out and r["st"] in out[r["d"]]:
                out[r["d"]][r["st"]] = r["n"]
    return [{"day": k, **v} for k, v in out.items()]

@app.get("/api/health")
def health():
    """Is the judging model reachable? Reports state only; makes no model call."""
    import time
    keys = llm._gemini_keys()
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    live = [(ki, m) for ki in range(len(keys)) for m in llm.GEMINI_MODELS if llm._dead.get((ki, m), 0) <= time.time()]
    recent_fail = llm._last_fail > llm._last_ok and time.time() - llm._last_fail < 1800
    state = "offline" if not (keys or has_anthropic) else ("limited" if (not live or recent_fail) else "online")
    return {"judge": state, "keys": len(keys), "models": llm.GEMINI_MODELS}

@app.get("/api/audit")
def audit_log(limit: int = 60):
    with db.session() as c:
        return db.rows(c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)))

@app.get("/api/queue")
def queue(church_id: int):
    with db.session() as c:
        return db.rows(c.execute(
            """SELECT s.id, s.content_type, s.raw_text, substr(s.raw_text,1,200) AS preview, v.status, v.reasons
               FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? AND v.status IN ('needs_review','blocked') AND v.final_status IS NULL""", (church_id,)))

@app.post("/api/submissions/{sid}/decide")
def decide(sid: int, b: DecideIn):
    if b.decision not in ("approved", "rejected", "edited"):
        raise HTTPException(400, "bad decision")
    with db.session() as c:
        row = c.execute("SELECT church_id FROM submission WHERE id=?", (sid,)).fetchone()
        if not row:
            raise HTTPException(404)
        c.execute("UPDATE verdict SET final_status=?, reviewer=?, edited_text=?, decided_at=CURRENT_TIMESTAMP WHERE submission_id=?",
                  (b.decision, b.reviewer, b.edited_text or None, sid))
        if b.save_ruling and b.claim_pattern:
            c.execute("INSERT INTO ruling(church_id,claim_pattern,decision,pastor_note) VALUES(?,?,?,?)",
                      (row["church_id"], b.claim_pattern, b.decision, b.note))
        db.audit(c, b.reviewer, f"decide:{b.decision}", f"submission:{sid}")
    return {"ok": True}

_static = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

@app.get("/app", include_in_schema=False)
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(_static, "app.html"))

if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
