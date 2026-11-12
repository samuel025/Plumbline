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

@app.get("/api/queue")
def queue(church_id: int):
    with db.session() as c:
        return db.rows(c.execute(
            """SELECT s.id, s.content_type, substr(s.raw_text,1,200) AS preview, v.status, v.reasons
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
        c.execute("UPDATE verdict SET final_status=?, reviewer=?, decided_at=CURRENT_TIMESTAMP WHERE submission_id=?",
                  (b.decision, b.reviewer, sid))
        if b.save_ruling and b.claim_pattern:
            c.execute("INSERT INTO ruling(church_id,claim_pattern,decision,pastor_note) VALUES(?,?,?,?)",
                      (row["church_id"], b.claim_pattern, b.decision, b.note))
        db.audit(c, b.reviewer, f"decide:{b.decision}", f"submission:{sid}")
    return {"ok": True}

_static = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
