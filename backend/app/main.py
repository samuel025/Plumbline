import html, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import bible, db, pipeline, commitments as commitments_mod, llm
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
    vd["flags"] = json.loads(vd["flags"]) if vd.get("flags") else []
    secs = pipeline.split_sections(sub["raw_text"])
    return {"submission": dict(sub), "verdict": vd, "claims": claims,
            "sections": [{"index": i, "preview": " ".join(t.split()[:14]), "words": len(t.split())} for i, t in enumerate(secs)] if len(secs) > 1 else [],
            "loose_scripture": [x for x in scripture if x["claim_id"] is None],
            "loose_findings": [x for x in findings if x["claim_id"] is None],
            "ai_disclosure": "This verdict was produced by an AI system. A pastor makes the final decision."}

@app.post("/api/submissions/{sid}/recheck")
def recheck(sid: int):
    """Run the whole pipeline again on the same text (e.g. after the model was unavailable). Replaces the old result."""
    with db.session() as c:
        if not c.execute("SELECT 1 FROM submission WHERE id=?", (sid,)).fetchone():
            raise HTTPException(404)
        db.audit(c, "pastor", "recheck", f"submission:{sid}")
    r = pipeline.run(sid)
    return {"submission_id": sid, "status": r["status"]}

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
            """SELECT substr(s.created_at,1,10) d, v.status st, COUNT(*) n FROM submission s JOIN verdict v ON v.submission_id=s.id
               WHERE s.church_id=? AND substr(s.created_at,1,10)>=? GROUP BY substr(s.created_at,1,10), v.status""", (cid, start.isoformat())):
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
    return {"judge": state, "keys": len(keys), "models": llm.GEMINI_MODELS, "key_states": llm.key_states(),
            "last_trail": llm._last_trail}

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

MAX_UPLOAD = 5_000_000

def _docx_text(data: bytes) -> str:
    import io, re, zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n\n", xml)
    xml = re.sub(r"<w:tab/>|<w:br/>", " ", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))

def _pdf_text(data: bytes) -> str:
    import io
    from pypdf import PdfReader
    import unicodedata
    raw = "\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    return unicodedata.normalize("NFKC", raw)   # PDFs store ligatures like "ﬁ" as single characters

@app.post("/api/extract-text")
async def extract_text(request: Request, filename: str = "upload.txt"):
    """Read text out of an uploaded .txt, .md, .docx or .pdf (raw bytes in the body). Nothing is stored."""
    data = await request.body()
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(413 if data else 400, "File is empty or larger than 5 MB.")
    ext = os.path.splitext(filename.lower())[1]
    try:
        if ext in (".txt", ".md", ".text"):
            text = data.decode("utf-8", "replace")
        elif ext == ".docx":
            text = _docx_text(data)
        elif ext == ".pdf":
            text = _pdf_text(data)
        else:
            raise HTTPException(415, "Use a .txt, .md, .docx or .pdf file.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not read that file ({type(e).__name__}). Is it a valid {ext} file?")
    text = text.replace("\r\n", "\n").strip()
    if len(text) < 5:
        raise HTTPException(422, "No text found in that file. Scanned PDFs need OCR first.")
    return {"text": text, "words": len(text.split()), "filename": filename}

REWRITE_SYS = """You revise church-related text so that it fixes the listed problems and nothing else.
Return ONLY JSON: {"revised_text": str}
Rules: keep the author's voice, length and structure; change only the sentences named in the problems; remove or soften
claims that conflict with the church's commitments; quote Scripture exactly or paraphrase it without quotation marks;
never invent Bible references or add new doctrine; if a problem cannot be fixed by editing (crisis, abuse), return the text unchanged."""

@app.post("/api/submissions/{sid}/rewrite")
def rewrite(sid: int):
    """Propose a corrected version of a flagged submission. It is only a draft: a pastor still decides."""
    with db.session() as c:
        d = _detail(c, sid)
        if not d:
            raise HTTPException(404)
        coms = db.rows(c.execute("SELECT text FROM commitment WHERE church_id=? AND status='approved'", (d["submission"]["church_id"],)))
    flags = [f for f in d["verdict"].get("flags", []) if f["kind"] != "model_unavailable"]
    if not flags:
        raise HTTPException(400, "Nothing to fix: this submission has no content flags.")
    problems = "\n".join(f"- {f['title']}" + (f" (sentence: {f['claim']})" if f["claim"] else "") +
                         (f": {f['detail']}" if f["detail"] else "") + (f" Fix: {f['fix']}" if f["fix"] else "") for f in flags)
    user = "COMMITMENTS:\n" + "\n".join(f"- {x['text']}" for x in coms) + f"\n\nPROBLEMS:\n{problems}\n\nTEXT:\n{d['submission']['raw_text']}"
    try:
        out = llm.call_json(REWRITE_SYS, user, max_tokens=4000)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, f"The model could not write a revision: {e}")
    revised = (out.get("revised_text") or "").strip()
    if not revised:
        raise HTTPException(502, "The model returned no revision.")
    # the draft must not introduce references that do not exist
    bad = [r for r in bible.find_references(revised) if bible.check(r, None)["result"] == "not_found"]
    with db.session() as c:
        db.audit(c, "system", "suggest_rewrite", f"submission:{sid}")
    return {"revised_text": revised, "unverified_references": bad}

_LBL = {"approved": "Approved", "needs_review": "Needs pastor review", "blocked": "Blocked", "rejected": "Rejected", "edited": "Approved with edits"}

@app.get("/api/submissions/{sid}/report", include_in_schema=False)
def report(sid: int):
    """A self-contained printable page (use the browser's Print > Save as PDF)."""
    from fastapi.responses import HTMLResponse
    with db.session() as c:
        d = _detail(c, sid)
        if not d:
            raise HTTPException(404)
        church = c.execute("SELECT name FROM church WHERE id=?", (d["submission"]["church_id"],)).fetchone()
    e, v, sub = html.escape, d["verdict"], d["submission"]
    flags = "".join(f"<li><b>{e(f['title'])}</b>" + (f"<blockquote>{e(f['claim'])}</blockquote>" if f["claim"] else "") +
                    (f"<p>{e(f['detail'])}</p>" if f["detail"] else "") + (f"<p><i>What to do:</i> {e(f['fix'])}</p>" if f["fix"] else "") + "</li>"
                    for f in v.get("flags", [])) or "<li>No issues found.</li>"
    scr = "".join(f"<li><b>{e(x['reference'])}</b>: {e(x['result'].replace('_', ' '))}" + (f" ({e(x['translation'])})" if x.get("translation") else "") + "</li>"
                  for x in [*d["loose_scripture"], *[s for cl in d["claims"] for s in cl["scripture"]]]) or "<li>No Scripture references.</li>"
    final = (f"<p><b>Pastor decision:</b> {e(_LBL.get(v['final_status'], v['final_status']))} by {e(v.get('reviewer') or 'pastor')}, {e(v.get('decided_at') or '')}</p>"
             if v.get("final_status") else "")
    edited = f"<h2>Approved edited text</h2><blockquote>{e(v['edited_text'])}</blockquote>" if v.get("edited_text") else ""
    page = f"""<!doctype html><meta charset="utf-8"><title>Plumbline report #{sid}</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:720px;margin:40px auto;padding:0 20px;color:#222}}h1{{font-size:24px;margin:0}}h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ccc;padding-bottom:4px}}
blockquote{{margin:8px 0;padding:8px 14px;background:#f5f2ea;border-left:3px solid #b9a77a;white-space:pre-wrap}}.v{{font-size:20px;font-weight:bold}}.m{{color:#666;font-size:13px}}
@media print{{button{{display:none}}}}</style>
<button onclick="print()" style="float:right">Print / Save as PDF</button>
<h1>Plumbline review report</h1><div class="m">{e(church['name'] if church else '')} · submission #{sid} · {e(sub['content_type'] or 'content')} · {e(sub['created_at'] or '')}</div>
<h2>Verdict</h2><p class="v">{e(_LBL.get(v.get('status'), v.get('status') or ''))}</p>{final}
<h2>Content checked</h2><blockquote>{e(sub['raw_text'])}</blockquote>
<h2>Reasons</h2><ul>{flags}</ul><h2>Scripture checks</h2><ul>{scr}</ul>{edited}
<p class="m" style="margin-top:32px">{e(d['ai_disclosure'])} Scripture checked against KJV, WEB, ASV and BSB.</p>"""
    return HTMLResponse(page)

_static = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

@app.get("/app", include_in_schema=False)
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(_static, "app.html"))

if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
