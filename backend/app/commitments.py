"""Turn a pasted statement of faith into DRAFT commitments. Nothing is used for judging until a pastor approves it."""
from . import db, llm

SYS = """You convert a church's statement of faith into a short list of doctrinal commitments.
Return ONLY JSON: {"commitments":[{"text":str,"topic":str,"source_excerpt":str}]}
Rules: one commitment per distinct belief, in plain words; do not add beliefs the text does not state;
source_excerpt must be copied verbatim from the text; topic is 1-3 words. Aim for 8-20 commitments."""

def verify_excerpt(excerpt: str, source: str) -> bool:
    norm = lambda s: " ".join((s or "").split()).lower()
    return bool(excerpt) and norm(excerpt) in norm(source)

def extract(church_id: int, statement: str) -> list[dict]:
    data = llm.call_json(SYS, statement, max_tokens=6000)
    out = []
    with db.session() as c:
        for item in data.get("commitments", []):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            exc = item.get("source_excerpt") or ""
            ok = verify_excerpt(exc, statement)
            cur = c.execute("INSERT INTO commitment(church_id,text,topic,source_excerpt,status) VALUES(?,?,?,?, 'draft')",
                            (church_id, text, item.get("topic", ""), exc if ok else ""))
            out.append({"id": cur.lastrowid, "text": text, "topic": item.get("topic", ""),
                        "source_excerpt": exc if ok else "", "excerpt_verified": ok, "status": "draft"})
        db.audit(c, "system", "extract_commitments", f"church:{church_id}")
    return out
