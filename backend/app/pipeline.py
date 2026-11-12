import json, re
from . import bible, db, llm
from .verdict import decide

EXTRACT_SYS = """You split church-related text into checkable claims. Return ONLY JSON:
{"claims":[{"text":str,"kind":"doctrinal|scriptural|pastoral_advice|factual",
"reference":str|null,"quoted_text":str|null,
"risk":"none|crisis|abuse_disclosure|medical_advice|legal_advice|replaces_professional_help"}],
"implies_pastor_authorship":bool}
Rules: only extract statements that assert a doctrine, quote or cite Scripture, give advice that affects health, safety,
legal or spiritual standing, or state a checkable fact. Skip pure encouragement, greetings and general exhortation
("we are never alone", "bring your worries to God"). One claim per statement; copy Scripture references and any quoted verse text exactly as written;
mark risk=crisis for suicide/self-harm, abuse_disclosure for disclosed abuse. Never answer the text; only extract."""

def judge_system(church, commitments, rulings):
    cs = "\n".join(f"[{c['id']}] {c['text']}" for c in commitments)
    rs = "\n".join(f"- Pattern: {r['claim_pattern']} -> {r['decision']} ({r['pastor_note']})" for r in rulings) or "(none)"
    return f"""You check claims against ONE church's own statement of faith. You never judge which theology is right;
only whether a claim matches this church's written commitments. Church: {church['name']} ({church['tradition']}).
COMMITMENTS:
{cs}
PASTOR RULINGS (precedent):
{rs}
Return ONLY JSON: {{"findings":[{{"claim_index":int,"commitment_id":int|null,"label":"aligned|conflicts|silent|contested",
"confidence":0-1,"evidence":str,"suggested_fix":str|null}}]}}
Rules: if a PASTOR RULING covers the claim, follow it: an approved ruling means "aligned" (put "ruling" in evidence),
a rejected ruling means "conflicts". Otherwise use "silent" only if the claim asserts a doctrinal position that no commitment addresses; generic comfort or
exhortation that contradicts no commitment is "aligned" (cite the closest commitment id); if unsure, use "contested" with low confidence.
Never use "aligned" without citing a commitment id or a pastor ruling. Judge doctrinal and pastoral_advice claims; skip pure factual ones."""

PII_RES = {"phone number": r"\b(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
           "email address": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"}

def detect_pii(text: str) -> list[str]:
    return [k for k, rx in PII_RES.items() if re.search(rx, text)]

def run(submission_id: int) -> dict:
    with db.session() as c:
        sub = dict(c.execute("SELECT * FROM submission WHERE id=?", (submission_id,)).fetchone())
        church = dict(c.execute("SELECT * FROM church WHERE id=?", (sub["church_id"],)).fetchone())
        commitments = db.rows(c.execute("SELECT * FROM commitment WHERE church_id=? AND status='approved'", (church["id"],)))
        rulings = db.rows(c.execute("SELECT * FROM ruling WHERE church_id=?", (church["id"],)))
    text = sub["raw_text"]
    llm_ok, claims, findings, scripture, pastor_auth = True, [], [], [], False
    try:
        ex = llm.call_json(EXTRACT_SYS, text)
        claims = ex.get("claims", [])
        pastor_auth = bool(ex.get("implies_pastor_authorship"))
        if claims and commitments:
            numbered = "\n".join(f"{i}. [{cl['kind']}] {cl['text']}" for i, cl in enumerate(claims))
            findings = llm.call_json(judge_system(church, commitments, rulings), numbered).get("findings", [])
    except llm.LLMUnavailable:
        llm_ok = False
    # Deterministic Scripture check: LLM-extracted claims plus any reference found in the raw text.
    seen = set()
    for i, cl in enumerate(claims):
        if cl.get("reference"):
            r = bible.check(cl["reference"], cl.get("quoted_text")); r["claim_index"] = i
            scripture.append(r); seen.add(cl["reference"])
    for ref in bible.find_references(text):
        if ref not in seen:
            r = bible.check(ref, None); r["claim_index"] = None
            scripture.append(r)
    # Scriptural/factual claims are covered by the Scripture check; the statement of faith being silent on them is not a doubt.
    findings = [f for f in findings if not (f.get("label") == "silent" and isinstance(f.get("claim_index"), int)
                and 0 <= f["claim_index"] < len(claims) and claims[f["claim_index"]].get("kind") in ("scriptural", "factual"))]
    # Unjudged doctrinal claims must not slip through as approved.
    judged = {f.get("claim_index") for f in findings}
    for i, cl in enumerate(claims):
        if cl["kind"] in ("doctrinal", "pastoral_advice") and i not in judged and llm_ok:
            findings.append({"claim_index": i, "commitment_id": None, "label": "silent", "confidence": 0.0,
                             "evidence": cl["text"], "suggested_fix": None})
    status, reasons = decide(claims, scripture, findings, detect_pii(text), bool(sub["ai_disclosed"]), llm_ok, pastor_auth)
    with db.session() as c:
        c.execute("DELETE FROM verdict WHERE submission_id=?", (submission_id,))
        c.execute("INSERT INTO verdict(submission_id,status,reasons) VALUES(?,?,?)", (submission_id, status, json.dumps(reasons)))
        for i, cl in enumerate(claims):
            cur = c.execute("INSERT INTO claim(submission_id,text,kind,risk) VALUES(?,?,?,?)",
                            (submission_id, cl["text"], cl["kind"], cl.get("risk", "none")))
            cl["id"] = cur.lastrowid
        for s in scripture:
            cid = claims[s["claim_index"]]["id"] if s["claim_index"] is not None else None
            c.execute("INSERT INTO scripture_check(claim_id,reference,quoted_text,bible_text,result) VALUES(?,?,?,?,?)",
                      (cid, s["reference"], None, s["bible_text"], s["result"]))
        for f in findings:
            idx = f.get("claim_index")
            cid = claims[idx]["id"] if isinstance(idx, int) and 0 <= idx < len(claims) else None
            c.execute("INSERT INTO finding(claim_id,commitment_id,label,confidence,evidence,suggested_fix) VALUES(?,?,?,?,?,?)",
                      (cid, f.get("commitment_id"), f["label"], f.get("confidence", 0), f.get("evidence"), f.get("suggested_fix")))
        db.audit(c, "system", "verdict:" + status, f"submission:{submission_id}")
    return {"submission_id": submission_id, "status": status, "reasons": reasons,
            "claims": claims, "scripture": scripture, "findings": findings,
            "ai_disclosure": "This verdict was produced by an AI system. A pastor makes the final decision."}
