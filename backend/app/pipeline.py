import json, re
from . import bible, db, llm
from .verdict import decide_full

EXTRACT_SYS = """You split church-related text into checkable claims. Return ONLY JSON:
{"claims":[{"text":str,"kind":"doctrinal|scriptural|pastoral_advice|factual",
"reference":str|null,"quoted_text":str|null,
"risk":"none|crisis|abuse_disclosure|medical_advice|legal_advice|replaces_professional_help"}],
"implies_pastor_authorship":bool}
Rules: only extract statements that assert a doctrine, quote or cite Scripture, give advice that affects health, safety,
legal or spiritual standing, or state a checkable fact. Skip pure encouragement, greetings and general exhortation
("we are never alone", "bring your worries to God"). One claim per statement; copy Scripture references and any quoted verse text exactly as written;
mark risk=crisis for suicide/self-harm, abuse_disclosure for disclosed abuse. Use medical_advice, legal_advice or
replaces_professional_help ONLY when the text itself gives such advice or tells someone to avoid or replace a doctor,
counselor or lawyer; advice that encourages seeking professional help is risk "none". Never answer the text; only extract."""

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
Rules: before choosing "silent", check EVERY commitment; if one states the same belief as the claim, the label is "aligned"
and you must cite that commitment's id. If a PASTOR RULING covers the claim, follow it: an approved ruling means "aligned" (put "ruling" in evidence),
a rejected ruling means "conflicts". Otherwise use "silent" only if the claim asserts a doctrinal position that no commitment addresses; generic comfort or
exhortation that contradicts no commitment is "aligned" (cite the closest commitment id); if unsure, use "contested" with low confidence.
Confidence guide: 0.9+ when a commitment states the same belief or the claim is generic comfort that contradicts nothing;
0.7-0.85 when the match is partial or you had to interpret; below 0.6 when unsure. Do not give 0.7 to a claim that plainly matches.
Never use "aligned" without citing a commitment id or a pastor ruling. Judge doctrinal and pastoral_advice claims; skip pure factual ones."""

NOTES_HINT = """
The text is sermon notes or an outline, so it may be fragments, bullets or shorthand ("grace - Eph 2:8-9 - not works").
Read each fragment as the statement it implies, expand abbreviated references (Eph 2:8 = Ephesians 2:8), and extract it as a claim."""

def extract_system(content_type: str) -> str:
    ct = (content_type or "").lower()
    return EXTRACT_SYS + (NOTES_HINT if "notes" in ct or "outline" in ct else "")

SECTION_WORDS = 600      # long text is checked in sections so no single model answer gets cut off
JUDGE_BATCH = 20         # claims judged per model call

def split_sections(text: str, target: int = SECTION_WORDS) -> list[str]:
    """Split on blank lines (then sentences/lines for very long blocks), packing pieces up to about `target` words."""
    units = []
    for para in re.split(r"\n\s*\n", text.strip()):
        if not para.strip():
            continue
        if len(para.split()) <= target * 1.5:
            units.append(para.strip())
        else:
            units += [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n", para) if u.strip()]
    out, cur, n = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and n + w > target:
            out.append("\n\n".join(cur)); cur, n = [], 0
        cur.append(u); n += w
    if cur:
        out.append("\n\n".join(cur))
    return out or [text.strip()]

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
    llm_ok, claims, findings, scripture, pastor_auth, llm_error = True, [], [], [], False, ""
    sections = split_sections(text)
    seen_claims: set[str] = set()
    try:
        for k, sec in enumerate(sections):
            ex = llm.call_json(extract_system(sub["content_type"]), sec, max_tokens=6000)
            pastor_auth = pastor_auth or bool(ex.get("implies_pastor_authorship"))
            for cl in ex.get("claims", []):
                key = " ".join(re.sub(r"[^a-z0-9 ]", "", (cl.get("text") or "").lower()).split())
                if not key or key in seen_claims:      # the same sentence repeated in a long text is judged once
                    continue
                seen_claims.add(key)
                cl["section"] = k
                claims.append(cl)
        if claims and commitments:
            sysmsg = judge_system(church, commitments, rulings)
            for start in range(0, len(claims), JUDGE_BATCH):
                batch = claims[start:start + JUDGE_BATCH]
                numbered = "\n".join(f"{i}. [{cl['kind']}] {cl['text']}" for i, cl in enumerate(batch))
                for f in llm.call_json(sysmsg, numbered, max_tokens=6000).get("findings", []):
                    if isinstance(f.get("claim_index"), int) and 0 <= f["claim_index"] < len(batch):
                        f["claim_index"] += start
                        findings.append(f)
    except llm.LLMUnavailable as e:
        llm_ok, llm_error = False, str(e)
    # Deterministic Scripture check: LLM-extracted claims plus any reference found in the raw text.
    seen = set()
    for i, cl in enumerate(claims):
        if cl.get("reference"):
            ref = bible.clean_ref(cl["reference"])
            r = bible.check(ref, cl.get("quoted_text")); r["claim_index"] = i
            scripture.append(r); seen.add(ref)
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
    status, reasons, flags = decide_full(claims, scripture, findings, detect_pii(text), bool(sub["ai_disclosed"]), llm_ok, pastor_auth,
                                         {c["id"]: c["text"] for c in commitments}, llm_error)
    sec_of = {cl["text"]: cl.get("section", 0) for cl in claims}
    for f in flags:
        if f.get("claim") in sec_of:
            f["section"] = sec_of[f["claim"]]
    with db.session() as c:
        # a re-check replaces the previous run's results instead of piling on top of them
        old = "claim_id IN (SELECT id FROM claim WHERE submission_id=?)"
        c.execute(f"DELETE FROM scripture_check WHERE submission_id=? OR {old}", (submission_id, submission_id))
        c.execute(f"DELETE FROM finding WHERE submission_id=? OR {old}", (submission_id, submission_id))
        c.execute("DELETE FROM claim WHERE submission_id=?", (submission_id,))
        c.execute("DELETE FROM verdict WHERE submission_id=?", (submission_id,))
        c.execute("INSERT INTO verdict(submission_id,status,reasons,flags) VALUES(?,?,?,?)",
                  (submission_id, status, json.dumps(reasons), json.dumps(flags)))
        for i, cl in enumerate(claims):
            cur = c.execute("INSERT INTO claim(submission_id,text,kind,risk,section) VALUES(?,?,?,?,?)",
                            (submission_id, cl["text"], cl["kind"], cl.get("risk", "none"), cl.get("section", 0)))
            cl["id"] = cur.lastrowid
        for s in scripture:
            cid = claims[s["claim_index"]]["id"] if s["claim_index"] is not None else None
            quoted = claims[s["claim_index"]].get("quoted_text") if s["claim_index"] is not None else None
            c.execute("INSERT INTO scripture_check(submission_id,claim_id,reference,quoted_text,bible_text,result,translation) VALUES(?,?,?,?,?,?,?)",
                      (submission_id, cid, s["reference"], quoted, s["bible_text"], s["result"], s.get("translation")))
        for f in findings:
            idx = f.get("claim_index")
            cid = claims[idx]["id"] if isinstance(idx, int) and 0 <= idx < len(claims) else None
            c.execute("INSERT INTO finding(submission_id,claim_id,commitment_id,label,confidence,evidence,suggested_fix) VALUES(?,?,?,?,?,?,?)",
                      (submission_id, cid, f.get("commitment_id"), f["label"], f.get("confidence", 0), f.get("evidence"), f.get("suggested_fix")))
        db.audit(c, "system", "verdict:" + status, f"submission:{submission_id}")
    return {"submission_id": submission_id, "status": status, "reasons": reasons, "flags": flags,
            "sections": len(sections), "claims": claims, "scripture": scripture, "findings": findings,
            "ai_disclosure": "This verdict was produced by an AI system. A pastor makes the final decision."}
