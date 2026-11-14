import json, re
from . import bible, db, heresies, llm
from .verdict import decide_full

EXTRACT_SYS = """You split church-related text into checkable claims. Return ONLY JSON:
{"claims":[{"text":str,"kind":"doctrinal|scriptural|pastoral_advice|factual",
"reference":str|null,"quoted_text":str|null,
"risk":"none|crisis|abuse_disclosure|medical_advice|legal_advice|replaces_professional_help",
"stance":"asserts|refutes","stance_cue":str|null}],
"implies_pastor_authorship":bool}
Rules: only extract statements that assert a doctrine, quote or cite Scripture, give advice that affects health, safety,
legal or spiritual standing, or state a checkable fact. Skip pure encouragement, greetings and general exhortation
("we are never alone", "bring your worries to God"). One claim per statement; copy Scripture references and any quoted verse text exactly as written;
mark risk=crisis for suicide/self-harm, abuse_disclosure for disclosed abuse. Use medical_advice, legal_advice or
replaces_professional_help ONLY when the text itself gives such advice or tells someone to avoid or replace a doctor,
counselor or lawyer; advice that encourages seeking professional help is risk "none". Never answer the text; only extract.
Stance: "asserts" when the text itself holds or teaches the claim (the default). Use "refutes" ONLY when the text states the claim as someone else's view
in order to reject it ("Some say X, but Scripture says otherwise", "the error of X", "it is false that X"). For "refutes", copy into stance_cue the exact words
from the text that show the rejection; if you cannot quote such words, the stance is "asserts". If the text endorses the view ("Some say X, and they are right"), it is "asserts"."""

def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split())

# Words that mark a view as someone else's, or as rejected. The model's cue must contain one: it cannot just quote any phrase.
REJECT_RE = re.compile(
    r"\b(wrong(ly)?|false(ly)?|untrue|incorrect|mistaken(ly)?|misguided|misleading|error|errors|heresy|heresies|heretical|heretics?|myth|lie|lies|"
    r"reject(s|ed|ing)?|den(y|ies|ied)|refut\w+|condemn\w*|not true|is not what|says otherwise|say otherwise|teaches otherwise|"
    r"(some|many|others|people|critics|skeptics|they|those who|certain \w+)\s+(say|says|claim|claims|believe|believes|teach|teaches|think|thinks|argue|argues|insist|insists))\b", re.I)

def clean_stance(cl: dict, section_text: str) -> None:
    """'refutes' is trusted only when the quoted cue really appears in the text AND itself contains a rejection or
    attribution marker; otherwise the claim counts as asserted (the safe direction: it gets judged in full)."""
    cue = _norm(cl.get("stance_cue"))
    if cl.get("stance") == "refutes" and cue and cue in _norm(section_text) and REJECT_RE.search(cl.get("stance_cue") or ""):
        cl["stance"] = "refutes"
    else:
        cl["stance"] = "asserts"

REFUTES_TAG = " [the text REJECTS this view]"

def numbered(claims: list[dict]) -> str:
    return "\n".join(f"{i}. [{cl['kind']}] {cl['text']}" + (REFUTES_TAG if cl.get("stance") == "refutes" else "")
                     for i, cl in enumerate(claims))

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
Never use "aligned" without citing a commitment id or a pastor ruling. Judge doctrinal and pastoral_advice claims; skip pure factual ones.
Stance: a claim tagged [the text REJECTS this view] is a view the text quotes in order to reject it. Judge the text's rejection, not the view: if a commitment
also rejects that view (or affirms its opposite), the label is "aligned" (cite that commitment); if the view is what a commitment affirms, the text opposes the
church, so the label is "conflicts"; if no commitment bears on it, "silent". Untagged claims are the text's own position."""

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
    heresy_hits, rejected = [], set()
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
                clean_stance(cl, sec)
                cl["section"] = k
                claims.append(cl)
        if claims and commitments:
            sysmsg = judge_system(church, commitments, rulings)
            for start in range(0, len(claims), JUDGE_BATCH):
                batch = claims[start:start + JUDGE_BATCH]
                numbered_batch = numbered(batch)
                for f in llm.call_json(sysmsg, numbered_batch, max_tokens=6000).get("findings", []):
                    if isinstance(f.get("claim_index"), int) and 0 <= f["claim_index"] < len(batch):
                        f["claim_index"] += start
                        findings.append(f)
        if claims:   # global rule: applies to every church, whatever its own statement of faith says
            heresy_hits, rejected = heresies.detect(claims)
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
    # Rejecting a catalogued heresy needs no church-specific commitment: the church's silence on it is not a gap.
    findings = [f for f in findings if not (f.get("label") == "silent" and f.get("claim_index") in rejected)]
    # Unjudged doctrinal claims must not slip through as approved.
    judged = {f.get("claim_index") for f in findings}
    for i, cl in enumerate(claims):
        if cl["kind"] in ("doctrinal", "pastoral_advice") and i not in judged and i not in rejected and llm_ok:
            findings.append({"claim_index": i, "commitment_id": None, "label": "silent", "confidence": 0.0,
                             "evidence": cl["text"], "suggested_fix": None})
    status, reasons, flags = decide_full(claims, scripture, findings, detect_pii(text), bool(sub["ai_disclosed"]), llm_ok, pastor_auth,
                                         {c["id"]: c["text"] for c in commitments}, llm_error, heresy_hits, text)
    for h in heresy_hits:   # recorded alongside the other findings (the verdict flag is built in decide_full)
        findings.append({"claim_index": h["claim_index"], "commitment_id": None, "label": "heresy", "confidence": h["confidence"],
                         "evidence": f"{h['heresy']['name']}: {h['why']}", "suggested_fix": h["heresy"]["orthodox"]})
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
