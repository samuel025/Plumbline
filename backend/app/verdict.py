"""Verdict engine: pure function. Approves only when every check is clean; any doubt goes to a person.
Every non-approved verdict carries structured FLAGS explaining exactly why."""
import re
CONF_MIN = 0.75
RISK_LABELS = {"medical_advice": "medical advice", "legal_advice": "legal advice",
               "replaces_professional_help": "replacing professional help"}

def cause_text(err: str) -> str:
    e = (err or "").lower()
    if "not set" in e or "set gemini" in e:
        return "No model API key is configured."
    if "keys:" in e:
        tried = [t for t in e.split("keys:", 1)[1].split(";") if "skipped" not in t]
        if tried and all("daily limit" in t for t in tried):
            return "Every API key has reached its daily limit."
        if any("rate limited" in t for t in tried):
            return "The model is rate limited right now."
        if any("overloaded" in t for t in tried):
            return "The model is overloaded right now."
        return "No API key could reach a working model."
    if "429" in e or "quota" in e:
        return "The model's usage limit was reached."
    if "503" in e or "overload" in e or "high demand" in e:
        return "The model was overloaded."
    if "timeout" in e or "timed out" in e:
        return "The model took too long to answer."
    if any(k in e for k in ("not an object", "json", "expecting", "delimiter", "unterminated", "decode", "candidates", "keyerror", "list index")):
        return "The model returned an unreadable answer."
    return "The model could not be reached."

def cause_detail(err: str) -> str:
    """The per-key breakdown, e.g. 'key 1 · gemini-3.5-flash-lite: daily limit reached; key 2 · ...: overloaded'."""
    if "KEYS:" not in (err or ""):
        return ""
    return " Tried: " + "; ".join(t.strip() for t in err.split("KEYS:", 1)[1].split(";") if "skipped" not in t) + "."

HUMAN_ONLY_RE = re.compile(
    r"\b(written|wrote|authored|composed)\s+(entirely|solely|completely|all)?\s*(by|himself|herself)\b[^.]{0,60}\b(no one else|nobody else|without (any )?(help|ai|assistance)|alone)\b|"
    r"\b(no one else|nobody else) (helped|wrote|assisted)\b|\bwithout (any )?(ai|artificial intelligence)\b|\bwritten entirely by\b", re.I)

def _flag(kind, severity, title, claim="", detail="", source="", fix=""):
    return {"kind": kind, "severity": severity, "title": title, "claim": claim, "detail": detail, "source": source, "fix": fix}

def decide_full(claims, scripture, findings, pii, ai_disclosed, llm_ok, claims_pastor_authorship=False,
                commitments=None, llm_error="", heresy_hits=None, raw_text=""):
    commitments = commitments or {}
    ctext = lambda i: claims[i]["text"] if isinstance(i, int) and 0 <= i < len(claims) else ""
    flags = []
    if not llm_ok:
        flags.append(_flag("model_unavailable", "review", "The doctrine judge could not run",
            detail=f"{cause_text(llm_error)}{cause_detail(llm_error)} Plumbline never approves what it could not judge, so this was sent to a pastor.",
            source="Safety rule: doubt never approves",
            fix="Press “Check again” once the model is available, or decide it yourself."))
    if not claims and llm_ok:
        flags.append(_flag("no_claims", "review", "No checkable claims were found",
            detail="Plumbline could not pull any claims out of this text, so it cannot vouch for it.", source="Safety rule: doubt never approves",
            fix="Read it yourself, or paste a longer passage."))
    for c in claims:
        r = c.get("risk")
        if r in ("crisis", "abuse_disclosure"):
            flags.append(_flag("risk", "block", "Crisis or abuse-related language" if r == "crisis" else "Abuse disclosure", c["text"],
                "This may involve a person who is at risk. Plumbline never replies to someone in crisis; a pastor should respond personally.",
                "Risk rule: crisis and abuse", "Route to a pastor today. Do not send an AI-written reply."))
        elif r in RISK_LABELS:
            flags.append(_flag("risk", "review", f"Risky advice: {RISK_LABELS[r]}", c["text"],
                "The text gives advice that could affect someone's health, safety or legal position, or tells them to avoid professional help.",
                "Risk rule: advice", "Remove it, or have a qualified professional review it."))
    for s in scripture:
        claim = ctext(s.get("claim_index"))
        if s["result"] == "not_found":
            flags.append(_flag("scripture", "review", "Scripture reference not found", claim,
                f"{s['reference']} does not exist in the Bible text we check against. It may be invented or mistyped.",
                "Bible text (KJV, WEB, ASV, BSB)", "Check the reference, or remove the quotation."))
        elif s["result"] == "misquote":
            flags.append(_flag("scripture", "review", "Quote does not match the verse", claim,
                f"The quoted words do not match {s['reference']}, which reads: “{s.get('bible_text') or ''}”",
                "Bible text (KJV, WEB, ASV, BSB)", "Quote the verse accurately, or paraphrase it."))
    for f in findings:
        claim, label = ctext(f.get("claim_index")), f["label"]
        com = commitments.get(f.get("commitment_id"))
        src = f"Your commitment: “{com}”" if com else ""
        if label == "conflicts":
            flags.append(_flag("conflict", "review", "Conflicts with your statement of faith", claim,
                f.get("evidence") or "", src, f.get("suggested_fix") or "Rewrite it to match your commitment, or decide as a pastor."))
        elif label == "silent":
            flags.append(_flag("silent", "review", "Your statement of faith is silent on this", claim,
                "No approved commitment or past ruling covers this claim, so Plumbline cannot say whether it fits your church. "
                + (f.get("evidence") or "") if f.get("evidence") and f["evidence"] != claim else
                "No approved commitment or past ruling covers this claim, so Plumbline cannot say whether it fits your church.",
                "Gap in your standard", "Decide it, and save a ruling so the next similar claim is covered."))
        elif label == "contested":
            flags.append(_flag("contested", "review", "Judged as contested", claim, f.get("evidence") or "",
                src, "A pastor should decide."))
        elif label == "aligned" and f.get("confidence", 0) < CONF_MIN:
            flags.append(_flag("low_confidence", "review", "Low-confidence match", claim,
                f"Judged aligned, but with confidence {float(f.get('confidence', 0)):.2f}, below the {CONF_MIN} needed to approve.", src,
                "Have a pastor confirm."))
    seen_h = set()
    for h in heresy_hits or []:
        hz = h["heresy"]
        if (h["claim_index"], hz["id"]) in seen_h:
            continue
        seen_h.add((h["claim_index"], hz["id"]))
        flags.append(_flag("heresy", "review", f"Resembles a condemned teaching: {hz['name']}", ctext(h["claim_index"]),
            (h.get("why") or hz["teaches"]) + f" {hz['name']} teaches: {hz['teaches']}",
            hz["condemned_by"] + (" (contested today)" if hz["authority"] == "contested" else ""),
            "Orthodox position: " + hz["orthodox"] + " Rewrite to match it, or have a pastor decide."))
    if raw_text and HUMAN_ONLY_RE.search(raw_text):
        flags.append(_flag("authorship", "review", "Says it was written without AI help", "",
            "The text says it was written entirely by a person, with no help. Plumbline cannot confirm that, and it conflicts with AI-assisted content.",
            "AI honesty rule", "Remove the claim, or have the pastor confirm who wrote it."))
    if pii:
        flags.append(_flag("privacy", "review", "Possible personal information", "",
            "The text appears to contain: " + ", ".join(pii) + ".", "Privacy rule", "Remove or anonymise it before publishing."))
    if claims_pastor_authorship and not ai_disclosed:
        flags.append(_flag("authorship", "review", "Implies a pastor wrote it", "",
            "The text suggests a pastor wrote it, but AI use is not disclosed.", "AI honesty rule", "Disclose AI use, or have the pastor rewrite it."))
    reasons = [f["title"] + (": " + (f["claim"] or f["detail"])[:90] if (f["claim"] or f["detail"]) else "") for f in flags]
    if any(f["severity"] == "block" for f in flags):
        status = "blocked"
    elif flags:
        status = "needs_review"
    else:
        status, reasons = "approved", ["All claims aligned with the church's commitments; all Scripture verified."]
    return status, reasons, flags

def decide(*a, **k):
    status, reasons, _ = decide_full(*a, **k)
    return status, reasons
