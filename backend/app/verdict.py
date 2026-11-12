"""Verdict engine: pure function. Approves only when every check is clean; any doubt goes to a person."""
CONF_MIN = 0.75

def decide(claims: list[dict], scripture: list[dict], findings: list[dict], pii: list[str],
           ai_disclosed: bool, llm_ok: bool, claims_pastor_authorship: bool = False) -> tuple[str, list[str]]:
    blocked, review = [], []
    if not llm_ok:
        review.append("Automated judging was unavailable, so a pastor must review.")
    if not claims and llm_ok:
        review.append("No checkable claims were extracted; a person should read it.")
    for c in claims:
        if c.get("risk") in ("crisis", "abuse_disclosure"):
            blocked.append(f"Crisis or abuse-related content ({c['risk']}): \"{c['text'][:80]}\". Route to a pastor; do not answer with AI.")
        elif c.get("risk") in ("medical_advice", "legal_advice", "replaces_professional_help"):
            review.append(f"Risky advice ({c['risk']}): \"{c['text'][:80]}\"")
    for s in scripture:
        if s["result"] == "not_found":
            review.append(f"Scripture reference not found: {s['reference']}")
        elif s["result"] == "misquote":
            review.append(f"Quote does not match {s['reference']}")
    for f in findings:
        if f["label"] == "conflicts":
            review.append(f"Conflicts with commitment: {f.get('evidence','')}")
        elif f["label"] in ("silent", "contested"):
            review.append(f"Statement of faith is {f['label']} on: {f.get('evidence','')}")
        elif f["label"] == "aligned" and f.get("confidence", 0) < CONF_MIN:
            review.append("Low-confidence judgment; needs a human look.")
    if pii:
        review.append("Possible personal information: " + ", ".join(pii))
    if claims_pastor_authorship and not ai_disclosed:
        review.append("Text implies a pastor wrote it but AI use is not disclosed.")
    if blocked:
        return "blocked", blocked + review
    if review:
        return "needs_review", review
    return "approved", ["All claims aligned with the church's commitments; all Scripture verified."]
