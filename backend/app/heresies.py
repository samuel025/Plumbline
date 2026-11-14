"""Global check against historically condemned teachings (heresies).

Unlike the statement-of-faith judge, this applies to every church. The catalog (data/heresies.json) is curated;
each entry says who condemned it and how strongly (`authority`): ecumenical council, Reformation confessions,
a modern denunciation, or contested. The model only decides whether a claim *asserts* one of these teachings.
"""
import json, os
from . import llm

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "heresies.json")
BATCH = 20
MIN_CONFIDENCE = 0.5
CHECKED_KINDS = ("doctrinal", "scriptural", "pastoral_advice")

_cat = None

def catalog() -> list[dict]:
    global _cat
    if _cat is None:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            _cat = json.load(f)
    return _cat

def by_id() -> dict:
    return {h["id"]: h for h in catalog()}

def system_prompt() -> str:
    entries = "\n".join(
        f"- {h['id']}: {h['name']}. Teaches: {h['teaches']} Typical statements: " + "; ".join(f'"{m}"' for m in h["markers"])
        for h in catalog())
    return f"""You check claims from church content against a fixed list of historically condemned teachings.
You do not judge any church's own theology and you do not judge whether a claim is true in general; you only decide whether a
claim ASSERTS (or clearly implies) one of the listed teachings.
LIST:
{entries}
Return ONLY JSON: {{"matches":[{{"claim_index":int,"heresy_id":str,"stance":"endorses|refutes|unclear","confidence":0-1,"why":str}}]}}
Rules:
- A match needs the claim itself to state or clearly imply the teaching. Orthodox statements are NOT matches: "Jesus is fully God and fully man",
  "we are saved by grace through faith, and good works are its fruit", "obedience flows from gratitude" are orthodox.
- Do not match on shared vocabulary alone. A claim about grace is not antinomian unless it says obedience or holiness does not matter.
- stance "refutes" ONLY when the claim itself names or describes the teaching in order to reject it ("Some wrongly say...", "the heresy of...").
  If the claim states it flatly, or you cannot tell whether the speaker holds it, use "endorses" or "unclear". Doubt is never "refutes".
- Return nothing for claims that match no entry. One match per claim and entry at most; put the claim's index exactly as numbered.
- "why" is one plain sentence a pastor can read, naming what in the claim matches."""

def detect(claims: list[dict]) -> list[dict]:
    """Returns hits: {claim_index, heresy, stance, confidence, why}. Raises llm.LLMUnavailable if the model is unreachable."""
    idx = [i for i, c in enumerate(claims) if c.get("kind") in CHECKED_KINDS]
    cat, hits = by_id(), []
    sysmsg = system_prompt()
    for start in range(0, len(idx), BATCH):
        part = idx[start:start + BATCH]
        numbered = "\n".join(f"{j}. {claims[i]['text']}" for j, i in enumerate(part))
        for m in llm.call_json(sysmsg, numbered, max_tokens=4000).get("matches", []):
            j, h = m.get("claim_index"), cat.get(m.get("heresy_id"))
            conf = m.get("confidence")
            if not (isinstance(j, int) and 0 <= j < len(part) and h and isinstance(conf, (int, float))):
                continue
            if m.get("stance") == "refutes" or conf < MIN_CONFIDENCE:
                continue
            hits.append({"claim_index": part[j], "heresy": h, "stance": m.get("stance", "unclear"),
                         "confidence": float(conf), "why": (m.get("why") or "").strip()})
    return hits
