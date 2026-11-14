"""Scripture verification for quotes in church content.

Deterministic first: strict word-for-word matching, and a search of the whole Bible for where a quote's words occur.
Whatever is not an exact quote is then judged by the model *against the real verse text we supply* (never against its
own memory of Scripture): faithful, distorted, or unsure. Unsure and distorted go to a pastor.

verify() result values: exact | paraphrase | misquote | not_found | not_in_bible | unverified
"""
from . import bible, llm

CITED_MIN_COVERAGE = 0.45   # quote shares at least this much (idf-weighted) of its meaningful words with the cited passage
SEARCH_MIN_COVERAGE = 0.5   # same, for a Bible-wide match when no usable reference was given
BATCH = 12

MEANING_SYS = """You compare a quotation from church content with the real text of the Bible passages supplied. You are given the
verse text; never rely on your memory of Scripture. For each item decide whether the quotation faithfully conveys what the supplied
passage says.
Return ONLY JSON: {"results":[{"i":int,"verdict":"faithful|distorted|unsure","ref":str|null,"why":str}]}
- faithful: the quotation says the same thing as a passage (a rewording is fine) without adding a claim, dropping a condition,
  reversing, or changing who or what it is about. "ref" is the passage it matches.
- Check the subject and the predicate separately. The quotation must say the same thing about the same thing. If it drops or swaps a word
  that changes what is being said or about whom ("the love of money" shortened to "money"; "all who believe" to "all people"), it is distorted,
  even when most of the words match.
- distorted: it changes the meaning, reverses a point, adds something the passage does not say, or states as Scripture something
  the passages do not say. This includes popular sayings that are not in the Bible.
- unsure: you cannot tell from the supplied text. Doubt is never "faithful".
"why" is one plain sentence for a pastor: say what the passage actually says versus what the quotation claims."""


def _cited_passage(ref: str) -> dict[str, str]:
    return {tr: t for tr in bible.TRANSLATIONS if (t := bible.lookup(ref, tr)) is not None}


def _deterministic(ref: str | None, quote: str | None) -> dict:
    """Returns a final result dict, or {'candidate': [(ref, translation, text), ...]} needing the meaning check."""
    ref = bible.clean_ref(ref) if ref else None
    cited = _cited_passage(ref) if ref else {}
    if ref and not cited:                         # malformed, or a book/chapter/verse that does not exist
        return {"reference": ref, "bible_text": None, "result": "not_found", "translation": None}
    if not quote or not quote.strip():
        if not cited:
            return {"reference": ref, "bible_text": None, "result": "not_found", "translation": None}
        tr = "KJV" if "KJV" in cited else next(iter(cited))
        return {"reference": ref, "bible_text": cited[tr], "result": "exact", "translation": tr}   # exists; nothing quoted
    q = bible.tokens(quote)
    qset = set(q)
    if cited:
        for tr, text in cited.items():
            if bible.contains_quote(quote, text):
                return {"reference": ref, "bible_text": text, "result": "exact", "translation": tr}
        scored = {tr: bible.window_scores(qset, text)[0] for tr, text in cited.items()}
        best = max(scored, key=scored.get)
        if scored[best] >= CITED_MIN_COVERAGE:
            order = sorted(scored, key=scored.get, reverse=True)[:2]   # two wordings of the same passage help the comparison
            return {"candidate": [(ref, tr, cited[tr]) for tr in order]}
    hits = bible.search(quote, k=3)
    if hits and hits[0]["exact"]:
        h = hits[0]
        note = (f"The quoted words are word-for-word in {h['reference']} ({h['translation']}), not in {ref}." if ref
                else f"These words are {h['reference']} ({h['translation']}); the reference was left out.")
        return {"reference": ref or h["reference"], "found_at": h["reference"], "bible_text": h["text"], "translation": h["translation"],
                "result": "misquote" if ref else "exact", "note": note}
    if hits and hits[0]["coverage"] >= SEARCH_MIN_COVERAGE:
        return {"candidate": [(h["reference"], h["translation"], h["text"]) for h in hits], "cited": ref}
    if ref:
        return {"reference": ref, "bible_text": cited[next(iter(cited))] if cited else None, "result": "misquote",
                "translation": next(iter(cited)) if cited else None,
                "note": f"The quoted words do not match {ref}, and no passage of the Bible says anything like them."}
    return {"reference": None, "bible_text": None, "result": "not_in_bible", "translation": None,
            "note": "Presented as Scripture, but nothing in the Bible says this."}


def _meaning(items: list[tuple[int, str, list]]) -> dict[int, dict]:
    """items: (index, quote, passages). One batched model call per BATCH; raises llm.LLMUnavailable if the model is down."""
    out: dict[int, dict] = {}
    for s in range(0, len(items), BATCH):
        part = items[s:s + BATCH]
        lines = []
        for j, (_, quote, passages) in enumerate(part):
            vs = "\n".join(f"   [{r} {tr}] {t}" for r, tr, t in passages)
            lines.append(f"{j}. QUOTATION: {quote}\n   PASSAGES:\n{vs}")
        res = llm.call_json(MEANING_SYS, "\n".join(lines), max_tokens=3000).get("results", [])
        for m in res:
            j = m.get("i")
            if isinstance(j, int) and 0 <= j < len(part):
                out[part[j][0]] = m
    return out


def verify_many(items: list[tuple[str | None, str | None]]) -> list[dict]:
    """items: (reference or None, quoted text or None). Returns one result dict per item, in order."""
    results: list[dict | None] = [None] * len(items)
    pending = []
    for i, (ref, quote) in enumerate(items):
        d = _deterministic(ref, quote)
        if "candidate" in d:
            pending.append((i, quote, d["candidate"], d.get("cited"), ref))
        else:
            results[i] = d
    verdicts = _meaning([(i, q, c) for i, q, c, _, _ in pending]) if pending else {}
    for i, quote, cand, cited_hint, ref in pending:
        m = verdicts.get(i) or {}
        v, why = m.get("verdict"), (m.get("why") or "").strip()
        r0, tr0, t0 = cand[0]
        if v == "faithful":
            match = next((c for c in cand if c[0] == m.get("ref")), cand[0])
            res = {"reference": ref or match[0], "bible_text": match[2], "result": "paraphrase", "translation": match[1],
                   "note": why or None}
            if not ref:
                res["note"] = f"Matches {match[0]}; the reference was left out. " + (why or "")
        elif v == "distorted":
            res = {"reference": ref, "bible_text": t0 if ref else None, "result": "misquote" if ref else "not_in_bible",
                   "translation": tr0 if ref else None, "note": why or "The quotation does not say what the passage says."}
            if not ref:
                res["note"] += f" (Closest passage: {r0}.)"
        else:
            res = {"reference": ref or r0, "bible_text": t0, "result": "unverified", "translation": tr0,
                   "note": why or "Could not confirm that this wording matches the passage."}
        results[i] = res
    return results  # type: ignore[return-value]


def verify(reference: str | None, quote: str | None) -> dict:
    return verify_many([(reference, quote)])[0]
