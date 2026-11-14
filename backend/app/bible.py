"""Scripture verification against bundled public-domain translations (KJV, WEB, ASV, BSB). A quote matches if it fits any of them."""
import difflib, json, os, re

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRANSLATIONS = {"KJV": "kjv.json", "WEB": "web.json", "ASV": "asv.json", "BSB": "bsb.json"}  # all public domain
_TEXTS: dict[str, dict[str, str]] = {}

BOOKS = ["Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges","Ruth","1 Samuel","2 Samuel",
 "1 Kings","2 Kings","1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job","Psalms","Proverbs",
 "Ecclesiastes","Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel","Hosea","Joel","Amos",
 "Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai","Zechariah","Malachi","Matthew","Mark","Luke",
 "John","Acts","Romans","1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians",
 "1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon","Hebrews","James","1 Peter",
 "2 Peter","1 John","2 John","3 John","Jude","Revelation"]
ALIASES = {"psalm": "Psalms", "song of songs": "Song of Solomon", "revelations": "Revelation"}

def verses(tr: str = "KJV") -> dict[str, str]:
    if tr not in _TEXTS:
        path = os.path.join(_DATA, TRANSLATIONS[tr])
        _TEXTS[tr] = json.load(open(path)) if os.path.exists(path) else {}
    return _TEXTS[tr]

def _alt():
    names = BOOKS + list(ALIASES)
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))

REF_RE = re.compile(rf"\b({_alt()})\.?\s+(\d{{1,3}}):(\d{{1,3}})(?:\s*[-–]\s*(\d{{1,3}}))?", re.I)

def normalize_book(name: str) -> str:
    low = name.lower()
    if low in ALIASES:
        return ALIASES[low]
    for b in BOOKS:
        if b.lower() == low:
            return b
    return name

GENERIC_RE = re.compile(r"\b((?:[1-3]\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\.?\s+(\d{1,3}):(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?")

_FILLER = {"as", "see", "in", "per", "cf", "and", "also", "the", "read", "from", "says", "of", "to", "at", "by", "or"}

def clean_ref(ref: str) -> str:
    """Drop leading filler words ('As Hezekiah 4:2' -> 'Hezekiah 4:2')."""
    parts = ref.split(" ")
    while len(parts) > 2 and parts[0].lower() in _FILLER:
        parts = parts[1:]
    return " ".join(parts)

def canonical_ref(ref: str) -> str:
    """'Psalm 23:1' -> 'Psalms 23:1' so the same verse is never listed twice."""
    m = REF_RE.search(ref or "")
    if not m:
        return ref
    return f"{normalize_book(m.group(1))} {m.group(2)}:{m.group(3)}" + (f"-{m.group(4)}" if m.group(4) else "")

def find_references(text: str) -> list[str]:
    """Known books plus any 'Capitalized N:N' pattern, so invented books (e.g. 'Hezekiah 4:2') are caught."""
    known = [f"{normalize_book(m.group(1))} {m.group(2)}:{m.group(3)}" + (f"-{m.group(4)}" if m.group(4) else "")
            for m in REF_RE.finditer(text)]
    extra = [f"{m.group(1)} {m.group(2)}:{m.group(3)}" for m in GENERIC_RE.finditer(text)
             if not REF_RE.fullmatch(m.group(0))]
    out = list(known)
    for e in (clean_ref(e) for e in extra):
        if e not in out and not any(e.split(" ", 1)[1] in k for k in known):
            out.append(e)
    return out

def lookup(reference: str, tr: str = "KJV") -> str | None:
    m = REF_RE.search(reference)
    if not m:
        return None
    book, ch, v1 = normalize_book(m.group(1)), m.group(2), int(m.group(3))
    v2 = int(m.group(4)) if m.group(4) else v1
    parts = []
    for v in range(v1, v2 + 1):
        t = verses(tr).get(f"{book} {ch}:{v}")
        if t is None:
            return None
        parts.append(t)
    return " ".join(parts)

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower().replace("—", " ")).split())

def _grade(quoted: str, found: str) -> tuple[str, float]:
    a, b = _norm(quoted), _norm(found)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or ratio >= 0.95:
        return "exact", ratio
    return ("paraphrase" if ratio >= 0.55 else "misquote"), ratio

def check(reference: str, quoted: str | None) -> dict:
    """result: exact | paraphrase | misquote | not_found. The quote is compared with every bundled translation
    and graded by the best match, so a correct ESV-style quote of a verse is not called a misquote just because it isn't KJV."""
    found = {tr: t for tr in TRANSLATIONS if (t := lookup(reference, tr)) is not None}
    if not found:
        return {"reference": reference, "bible_text": None, "result": "not_found", "translation": None}
    if not quoted or not quoted.strip():
        tr = "KJV" if "KJV" in found else next(iter(found))
        return {"reference": reference, "bible_text": found[tr], "result": "exact", "translation": tr}  # exists, nothing quoted
    best = max(found, key=lambda tr: _grade(quoted, found[tr])[1])
    res, ratio = _grade(quoted, found[best])
    return {"reference": reference, "bible_text": found[best], "result": res, "similarity": round(ratio, 2), "translation": best}


# ---------------------------------------------------------------------------------------------------------------
# Bible-wide search: find where a quote's words occur anywhere in the bundled translations.
# ---------------------------------------------------------------------------------------------------------------
import math
from array import array
from collections import defaultdict

_STOP = set("""a an the and or but if so as at by for from in into of on onto to unto with without upon is are was were be been am
it its he him his she her they them their we us our you your ye thou thee thy thine i me my not no nor that this these those which who whom
whose what when then than there here shall will would should could may might can do does did doth hath have has had said saith also all""".split())

def _stem(w: str) -> str:
    for suf in ("ing", "eth", "est", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w

def tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP]

_IDX: dict = {}

def _index() -> dict:
    """Inverted index over single verses of every translation: token -> array of doc ids. Built once, on first use."""
    if _IDX:
        return _IDX
    docs, post = [], defaultdict(lambda: array("I"))
    order, pos = {}, {}
    for tr in TRANSLATIONS:
        vs = verses(tr)
        order[tr] = list(vs)
        pos[tr] = {k: i for i, k in enumerate(order[tr])}
        for key, text in vs.items():
            d = len(docs)
            docs.append((tr, key))
            for t in set(tokens(text)):
                post[t].append(d)
    _IDX.update(docs=docs, post=dict(post), order=order, pos=pos, n=len(docs))
    return _IDX

def _chapter(key: str) -> str:
    return key.rsplit(":", 1)[0]

def _window(tr: str, start: int, size: int) -> tuple[str, str] | None:
    ix = _index(); keys = ix["order"][tr]
    if start < 0 or start + size > len(keys):
        return None
    part = keys[start:start + size]
    if len({_chapter(k) for k in part}) != 1:
        return None
    ref = part[0] if size == 1 else f"{part[0]} - {part[-1].rsplit(':', 1)[1]}"
    return ref, " ".join(verses(tr)[k] for k in part)

def _idf(t: str) -> float:
    ix = _index()
    return math.log(ix["n"] / (1 + len(ix["post"].get(t, ())))) + 1

def window_scores(qtoks: set, text: str) -> tuple[float, float]:
    """(coverage, precision): how much of the quote's meaningful words the passage contains, and how focused the passage is."""
    wt = set(tokens(text))
    hit = sum(_idf(t) for t in qtoks & wt)
    cov = hit / (sum(_idf(t) for t in qtoks) or 1)
    prec = hit / (sum(_idf(t) for t in wt) or 1)
    return cov, prec

_BOUNDARY = ",;:.!?—–-\"“‘'("

def _piece_in(piece: str, passage: str) -> bool:
    """The piece appears word-for-word, and begins at the start of the passage or after punctuation. A fragment that begins
    mid-phrase ('money is the root of all evil' out of 'the love of money is...') may have dropped a qualifier, so it is not exact."""
    words = re.findall(r"[a-z0-9]+", piece.lower())
    if not words:
        return False
    rx = re.compile(r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, words)) + r"(?![a-z0-9])", re.I)
    for m in rx.finditer(passage.replace("’", "'")):
        before = passage[:m.start()].rstrip()
        if not before or before[-1] in _BOUNDARY:
            return True
    return False

def contains_quote(quote: str, passage: str) -> bool:
    """Strict: every ellipsis-separated piece of the quote appears word-for-word in the passage, at a clause start."""
    pieces = [p for p in re.split(r"\.{2,}|…", quote) if re.search(r"[A-Za-z0-9]", p)]
    return bool(pieces) and all(_piece_in(p, passage) for p in pieces)

def search(quote: str, k: int = 3, max_window: int = 3) -> list[dict]:
    """Best-matching passages (1-3 verses) for a quote, across all translations, best first."""
    q = set(tokens(quote))
    if not q:
        return []
    ix = _index()
    score: dict[int, float] = defaultdict(float)
    for t in q:
        for d in ix["post"].get(t, ()):
            score[d] += _idf(t)
    best: dict[tuple, dict] = {}
    for d in sorted(score, key=score.get, reverse=True)[:30]:
        tr, key = ix["docs"][d]
        p = ix["pos"][tr][key]
        for size in range(1, max_window + 1):
            for off in range(size):
                w = _window(tr, p - off, size)
                if not w:
                    continue
                ref, text = w
                cov, prec = window_scores(q, text)
                exact = contains_quote(quote, text)
                s = (1.0 if exact else cov) + 0.15 * prec - 0.02 * (size - 1)
                cur = best.get((tr, ref))
                if not cur or s > cur["score"]:
                    best[(tr, ref)] = {"reference": ref, "translation": tr, "text": text, "coverage": round(cov, 2),
                                       "exact": exact, "score": s}
    out, seen = [], set()
    for r in sorted(best.values(), key=lambda r: r["score"], reverse=True):
        if r["reference"] in seen:      # same passage in another translation: keep the best-scoring one only
            continue
        seen.add(r["reference"]); out.append(r)
        if len(out) == k:
            break
    return out
