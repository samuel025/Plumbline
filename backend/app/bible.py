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
