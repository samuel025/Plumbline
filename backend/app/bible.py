"""Scripture verification against a bundled public-domain text (KJV)."""
import difflib, json, os, re

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kjv.json")
_VERSES: dict[str, str] = {}

BOOKS = ["Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges","Ruth","1 Samuel","2 Samuel",
 "1 Kings","2 Kings","1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job","Psalms","Proverbs",
 "Ecclesiastes","Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel","Hosea","Joel","Amos",
 "Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai","Zechariah","Malachi","Matthew","Mark","Luke",
 "John","Acts","Romans","1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians",
 "1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon","Hebrews","James","1 Peter",
 "2 Peter","1 John","2 John","3 John","Jude","Revelation"]
ALIASES = {"psalm": "Psalms", "song of songs": "Song of Solomon", "revelations": "Revelation"}

def verses() -> dict[str, str]:
    global _VERSES
    if not _VERSES:
        with open(_PATH) as f:
            raw = json.load(f)
        # source file names two-word books as "1" / "2"; rebuild from BOOKS order when needed
        _VERSES = raw
    return _VERSES

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

def find_references(text: str) -> list[str]:
    """Known books plus any 'Capitalized N:N' pattern, so invented books (e.g. 'Hezekiah 4:2') are caught."""
    known = [f"{normalize_book(m.group(1))} {m.group(2)}:{m.group(3)}" + (f"-{m.group(4)}" if m.group(4) else "")
            for m in REF_RE.finditer(text)]
    extra = [f"{m.group(1)} {m.group(2)}:{m.group(3)}" for m in GENERIC_RE.finditer(text)
             if not REF_RE.fullmatch(m.group(0))]
    return known + [e for e in extra if e not in known and not any(e.split(" ",1)[0] in k for k in known if e.split(" ",1)[1] in k)]

def lookup(reference: str) -> str | None:
    m = REF_RE.search(reference)
    if not m:
        return None
    book, ch, v1 = normalize_book(m.group(1)), m.group(2), int(m.group(3))
    v2 = int(m.group(4)) if m.group(4) else v1
    parts = []
    for v in range(v1, v2 + 1):
        t = verses().get(f"{book} {ch}:{v}")
        if t is None:
            return None
        parts.append(t)
    return " ".join(parts)

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower().replace("—", " ")).split())

def check(reference: str, quoted: str | None) -> dict:
    """result: exact | paraphrase | misquote | not_found"""
    found = lookup(reference)
    if found is None:
        return {"reference": reference, "bible_text": None, "result": "not_found"}
    if not quoted or not quoted.strip():
        return {"reference": reference, "bible_text": found, "result": "exact"}  # reference exists, nothing quoted
    a, b = _norm(quoted), _norm(found)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or ratio >= 0.95:
        res = "exact"
    elif ratio >= 0.55:
        res = "paraphrase"
    else:
        res = "misquote"
    return {"reference": reference, "bible_text": found, "result": res, "similarity": round(ratio, 2)}
