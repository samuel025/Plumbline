"""Convert a bible.helloao.org complete.json into data/<id>.json: {"John 3:16": "text"}.
Usage: build_helloao.py <complete.json> <out.json>. Use only public-domain translations (KJV, WEB, ASV, BSB)."""
import json, sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
def flat(parts):
    out = []
    for p in parts:
        if isinstance(p, str): out.append(p)
        elif isinstance(p, dict) and "text" in p: out.append(p["text"])
        elif isinstance(p, dict) and "noteId" in p: out.append(" ")   # footnote marker sits between words
    return "".join(out)
NAMES = {"Song of Songs": "Song of Solomon", "Song": "Song of Solomon"}
out = {}
for book in raw["books"]:
    for ch in book["chapters"]:
        c = ch["chapter"]
        for item in c["content"]:
            if item.get("type") == "verse":
                t = " ".join(flat(item["content"]).split())
                if t: out[f"{NAMES.get(book['name'], book['name'])} {c['number']}:{item['number']}"] = t
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(out), "verses")
