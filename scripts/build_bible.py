"""Convert the public-domain KJV JSON into data/kjv.json: {"John 3:16": "text"}."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
raw = json.load(open(src, encoding="utf-8-sig"))
out = {}
for book in raw:
    for ci, chapter in enumerate(book["chapters"], 1):
        for vi, text in enumerate(chapter, 1):
            out[f"{book['name']} {ci}:{vi}"] = text.strip()
json.dump(out, open(dst, "w"), ensure_ascii=False)
print(len(out), "verses")
