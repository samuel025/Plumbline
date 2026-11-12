# Plumbline

Review layer for church AI: checks AI-written text against a church's own statement of faith and against Scripture
(bundled public-domain KJV), then returns Approved / Needs pastor review / Blocked with evidence.
Design rule: uncertainty never approves. All data is synthetic.

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python scripts/build_bible.py <en_kjv.json> data/kjv.json   # already generated
    echo "GEMINI_API_KEY=your_key" > .env   # or ANTHROPIC_API_KEY; without a key every submission routes to a pastor
    (cd backend && ../.venv/bin/python -m app.seed)                      # seed two fictional churches
    cd backend && ../.venv/bin/uvicorn app.main:app --reload              # UI at http://localhost:8000
    .venv/bin/python backend/eval.py                                      # labeled eval
    .venv/bin/python -m pytest tests
