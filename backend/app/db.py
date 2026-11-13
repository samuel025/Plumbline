import json, os, sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("PLUMBLINE_DB", os.path.join(os.path.dirname(__file__), "..", "..", "data", "plumbline.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS church(id INTEGER PRIMARY KEY, name TEXT, tradition TEXT, tone_guide TEXT);
CREATE TABLE IF NOT EXISTS commitment(id INTEGER PRIMARY KEY, church_id INT, text TEXT, topic TEXT,
  source_excerpt TEXT, status TEXT DEFAULT 'draft', version INT DEFAULT 1);
CREATE TABLE IF NOT EXISTS submission(id INTEGER PRIMARY KEY, church_id INT, content_type TEXT, raw_text TEXT,
  source_tool TEXT, ai_disclosed INT DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS claim(id INTEGER PRIMARY KEY, submission_id INT, text TEXT, kind TEXT, span TEXT, risk TEXT);
CREATE TABLE IF NOT EXISTS scripture_check(id INTEGER PRIMARY KEY, claim_id INT, reference TEXT, quoted_text TEXT,
  bible_text TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS finding(id INTEGER PRIMARY KEY, claim_id INT, commitment_id INT, label TEXT,
  confidence REAL, evidence TEXT, suggested_fix TEXT);
CREATE TABLE IF NOT EXISTS verdict(id INTEGER PRIMARY KEY, submission_id INT UNIQUE, status TEXT, reasons TEXT,
  reviewer TEXT, decided_at TEXT, final_status TEXT);
CREATE TABLE IF NOT EXISTS ruling(id INTEGER PRIMARY KEY, church_id INT, claim_pattern TEXT, decision TEXT, pastor_note TEXT);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, actor TEXT, action TEXT, target TEXT, at TEXT DEFAULT CURRENT_TIMESTAMP);
"""

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with session() as c:
        c.executescript(SCHEMA)
        for stmt in ("ALTER TABLE verdict ADD COLUMN edited_text TEXT",
                     "ALTER TABLE scripture_check ADD COLUMN submission_id INT",
                     "ALTER TABLE finding ADD COLUMN submission_id INT",
                     "ALTER TABLE verdict ADD COLUMN flags TEXT",
                     "ALTER TABLE scripture_check ADD COLUMN translation TEXT",
                     "ALTER TABLE claim ADD COLUMN section INT DEFAULT 0"):
            try:  # lightweight migrations for databases created by earlier versions
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass

def rows(cur):
    return [dict(r) for r in cur.fetchall()]

def audit(c, actor, action, target):
    c.execute("INSERT INTO audit_log(actor,action,target) VALUES(?,?,?)", (actor, action, target))
