"""Database layer.

Postgres (Neon) when DATABASE_URL is set; SQLite otherwise (local dev and tests).
Call sites write SQLite-style SQL (`?` placeholders, `cursor.lastrowid`); the Postgres
wrapper translates, so the rest of the app is dialect-agnostic.
"""
import os, re, sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("PLUMBLINE_DB", os.path.join(os.path.dirname(__file__), "..", "..", "data", "plumbline.db"))


def database_url():
    # Read per call so tests / scripts can switch backends by setting the env var.
    # An explicit PLUMBLINE_DB (scratch SQLite file) wins over DATABASE_URL.
    if os.environ.get("PLUMBLINE_DB"):
        return ""
    return os.environ.get("DATABASE_URL", "")


def is_pg():
    return bool(database_url())


TABLES = [
    "CREATE TABLE IF NOT EXISTS church(id {pk}, name TEXT, tradition TEXT, tone_guide TEXT)",
    """CREATE TABLE IF NOT EXISTS commitment(id {pk}, church_id INT, text TEXT, topic TEXT,
       source_excerpt TEXT, status TEXT DEFAULT 'draft', version INT DEFAULT 1)""",
    """CREATE TABLE IF NOT EXISTS submission(id {pk}, church_id INT, content_type TEXT, raw_text TEXT,
       source_tool TEXT, ai_disclosed INT DEFAULT 0, created_at TEXT DEFAULT {now})""",
    "CREATE TABLE IF NOT EXISTS claim(id {pk}, submission_id INT, text TEXT, kind TEXT, span TEXT, risk TEXT)",
    """CREATE TABLE IF NOT EXISTS scripture_check(id {pk}, claim_id INT, reference TEXT, quoted_text TEXT,
       bible_text TEXT, result TEXT)""",
    """CREATE TABLE IF NOT EXISTS finding(id {pk}, claim_id INT, commitment_id INT, label TEXT,
       confidence REAL, evidence TEXT, suggested_fix TEXT)""",
    """CREATE TABLE IF NOT EXISTS verdict(id {pk}, submission_id INT UNIQUE, status TEXT, reasons TEXT,
       reviewer TEXT, decided_at TEXT, final_status TEXT)""",
    "CREATE TABLE IF NOT EXISTS ruling(id {pk}, church_id INT, claim_pattern TEXT, decision TEXT, pastor_note TEXT)",
    "CREATE TABLE IF NOT EXISTS audit_log(id {pk}, actor TEXT, action TEXT, target TEXT, at TEXT DEFAULT {now})",
]
# Added after the first release; applied to databases created by earlier versions.
MIGRATIONS = [
    ("verdict", "edited_text TEXT"),
    ("scripture_check", "submission_id INT"),
    ("finding", "submission_id INT"),
    ("verdict", "flags TEXT"),
    ("scripture_check", "translation TEXT"),
    ("claim", "section INT DEFAULT 0"),
]
_PG = dict(pk="SERIAL PRIMARY KEY", now="(to_char(now() AT TIME ZONE 'utc','YYYY-MM-DD HH24:MI:SS'))")
_LITE = dict(pk="INTEGER PRIMARY KEY", now="CURRENT_TIMESTAMP")


class Row(dict):
    """dict that also supports positional access, like sqlite3.Row."""
    def __getitem__(self, k):
        if isinstance(k, int):
            return list(self.values())[k]
        return super().__getitem__(k)


def _row_factory(cursor):
    cols = [d.name for d in cursor.description] if cursor.description else []
    return lambda values: Row(zip(cols, values))


def _to_pg(sql):
    return sql.replace("%", "%%").replace("?", "%s")


class _PgConn:
    """Mimics the slice of sqlite3.Connection the app uses."""
    def __init__(self, conn):
        self._c = conn

    def execute(self, sql, args=()):
        sql = sql.strip()
        want_id = bool(re.match(r"insert\s", sql, re.I)) and not re.search(r"\sreturning\s", sql, re.I)
        if want_id:
            sql += " RETURNING id"
        cur = self._c.execute(_to_pg(sql), tuple(args))
        cur.lastrowid = cur.fetchone()["id"] if want_id else None
        return cur

    def commit(self):
        self._c.commit()

    def close(self):
        self._c.close()


def connect():
    url = database_url()
    if url:
        import psycopg
        # prepare_threshold=None: safe behind Neon's pooled (PgBouncer) endpoint.
        conn = psycopg.connect(url, prepare_threshold=None, connect_timeout=15)
        conn.row_factory = _row_factory
        return _PgConn(conn)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        if is_pg():
            conn._c.rollback()
        raise
    finally:
        conn.close()


def init():
    if is_pg():
        with session() as c:
            for t in TABLES:
                c.execute(t.format(**_PG))
            for table, col in MIGRATIONS:
                c.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with session() as c:
        for t in TABLES:
            c.execute(t.format(**_LITE))
        for table, col in MIGRATIONS:
            try:  # lightweight migrations for databases created by earlier versions
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass


def rows(cur):
    return [dict(r) for r in cur.fetchall()]


def audit(c, actor, action, target):
    c.execute("INSERT INTO audit_log(actor,action,target) VALUES(?,?,?)", (actor, action, target))
