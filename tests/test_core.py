import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import bible
from app.verdict import decide

def test_exact_and_missing_verse():
    assert bible.check("John 3:16", "For God so loved the world, that he gave his only begotten Son")["result"] == "exact"
    assert bible.check("Hezekiah 4:2", "x")["result"] == "not_found"
    assert bible.check("John 3:16", "God helps those who help themselves")["result"] == "misquote"

def test_clean_approves():
    st, _ = decide([{"text": "a", "kind": "doctrinal", "risk": "none"}], [],
                   [{"label": "aligned", "confidence": 0.9, "evidence": "c1"}], [], True, True)
    assert st == "approved"

def test_uncertainty_never_approves():
    for label in ("silent", "contested", "conflicts"):
        st, _ = decide([{"text": "a", "kind": "doctrinal", "risk": "none"}], [],
                       [{"label": label, "confidence": 0.9}], [], True, True)
        assert st == "needs_review"
    assert decide([{"text": "a", "kind": "x", "risk": "none"}], [], [], [], True, False)[0] == "needs_review"
    assert decide([], [], [], [], True, True)[0] == "needs_review"

def test_crisis_blocks():
    assert decide([{"text": "I want to die", "kind": "pastoral_advice", "risk": "crisis"}], [], [], [], True, True)[0] == "blocked"

def test_excerpt_must_appear_in_statement():
    from app.commitments import verify_excerpt
    src = "We believe in one God,\n eternally existing in three persons."
    assert verify_excerpt("one God, eternally existing", src)
    assert not verify_excerpt("God is a force", src)
    assert not verify_excerpt("", src)

def test_extract_makes_drafts_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_DB", str(tmp_path / "t.db"))
    import importlib
    from app import db, commitments
    importlib.reload(db); importlib.reload(commitments)
    db.init()
    with db.session() as c:
        c.execute("INSERT INTO church(name) VALUES('x')")
    monkeypatch.setattr(commitments.llm, "call_json", lambda *a, **k: {"commitments": [
        {"text": "God is one in three persons.", "topic": "trinity", "source_excerpt": "one God"},
        {"text": "Invented belief.", "topic": "x", "source_excerpt": "not in text"}]})
    out = commitments.extract(1, "We believe in one God, three persons.")
    assert [o["status"] for o in out] == ["draft", "draft"]
    assert [o["excerpt_verified"] for o in out] == [True, False]

def test_reference_not_duplicated_by_leading_word():
    refs = bible.find_references("As Hezekiah 4:2 says, and see John 3:16.")
    assert refs.count("Hezekiah 4:2") == 1 and "As Hezekiah 4:2" not in refs
    assert refs.count("John 3:16") == 1
