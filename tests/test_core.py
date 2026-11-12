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

def test_flags_explain_every_non_approved_verdict():
    from app.verdict import decide_full
    claims = [{"text": "God helps those who help themselves.", "kind": "doctrinal", "risk": "none"}]
    st, reasons, flags = decide_full(claims, [{"reference": "Hezekiah 4:2", "result": "not_found", "claim_index": 0, "bible_text": None}],
        [{"label": "conflicts", "confidence": .9, "evidence": "Contradicts grace alone.", "claim_index": 0, "commitment_id": 3, "suggested_fix": "Rely on grace."}],
        [], True, True, commitments={3: "Salvation is by grace alone."})
    assert st == "needs_review" and len(flags) == 2
    conflict = next(f for f in flags if f["kind"] == "conflict")
    assert conflict["claim"].startswith("God helps") and "grace alone" in conflict["source"] and conflict["fix"]
    assert all(f["title"] and f["detail"] for f in flags)

def test_model_down_explains_the_cause_and_never_approves():
    from app.verdict import decide_full
    st, _, flags = decide_full([{"text": "x", "kind": "doctrinal", "risk": "none"}], [], [], [], True, False,
                               llm_error="429 You exceeded your current quota")
    assert st == "needs_review" and flags[0]["kind"] == "model_unavailable" and "usage limit" in flags[0]["detail"]

def test_recheck_replaces_instead_of_duplicating(tmp_path, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_DB", str(tmp_path / "r.db"))
    import importlib
    from app import db, pipeline
    importlib.reload(db); importlib.reload(pipeline)
    from app import llm
    db.init()
    with db.session() as c:
        c.execute("INSERT INTO church(name) VALUES('x')")
        c.execute("INSERT INTO submission(church_id,raw_text,ai_disclosed) VALUES(1,'See Hezekiah 4:2.',1)")
    monkeypatch.setattr(pipeline.llm, "call_json", lambda *a, **k: (_ for _ in ()).throw(llm.LLMUnavailable("429 quota")))
    for _ in range(2):
        pipeline.run(1)
    with db.session() as c:
        assert c.execute("SELECT COUNT(*) FROM verdict").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM scripture_check").fetchone()[0] == 1
