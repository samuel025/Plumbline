import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import bible
from app.verdict import decide

def test_long_text_is_split_into_sections():
    from app.pipeline import split_sections
    para = "Grace is a gift from God to us. " * 70          # about 490 words
    secs = split_sections("\n\n".join([para] * 5))
    assert len(secs) > 1 and all(len(x.split()) <= 1000 for x in secs)
    assert len(split_sections("grace - Eph 2:8 - not works")) == 1
    assert len(split_sections("word. " * 3000)) > 1             # no blank lines: falls back to sentences

def test_other_public_domain_translations():
    web = "For God so loved the world, that he gave his only born Son, that whoever believes in him should not perish"
    r = bible.check("John 3:16", web)
    assert r["result"] == "exact" and r["translation"] in ("WEB", "BSB")
    bsb = bible.check("John 3:16", "that everyone who believes in Him shall not perish but have eternal life")
    assert bsb["result"] == "exact" and bsb["translation"] == "BSB"
    assert bible.check("Song of Solomon 1:1", None)["result"] == "exact"   # book naming consistent across translations
    assert bible.check("Acts 8:37", None)["result"] == "exact"             # verse only some translations carry

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

def test_cause_names_which_keys_failed():
    from app.verdict import cause_text, cause_detail
    err = "KEYS: key 1 · m: daily limit reached; key 2 · m: overloaded"
    assert cause_text(err) == "The model is overloaded right now."
    assert "key 2" in cause_detail(err) and "daily limit" in cause_detail(err)
    both = "KEYS: key 1 · m: daily limit reached; key 2 · m: daily limit reached"
    assert cause_text(both) == "Every API key has reached its daily limit."

def test_parser_ignores_trailing_text_and_second_object():
    from app.llm import _parse
    assert _parse('Here you go:\n```json\n{"a": 1}\n```\nHope that helps {not json}') == {"a": 1}
    assert _parse('{"findings": []}\n{"extra": true}') == {"findings": []}
    import pytest
    with pytest.raises(ValueError):
        _parse('[1, 2, 3]')


def test_file_upload_text_extraction():
    import io, zipfile
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    assert c.post("/api/extract-text?filename=a.txt", content=b"Grace alone saves.").json()["text"] == "Grace alone saves."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:r><w:t>First &amp; one.</w:t></w:r></w:p><w:p><w:r><w:t>Second para.</w:t></w:r></w:p></w:document>")
    r = c.post("/api/extract-text?filename=a.docx", content=buf.getvalue()).json()
    assert "First & one." in r["text"] and "Second para." in r["text"]
    assert c.post("/api/extract-text?filename=a.exe", content=b"x").status_code == 415
    assert c.post("/api/extract-text?filename=a.pdf", content=b"not a pdf").status_code == 422
    assert c.post("/api/extract-text?filename=a.txt", content=b"").status_code == 400


def test_refutes_stance_needs_a_real_quote_from_the_text():
    from app import pipeline
    text = "Some say good deeds earn salvation, but Scripture says otherwise."
    ok = {"text": "good deeds earn salvation", "stance": "refutes", "stance_cue": "but Scripture says otherwise"}
    pipeline.clean_stance(ok, text)
    assert ok["stance"] == "refutes"
    invented = {"text": "good deeds earn salvation", "stance": "refutes", "stance_cue": "this is plainly wrong"}
    pipeline.clean_stance(invented, text)
    assert invented["stance"] == "asserts"          # a cue that is not in the text is not trusted
    nocue = {"text": "x", "stance": "refutes"}
    pipeline.clean_stance(nocue, text)
    assert nocue["stance"] == "asserts"
    default = {"text": "x"}
    pipeline.clean_stance(default, text)
    assert default["stance"] == "asserts"           # doubt never counts as a refutation


def test_judge_input_tags_rejected_views():
    from app import pipeline
    out = pipeline.numbered([{"kind": "doctrinal", "text": "A", "stance": "refutes"}, {"kind": "doctrinal", "text": "B", "stance": "asserts"}])
    assert "A [the text REJECTS this view]" in out and "B [" not in out


def test_refutes_needs_a_rejection_marker_in_the_cue():
    from app import pipeline
    text = "Christ is two persons. Mary is the mother of the man Jesus, never the mother of God."
    flat = {"text": "Mary is the mother of the man Jesus", "stance": "refutes", "stance_cue": "never the mother of God"}
    pipeline.clean_stance(flat, text)
    assert flat["stance"] == "asserts"        # the cue is in the text but does not show a rejection
    text2 = "Some wrongly teach that Jesus was created."
    real = {"text": "Jesus was created", "stance": "refutes", "stance_cue": "Some wrongly teach"}
    pipeline.clean_stance(real, text2)
    assert real["stance"] == "refutes"


def test_human_only_authorship_claim_is_flagged():
    from app.verdict import decide_full
    st, _, flags = decide_full([], [], [], [], True, True, raw_text="This sermon was written entirely by Pastor James; no one else helped.")
    assert any(f["kind"] == "authorship" for f in flags) and st == "needs_review"


def test_scripture_exact_notfound_and_fragments():
    from app import bible, scripture
    kjv = bible.lookup("John 3:16", "KJV")
    assert scripture.verify("John 3:16", kjv)["result"] == "exact"
    assert scripture.verify("Hezekiah 4:2", "anything")["result"] == "not_found"
    assert scripture.verify("John 3:16", None)["result"] == "exact"
    # a real quote with the reference left out is found and credited to where it is
    r = scripture.verify(None, "for God so loved the world, that he gave his only begotten Son")
    assert r["result"] == "exact" and r["reference"].startswith("John 3:16")
    # right words, wrong reference: flagged, and says where the words really are
    r = scripture.verify("Romans 8:16", kjv)
    assert r["result"] == "misquote" and "John 3:16" in r["note"]
    # a fragment that begins mid-phrase is not 'exact' (it may drop a qualifier like 'the love of')
    assert not bible.contains_quote("money is the root of all evil", bible.lookup("1 Timothy 6:10", "KJV"))
    assert bible.contains_quote("the Lord is my shepherd; I shall not want", bible.lookup("Psalms 23:1", "KJV"))


def test_scripture_paraphrase_goes_through_the_meaning_check(monkeypatch):
    from app import llm, scripture
    seen = {}
    def fake(system, user, max_tokens=0):
        seen["user"] = user
        return {"results": [{"i": 0, "verdict": "distorted", "ref": "1 Timothy 6:10", "why": "It drops 'the love of'."}]}
    monkeypatch.setattr(llm, "call_json", fake)
    r = scripture.verify("1 Timothy 6:10", "Money is the root of all evil.")
    assert r["result"] == "misquote" and "love of" in r["note"]
    assert "love of money" in seen["user"]          # the model is shown the real verse text, not asked to recall it
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": []})
    assert scripture.verify("1 Timothy 6:10", "Money is the root of all evil.")["result"] == "unverified"   # no answer is never 'faithful'


def test_unreferenced_non_scripture_is_not_in_bible(monkeypatch):
    from app import llm, scripture
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": [{"i": 0, "verdict": "distorted", "ref": None, "why": "A saying, not Scripture."}]})
    r = scripture.verify(None, "God helps those who help themselves.")
    assert r["result"] == "not_in_bible" and r["reference"] is None
    assert scripture.verify(None, "Cleanliness is next to godliness.")["result"] == "not_in_bible"   # no passage even close: no model needed


COMMS = [{"id": 1, "text": "Salvation is by grace alone, through faith alone, in Christ alone; works are the fruit, not the root."},
         {"id": 2, "text": "Baptism is for believers only, by immersion."},
         {"id": 3, "text": "God is one in essence and three in persons."}]


def test_closest_commitments_ranks_by_shared_wording():
    from app import pipeline
    near = pipeline.closest_commitments("Our good works do not earn salvation; they grow out of it.", COMMS)
    assert near[0][0]["id"] == 1


def test_second_look_upgrades_only_on_yes_plus_shared_wording(monkeypatch):
    from app import llm, pipeline
    claims = [{"kind": "doctrinal", "text": "Our good works do not earn salvation; they grow out of it."}]
    silent = lambda: [{"claim_index": 0, "commitment_id": None, "label": "silent", "confidence": 0.0, "evidence": "x"}]
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": [{"i": 0, "commitment_id": 1, "same_belief": "yes", "why": "same"}]})
    out = pipeline.second_look(COMMS, claims, silent())
    assert out[0]["label"] == "aligned" and out[0]["commitment_id"] == 1
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": [{"i": 0, "commitment_id": 1, "same_belief": "partly", "why": "x"}]})
    assert pipeline.second_look(COMMS, claims, silent())[0]["label"] == "silent"          # only a clear yes upgrades
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": [{"i": 0, "commitment_id": 2, "same_belief": "yes", "why": "x"}]})
    assert pipeline.second_look(COMMS, claims, silent())[0]["label"] == "silent"          # yes for a commitment sharing no wording: refused
    conflict = [{"claim_index": 0, "commitment_id": 1, "label": "conflicts", "confidence": 0.9, "evidence": "x"}]
    monkeypatch.setattr(llm, "call_json", lambda *a, **k: {"results": [{"i": 0, "commitment_id": 1, "same_belief": "yes", "why": "x"}]})
    assert pipeline.second_look(COMMS, claims, conflict)[0]["label"] == "conflicts"       # a conflict is never overturned
