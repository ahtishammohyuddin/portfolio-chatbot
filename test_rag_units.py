"""Unit tests for rag.py using a small SYNTHETIC knowledge base.

Nothing here reads data/ or contains real biography content. "Jordan Example"
is an invented person. Tests marked "documents current behaviour" pin down how
the code behaves today, including known weaknesses, so a change shows up as a
failing test rather than going unnoticed.
"""

import inspect
import math

import pytest

from rag import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    Chunk,
    Retriever,
    build_prompt,
    load_chunks,
)

SAMPLE = """\
# Jordan Example profile

Jordan is a fictional person used only in tests.

## Technical skills
Python, SQL, pandas, Plotly.
Databases: PostgreSQL and SQLite.

## Education
Bachelor degree in computer science from Example University.

## Languages
English and Spanish.

## Projects
Warehouse tracker using RFID tags and a small dashboard.
A volunteer scheduling tool for the local team.
"""


@pytest.fixture
def kb_dir(tmp_path):
    (tmp_path / "sample.md").write_text(SAMPLE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def chunks(kb_dir):
    return load_chunks(str(kb_dir))


@pytest.fixture
def retriever(chunks):
    return Retriever(chunks)


def headings(results):
    return [chunk.heading for chunk, _ in results]


# ----------------------------------------------------------------- chunking


def test_sections_become_chunks_that_keep_their_heading(chunks):
    assert [c.heading for c in chunks] == [
        "Introduction", "Technical skills", "Education", "Languages", "Projects",
    ]
    for c in chunks:
        assert c.source == "sample.md"
        assert c.text.startswith(c.heading + "\n")


def test_document_title_line_is_not_a_chunk(chunks):
    assert not any("Jordan Example profile" in c.text for c in chunks)


def test_text_before_first_section_is_filed_under_introduction(chunks):
    intro = chunks[0]
    assert intro.heading == "Introduction"
    assert "fictional person" in intro.text


def test_empty_sections_are_dropped(tmp_path):
    (tmp_path / "a.md").write_text("## Empty\n\n## Full\nsome text\n", encoding="utf-8")
    assert [c.heading for c in load_chunks(str(tmp_path))] == ["Full"]


def test_only_md_and_txt_files_are_loaded_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("## Beta\nbeta text\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("## Alpha\nalpha text\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("## Gamma\ngamma text\n", encoding="utf-8")
    (tmp_path / "d.json").write_text('{"x": 1}', encoding="utf-8")
    loaded = load_chunks(str(tmp_path))
    assert [(c.source, c.heading) for c in loaded] == [("a.md", "Alpha"), ("b.txt", "Beta")]


def test_empty_data_directory_gives_no_chunks(tmp_path):
    assert load_chunks(str(tmp_path)) == []


def test_long_section_is_split_on_line_breaks_without_losing_lines(tmp_path):
    lines = [f"alpha{i} beta{i} gamma{i} delta{i} epsilon{i}" for i in range(8)]  # 5 words each
    (tmp_path / "long.md").write_text("## Long\n" + "\n".join(lines) + "\n", encoding="utf-8")
    pieces = load_chunks(str(tmp_path), max_words=12)
    assert len(pieces) == 4  # two lines (10 words) fit; three (15) do not
    rebuilt = []
    for p in pieces:
        assert p.heading == "Long"
        head, _, body = p.text.partition("\n")
        assert head == "Long"
        assert len(body.split()) <= 12
        rebuilt.extend(body.splitlines())
    assert rebuilt == lines


def test_section_of_exactly_max_words_stays_whole_and_one_more_word_splits_it(tmp_path):
    exact = "\n".join(["a1 a2 a3 a4 a5", "b1 b2 b3 b4 b5"])  # 10 words
    (tmp_path / "exact.md").write_text("## Exact\n" + exact + "\n", encoding="utf-8")
    assert len(load_chunks(str(tmp_path), max_words=10)) == 1
    (tmp_path / "exact.md").write_text("## Exact\n" + exact + " b6\n", encoding="utf-8")  # 11 words, 2 lines
    assert len(load_chunks(str(tmp_path), max_words=10)) == 2


def test_pieces_are_filled_up_to_exactly_max_words(tmp_path):
    lines = [f"p{i} q{i} r{i} s{i} t{i}" for i in range(4)]  # 5 words each
    (tmp_path / "fill.md").write_text("## Fill\n" + "\n".join(lines) + "\n", encoding="utf-8")
    pieces = load_chunks(str(tmp_path), max_words=10)
    assert len(pieces) == 2  # a piece of exactly 10 words is allowed
    assert all(len(p.text.partition("\n")[2].split()) == 10 for p in pieces)


def test_single_overlong_line_cannot_be_split_further(tmp_path):
    # Documents current behaviour: splitting happens on line breaks only.
    one_line = " ".join(f"w{i}" for i in range(30))
    (tmp_path / "x.md").write_text("## Wall\n" + one_line + "\n", encoding="utf-8")
    pieces = load_chunks(str(tmp_path), max_words=10)
    assert len(pieces) == 1
    assert len(pieces[0].text.split()) == 31  # heading word + 30 words


# ---------------------------------------------------------------- retrieval


def test_topic_question_finds_the_matching_section_first(retriever):
    assert headings(retriever.search("Which databases?"))[0] == "Technical skills"
    assert headings(retriever.search("Where is the bachelor degree from?"))[0] == "Education"
    assert headings(retriever.search("Which languages?"))[0] == "Languages"
    assert headings(retriever.search("RFID tags?"))[0] == "Projects"


def test_k_limits_the_number_of_results(retriever):
    question = "Python SQL pandas bachelor degree languages RFID"
    assert len(retriever.search(question, k=1)) == 1
    assert len(retriever.search(question, k=2)) == 2


def test_results_are_ordered_best_first_with_valid_scores(retriever):
    results = retriever.search("Python SQL pandas bachelor degree languages RFID", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 + 1e-9 for s in scores)


def test_question_with_no_shared_vocabulary_returns_nothing(retriever):
    assert retriever.search("What is the favourite football club?") == []


def test_question_made_only_of_stop_words_returns_nothing(retriever):
    assert retriever.search("What is he and where did he go?") == []


def test_word_order_matters_through_two_word_phrases(tmp_path):
    # Same words in both sections; only the second has "red apple" as a phrase.
    # The phrase section is listed second so a tie would not put it first by accident.
    (tmp_path / "order.md").write_text(
        "## One\napple tree red\n\n## Two\nred apple tree\n", encoding="utf-8"
    )
    retriever = Retriever(load_chunks(str(tmp_path)))
    assert headings(retriever.search("red apple"))[0] == "Two"


def test_empty_question_returns_nothing(retriever):
    assert retriever.search("") == []


def test_default_search_floor_is_0_08():
    default = inspect.signature(Retriever.search).parameters["min_score"].default
    assert default == 0.08


def test_score_exactly_at_the_floor_is_included_and_just_above_is_excluded(retriever):
    question = "Which databases?"
    score = retriever.search(question, k=1, min_score=0.0)[0][1]
    assert retriever.search(question, k=1, min_score=score) != []  # >= keeps it
    assert retriever.search(question, k=1, min_score=math.nextafter(score, 2.0)) == []


def test_raising_the_floor_removes_weak_matches(retriever):
    question = "Python SQL pandas bachelor degree languages RFID"
    loose = retriever.search(question, k=5, min_score=0.0)
    strict = retriever.search(question, k=5, min_score=0.5)
    assert len(strict) < len(loose)
    assert all(s >= 0.5 for _, s in strict)


# ---- known weaknesses of the keyword gate (document current behaviour) ---------


def test_keyword_vocabulary_gap_study_vs_education(retriever):
    # "study" never appears in the notes, so the Education section is not found.
    assert retriever.search("Where did he study?") == []


def test_documents_current_behaviour_shared_common_words_open_the_gate(retriever):
    # The question is out of scope, but "local team" appears in a project note,
    # so the retrieval gate does NOT stop it. Only the model prompt would refuse.
    results = retriever.search("What is the favourite local team?")
    assert results and headings(results)[0] == "Projects"


def test_documents_current_behaviour_partly_out_of_scope_question_is_not_refused(retriever):
    results = retriever.search("What is the favourite football club and which languages?")
    assert results and headings(results)[0] == "Languages"


# -------------------------------------------------------------------- prompt


def test_build_prompt_numbers_the_passages_and_ends_with_the_question():
    results = [
        (Chunk("Skills\nPython", "s.md", "Skills"), 0.5),
        (Chunk("Languages\nEnglish", "s.md", "Languages"), 0.4),
    ]
    prompt = build_prompt("Which skills?", results)
    assert prompt.startswith("CONTEXT:\n[1] Skills\nPython\n\n[2] Languages\nEnglish")
    assert prompt.endswith("\n\nQUESTION: Which skills?")


def test_build_prompt_with_no_results_has_an_empty_context():
    # Documents current behaviour. app.py never calls it with no results.
    assert build_prompt("Anything?", []) == "CONTEXT:\n\n\nQUESTION: Anything?"


def test_system_prompt_keeps_its_grounding_and_refusal_rules():
    for required in (
        "Answer ONLY from the CONTEXT",
        "Never guess",
        "Never invent",
        "not the same as false",
        "third person",
        "no markdown",
    ):
        assert required in SYSTEM_PROMPT, f"system prompt lost the rule: {required!r}"


def test_no_answer_text_does_not_leak_internals_or_deny_a_skill():
    lowered = NO_ANSWER.lower()
    for word in ("context", "provided", "documents"):
        assert word not in lowered
    assert "does not have" not in lowered
    assert "contact" in lowered
