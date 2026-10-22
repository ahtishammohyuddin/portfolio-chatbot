"""Retrieval checks against the real knowledge base in data/.

These are the eight original print-script checks, turned into assertions.
They exercise the keyword Retriever only (the embeddings path needs an API key
and is covered, with a fake embedder, in test_embed_retriever.py).

The knowledge base is private. Nothing here prints or quotes its contents:
failure messages name the question only, never the retrieved text or headings.
If data/ holds no .md or .txt file, every test in this module is skipped.

Run:  python -m pytest
"""

from pathlib import Path

import pytest

from rag import Retriever, load_chunks

DATA_DIR = Path(__file__).parent / "data"
MIN_SCORE_FLOOR = 0.08  # Retriever.search default

pytestmark = pytest.mark.real_kb


@pytest.fixture(scope="module")
def retriever():
    if not any(p.suffix.lower() in {".md", ".txt"} for p in DATA_DIR.glob("*")):
        pytest.skip("no knowledge base in data/")
    return Retriever(load_chunks(str(DATA_DIR)))


# (question, a word that must appear in the top result's section heading)
FOUND = [
    ("What databases does he know?", "databases"),
    ("Tell me about the MERIDIA ERP", "MERIDIA"),
    ("Did he build anything with RFID?", "School Bus"),
    ("What does he do at DevVibe?", "DevVibe"),
    ("Does he speak Urdu?", "Languages"),
]

# Not in the knowledge base: the bot must get no passages and so must refuse.
NOT_IN_KB = [
    "What is his CGPA?",
    "What is his favourite football team?",
]


@pytest.mark.parametrize("question, expected", FOUND)
def test_top_result_is_the_expected_section(retriever, question, expected):
    results = retriever.search(question)
    top_matches = bool(results) and expected.lower() in results[0][0].heading.lower()
    assert top_matches, f"top result for {question!r} is not the expected section"


@pytest.mark.parametrize("question, expected", FOUND)
def test_found_scores_meet_the_search_floor(retriever, question, expected):
    results = retriever.search(question)
    assert results, f"no results for {question!r}"
    below = sum(1 for _, score in results if score < MIN_SCORE_FLOOR)
    assert below == 0, f"{below} result(s) for {question!r} scored below the floor"


@pytest.mark.parametrize("question", NOT_IN_KB)
def test_question_outside_the_notes_gets_no_passages(retriever, question):
    count = len(retriever.search(question))  # count only, so a failure never prints text
    assert count == 0, f"{count} passage(s) returned for out-of-scope {question!r}"


# ---- "Where did he study?": the one original case that does not pass ----------
#
# The question says "study"; the notes say "Education". The keyword retriever
# matches shared words only, so it finds nothing. That is a documented
# limitation (README), not a regression. It is expressed twice on purpose:
#   1. the CURRENT behaviour, as a normal passing test, so it is pinned down;
#   2. the DESIRED behaviour, as a strict xfail, so it fails the suite the day
#      it starts passing and someone must promote it and retire test 1.

STUDY = "Where did he study?"


def test_current_keyword_behaviour_study_question_finds_nothing(retriever):
    count = len(retriever.search(STUDY))
    assert count == 0, "keyword retrieval now answers the study question: update these two tests"


@pytest.mark.xfail(
    strict=True,
    reason="Desired future behaviour: matching 'study' to 'Education' needs retrieval by meaning "
    "(embeddings). The keyword Retriever cannot do this.",
)
def test_desired_future_study_question_finds_the_education_section(retriever):
    results = retriever.search(STUDY)
    found = bool(results) and "education" in results[0][0].heading.lower()
    assert found, "top result for the study question is not the Education section"
