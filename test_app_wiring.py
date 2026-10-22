"""Offline wiring tests for app.py, using a fake Streamlit and a fake Gemini.

app.py runs its UI code at import, so each test executes it with runpy under:
  * a fake `streamlit` module that records what would be drawn and stands in for
    session state, secrets, caching and chat input;
  * a fake `google.genai` that records model and embedding calls and returns
    canned replies.
Synthetic notes only, written to a temporary directory that becomes the working
directory, so the real data/ and embeddings_cache.json are never read or written.
No network and no API key: socket connections are blocked while the app runs, and
every fake module, environment variable and the working directory is restored
after each run.

This tests the glue in app.py (mode setting, prompts, rendering, session
compatibility, usage limits). It does not prove real Streamlit or real Gemini
behave the same way as the fakes.
"""

import runpy
import socket
import sys
import types
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag import NO_ANSWER, SYSTEM_PROMPT, build_prompt, load_chunks
from grounding import GROUNDED_SYSTEM_PROMPT

APP = Path(__file__).parent / "app.py"
REPO_CACHE = Path(__file__).parent / "embeddings_cache.json"

QUESTION = "Which databases?"
NOTES = """\
# Jordan Example profile

## Technical skills
Python, SQL and pandas for data work.
Databases: PostgreSQL and SQLite.

## Education
Bachelor degree in computer science from Example University.

## Languages
English and Spanish.

## Projects
Warehouse tracker using RFID tags and a small dashboard.
"""

# Texts users see. Pinned here as literals so a wording change in app.py or
# grounding.py shows up as a failing test.
UNAVAILABLE = "Sorry, the AI service is unavailable right now. Please try again shortly."
RETRIEVAL_ONLY = "Closest matches from his notes are shown below."
SESSION_LIMIT_1 = "That's 1 questions, the limit for this free demo. To ask more, please contact Ahtisham directly."
DAILY_LIMIT = "This demo has reached its limit for today. Please try again tomorrow, or contact Ahtisham directly."
SOURCES_LABEL = "Quotes found in his notes:"

ENV_NAMES = ["GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBED_MODEL", "EMBED_MIN_SCORE",
             "MAX_QUESTIONS_PER_SESSION", "MAX_QUESTIONS_PER_DAY", "GROUNDING_MODE"]

VALID = ('{"answered": true, "answer": "He works with Python and SQL.", '
         '"citations": [{"ref": 1, "quote": "Python, SQL and pandas"}]}')

Event = namedtuple("Event", "path kind text")


class StopApp(BaseException):
    """Like Streamlit's st.stop(): not an Exception, so `except Exception` cannot swallow it."""


class Session(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


@dataclass
class AppRun:
    events: list = field(default_factory=list)
    model_calls: list = field(default_factory=list)
    embed_calls: int = 0
    session: Session = None
    cache: dict = field(default_factory=dict)
    stopped: bool = False

    def drawn(self, kind, path=("assistant",)):
        return [e.text for e in self.events if e.kind == kind and e.path == path]

    def sources(self, kind):
        return self.drawn(kind, ("assistant", "expander"))

    def all_text(self):
        return [str(e.text) for e in self.events if e.text is not None]


def _fake_streamlit(run, question):
    st = types.ModuleType("streamlit")
    st._is_fake = True
    stack = []

    def record(kind):
        return lambda text=None, *a, **k: run.events.append(Event(tuple(stack), kind, text))

    for name in ("set_page_config", "title", "caption", "info", "markdown", "text"):
        setattr(st, name, record(name))

    class Ctx:
        def __init__(self, name, label=None):
            self.name, self.label = name, label

        def __enter__(self):
            run.events.append(Event(tuple(stack), self.name, self.label))
            stack.append(self.name if self.name != "chat_message" else self.label)

        def __exit__(self, *exc):
            stack.pop()
            return False

    st.chat_message = lambda role: Ctx("chat_message", role)
    st.expander = lambda label: Ctx("expander", label)
    st.spinner = lambda label: Ctx("spinner", label)
    st.session_state = run.session
    st.secrets = {}
    st.chat_input = lambda *a, **k: question

    def stop():
        raise StopApp()

    st.stop = stop

    def cache_resource(fn):  # process-wide memo, like Streamlit's, shared across runs through run.cache
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            if key not in run.cache:
                run.cache[key] = fn(*args, **kwargs)
            return run.cache[key]
        return wrapper

    st.cache_resource = cache_resource
    return st


def _fake_google(run, reply, raises, embed_raises, embed_fail_from):
    class Resp:
        def __init__(self, text=None, embeddings=None):
            self.text, self.embeddings = text, embeddings

    class Models:
        def generate_content(self, model, contents, config):
            run.model_calls.append({"model": model, "contents": contents,
                                    "system": config.system_instruction, "temperature": config.temperature})
            if raises:
                raise RuntimeError("simulated quota error")
            return Resp(text=reply)

        def embed_content(self, model, contents, config):
            run.embed_calls += 1
            if embed_raises or (embed_fail_from and run.embed_calls >= embed_fail_from):
                raise RuntimeError("simulated embedding outage")
            # one identical unit vector for everything: every passage scores 1.0, in notes order
            return Resp(embeddings=[types.SimpleNamespace(values=[1.0, 0.0, 0.0]) for _ in contents])

    class Client:
        def __init__(self, api_key=None, http_options=None):
            self.models = Models()

    genai = types.ModuleType("google.genai")
    genai._is_fake = True
    genai.Client = Client
    gtypes = types.ModuleType("google.genai.types")
    gtypes.HttpOptions = lambda timeout=None: None
    gtypes.EmbedContentConfig = lambda **kw: None
    gtypes.GenerateContentConfig = lambda system_instruction=None, temperature=None: types.SimpleNamespace(
        system_instruction=system_instruction, temperature=temperature)
    genai.types = gtypes
    google = types.ModuleType("google")
    google.genai = genai
    return {"google": google, "google.genai": genai, "google.genai.types": gtypes}


@pytest.fixture
def run_app(tmp_path, monkeypatch):
    """run_app(question=..., env={...}, reply=..., ...) -> AppRun. Everything is undone after each call."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "sample.md").write_text(NOTES, encoding="utf-8")

    def run(question=None, *, env=None, reply="", raises=False, embed_raises=False, embed_fail_from=None,
            session=None, cache=None):
        result = AppRun(session=Session(session or {}), cache=cache if cache is not None else {})
        with monkeypatch.context() as m:
            m.chdir(tmp_path)
            for name in ENV_NAMES:
                m.delenv(name, raising=False)
            for name, value in (env or {}).items():
                m.setenv(name, value)

            def no_network(*a, **k):
                raise AssertionError("the app tried to open a network connection")

            m.setattr(socket.socket, "connect", no_network)
            m.setitem(sys.modules, "streamlit", _fake_streamlit(result, question))
            for name, module in _fake_google(result, reply, raises, embed_raises, embed_fail_from).items():
                m.setitem(sys.modules, name, module)
            try:
                runpy.run_path(str(APP), run_name="__main__")
            except StopApp:
                result.stopped = True
        return result

    run.expected_prompt = lambda q=QUESTION: build_prompt(
        q, [(c, 1.0) for c in load_chunks(str(data))[:3]])  # the first three passages, in notes order
    return run


KEY = {"GEMINI_API_KEY": "fake-key-not-real"}
ON = {**KEY, "GROUNDING_MODE": "on"}


# ------------------------------------------------------------ grounding mode off


def test_default_is_off_and_keeps_the_original_prompt_and_markdown_rendering(run_app):
    run = run_app(QUESTION, env=KEY, reply="A **markdown** answer")
    assert len(run.model_calls) == 1
    call = run.model_calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["contents"] == run_app.expected_prompt()
    assert (call["model"], call["temperature"]) == ("gemini-3.5-flash-lite", 0.2)
    assert run.drawn("markdown") == ["A **markdown** answer"]
    assert run.drawn("text") == [] and run.drawn("caption") == []
    saved = run.session["messages"][-1]
    assert saved["content"] == "A **markdown** answer"
    assert saved["plain"] is False and saved["citations"] == []


@pytest.mark.parametrize("value", ["off", "OFF", "", "true", "yes", "1", "onn", "enabled"])
def test_any_mode_value_other_than_on_behaves_as_off(run_app, value):
    run = run_app(QUESTION, env={**KEY, "GROUNDING_MODE": value}, reply='{"answered": true}')
    assert run.model_calls[0]["system"] == SYSTEM_PROMPT
    assert run.drawn("markdown") == ['{"answered": true}']  # not validated, shown as before


def test_off_mode_empty_reply_gives_the_standard_refusal(run_app):
    assert run_app(QUESTION, env=KEY, reply="").drawn("markdown") == [NO_ANSWER]


# ------------------------------------------------------------- grounding mode on


@pytest.mark.parametrize("value", ["on", "ON", " On "])
def test_on_mode_valid_answer_is_plain_text_with_the_revised_label_and_quote(run_app, value):
    run = run_app(QUESTION, env={**KEY, "GROUNDING_MODE": value}, reply=VALID)
    assert run.model_calls[0]["system"] == GROUNDED_SYSTEM_PROMPT
    assert run.model_calls[0]["contents"] == run_app.expected_prompt()
    assert run.drawn("text") == ["He works with Python and SQL.", '[1] Technical skills: "Python, SQL and pandas"']
    assert run.drawn("caption") == [SOURCES_LABEL]
    assert run.drawn("markdown") == []  # model text is never given to st.markdown
    saved = run.session["messages"][-1]
    assert saved["plain"] is True and saved["citations"] == ['[1] Technical skills: "Python, SQL and pandas"']
    assert not any("Verified" in t for t in run.all_text())


def test_on_mode_shows_the_passages_wording_for_a_quote_not_the_models(run_app):
    reply = ('{"answered": true, "answer": "He uses Python.", '
             '"citations": [{"ref": 1, "quote": "python,   SQL AND pandas"}]}')
    run = run_app(QUESTION, env=ON, reply=reply)
    assert run.drawn("text") == ["He uses Python.", '[1] Technical skills: "Python, SQL and pandas"']
    assert run.session["messages"][-1]["citations"] == ['[1] Technical skills: "Python, SQL and pandas"']


def test_on_mode_still_shows_the_retrieved_passages_under_the_answer(run_app):
    run = run_app(QUESTION, env=ON, reply=VALID)
    assert any(e.kind == "expander" and "search by meaning" in e.text for e in run.events)
    assert any(t.startswith("Technical skills\n") for t in run.sources("text"))


def test_on_mode_renders_markdown_and_html_in_an_answer_as_literal_text(run_app):
    nasty = "![x](http://example.invalid/p.png) <b>bold</b> **x**"
    reply = '{"answered": true, "answer": %s, "citations": [{"ref": 1, "quote": "Python, SQL and pandas"}]}' % (
        __import__("json").dumps(nasty))
    run = run_app(QUESTION, env=ON, reply=reply)
    assert nasty in run.drawn("text") and nasty not in run.drawn("markdown")


FABRICATED = '{"answered": true, "answer": "He knows Rust.", "citations": [{"ref": 1, "quote": "Expert in Rust systems"}]}'
NOT_RETRIEVED = ('{"answered": true, "answer": "He built a tracker.", '
                 '"citations": [{"ref": 1, "quote": "Warehouse tracker using RFID tags"}]}')  # real, but not retrieved
BAD_REF = '{"answered": true, "answer": "x y", "citations": [{"ref": 4, "quote": "Python, SQL and pandas"}]}'
MERGED_LINES = ('{"answered": true, "answer": "He uses Python and PostgreSQL.", '
                '"citations": [{"ref": 1, "quote": "for data work. Databases: PostgreSQL"}]}')  # crosses two lines
MID_WORD = ('{"answered": true, "answer": "He uses Python and SQL.", '
            '"citations": [{"ref": 1, "quote": "ython, SQL and pandas"}]}')  # starts inside a word
# Escaped JSON, so this source stays valid UTF-8: a lone UTF-16 surrogate in the answer / in a quote
SURROGATE_ANSWER = ('{"answered": true, "answer": "He uses Python \\ud800 daily", '
                    '"citations": [{"ref": 1, "quote": "Python, SQL and pandas"}]}')
SURROGATE_QUOTE = ('{"answered": true, "answer": "He uses Python daily", '
                   '"citations": [{"ref": 1, "quote": "Python, SQL and pandas\\udc00"}]}')
REFUSAL_AS_ANSWER = ('{"answered": true, "answer": "I don\'t have that information.", '
                     '"citations": [{"ref": 1, "quote": "Python, SQL and pandas"}]}')


@pytest.mark.parametrize(
    "reply",
    [FABRICATED, NOT_RETRIEVED, BAD_REF, MERGED_LINES, MID_WORD, SURROGATE_ANSWER, SURROGATE_QUOTE, REFUSAL_AS_ANSWER, "Just some prose, no JSON.", "",
     '{"answered": false, "answer": "", "citations": []}',
     '{"answered": true, "answer": "He uses Python.", "citations": []}'],
)
def test_on_mode_any_invalid_reply_gives_the_standard_refusal_and_no_sources_label(run_app, reply):
    run = run_app(QUESTION, env=ON, reply=reply)
    assert run.drawn("markdown") == [NO_ANSWER]
    assert run.drawn("text") == [] and run.drawn("caption") == []
    saved = run.session["messages"][-1]
    assert saved["content"] == NO_ANSWER and saved["plain"] is False and saved["citations"] == []


@pytest.mark.parametrize("reply", [SURROGATE_ANSWER, SURROGATE_QUOTE])
def test_on_mode_nothing_that_cannot_be_encoded_is_ever_drawn_or_stored(run_app, reply):
    run = run_app(QUESTION, env=ON, reply=reply)
    assert run.drawn("markdown") == [NO_ANSWER]
    for text in run.all_text() + [str(m["content"]) for m in run.session["messages"]]:
        text.encode("utf-8")  # raises UnicodeEncodeError if anything unencodable reached the page or the session


# ------------------------------------------------------------ errors and no key


@pytest.mark.parametrize("env", [KEY, ON])
def test_model_exception_keeps_the_existing_unavailable_message(run_app, env):
    run = run_app(QUESTION, env=env, raises=True)
    assert run.drawn("markdown") == [UNAVAILABLE]
    assert run.drawn("caption") == [] and len(run.model_calls) == 1


@pytest.mark.parametrize("mode", ["off", "on"])
def test_missing_api_key_keeps_retrieval_only_behaviour(run_app, mode):
    run = run_app(QUESTION, env={"GROUNDING_MODE": mode}, reply=VALID)
    assert run.model_calls == [] and run.embed_calls == 0
    assert run.drawn("markdown") == [RETRIEVAL_ONLY]
    assert any(e.kind == "info" and e.text.startswith("No GEMINI_API_KEY found") for e in run.events)
    assert any(e.kind == "expander" and "search by keywords" in e.text for e in run.events)
    assert any(t.startswith("Technical skills\n") for t in run.sources("text"))


@pytest.mark.parametrize("mode", ["off", "on"])
def test_no_matching_passages_gives_the_standard_refusal_without_calling_the_model(run_app, mode):
    run = run_app("What is the favourite football club?", env={"GROUNDING_MODE": mode}, reply=VALID)
    assert run.drawn("markdown") == [NO_ANSWER] and run.model_calls == []


# -------------------------------------------- retrieval settings are untouched


def test_embedding_threshold_setting_is_still_honoured(run_app):
    run = run_app(QUESTION, env={**ON, "EMBED_MIN_SCORE": "1.1"}, reply=VALID)
    assert run.drawn("markdown") == [NO_ANSWER] and run.model_calls == []  # nothing scores above 1.1


def test_embedding_outage_at_startup_still_falls_back_to_keyword_search(run_app):
    run = run_app(QUESTION, env=KEY, reply="ok then", embed_raises=True)
    assert any(e.kind == "expander" and "search by keywords" in e.text for e in run.events)
    assert run.drawn("markdown") == ["ok then"]


def test_embedding_outage_when_a_question_is_asked_still_falls_back_to_keyword_search(run_app):
    # Startup embeds the passages (call 1) fine; embedding the question (call 2) fails.
    run = run_app(QUESTION, env=KEY, reply="ok then", embed_fail_from=2)
    assert run.embed_calls == 2
    assert any(e.kind == "expander" and "search by keywords" in e.text for e in run.events)
    assert run.drawn("markdown") == ["ok then"] and len(run.model_calls) == 1


# ------------------------------------------------------------- old sessions

OLD_MESSAGE = {"role": "assistant", "content": "**old** answer", "mode": "meaning",
               "sources": [{"heading": "Old heading", "score": 0.7, "text": "old passage text"}]}


def test_old_session_message_without_the_newer_fields_renders_as_before(run_app):
    run = run_app(None, session={"messages": [OLD_MESSAGE]})
    assert run.drawn("markdown") == ["**old** answer"]  # old messages stay Markdown
    assert run.drawn("text") == [] and run.drawn("caption") == []
    assert run.sources("markdown") == ["**Old heading** (score 0.70)"]
    assert run.sources("text") == ["old passage text"]
    assert any(e.kind == "expander" and "search by meaning" in e.text for e in run.events)


def test_oldest_style_messages_without_sources_or_mode_still_render(run_app):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "older answer"}]
    run = run_app(None, session={"messages": history})
    assert run.drawn("markdown", ("user",)) == ["hi"]
    assert run.drawn("markdown") == ["older answer"]
    assert not any(e.kind == "expander" for e in run.events)


def test_new_style_grounded_message_replays_as_plain_text_with_its_citations(run_app):
    saved = {"role": "assistant", "content": "He uses **Python**.", "plain": True,
             "citations": ['[1] Technical skills: "Python, SQL and pandas"'], "sources": [], "mode": "meaning"}
    run = run_app(None, session={"messages": [saved]})
    assert run.drawn("text") == ["He uses **Python**.", '[1] Technical skills: "Python, SQL and pandas"']
    assert run.drawn("caption") == [SOURCES_LABEL] and run.drawn("markdown") == []


def test_a_new_question_keeps_old_history_and_appends_one_exchange(run_app):
    run = run_app(QUESTION, env=ON, reply=VALID, session={"messages": [OLD_MESSAGE]})
    messages = run.session["messages"]
    assert messages[0] == OLD_MESSAGE and len(messages) == 3  # old, the question, the new answer
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]


# ------------------------------------------------------------- usage limits


@pytest.mark.parametrize("env", [{**KEY, "MAX_QUESTIONS_PER_SESSION": "1"}, {**ON, "MAX_QUESTIONS_PER_SESSION": "1"}])
def test_session_limit_blocks_with_the_same_message_before_any_retrieval_or_model_call(run_app, env):
    run = run_app(QUESTION, env=env, reply=VALID, session={"asked": 1, "messages": []})
    assert run.stopped
    assert run.drawn("markdown") == [SESSION_LIMIT_1]
    assert run.model_calls == [] and run.embed_calls == 0
    saved = run.session["messages"][-1]
    assert saved == {"role": "assistant", "content": SESSION_LIMIT_1, "sources": [], "mode": "keywords"}


def test_daily_limit_is_shared_across_visitors_and_blocks_with_the_same_message(run_app):
    env = {**ON, "MAX_QUESTIONS_PER_DAY": "1"}
    first = run_app(QUESTION, env=env, reply=VALID)
    assert not first.stopped and first.session["asked"] == 1
    second = run_app(QUESTION, env=env, reply=VALID, cache=first.cache)  # a new visitor, same process
    assert second.stopped and second.drawn("markdown") == [DAILY_LIMIT]
    assert second.model_calls == [] and second.embed_calls == 0


def test_daily_counter_resets_when_the_day_changes(run_app):
    cache = {("daily_counter", (), ()): {"day": "2000-01-01", "count": 5}}
    run = run_app(QUESTION, env={**ON, "MAX_QUESTIONS_PER_DAY": "1"}, reply=VALID, cache=cache)
    assert not run.stopped and run.drawn("text")[0] == "He works with Python and SQL."
    counter = cache[("daily_counter", (), ())]
    assert counter["count"] == 1 and counter["day"] != "2000-01-01"


def test_default_limits_are_15_per_visit_and_300_per_day(run_app):
    run_app(QUESTION, env=KEY, reply="ok")  # runs without limit settings in the environment
    over_session = run_app(QUESTION, env=KEY, reply="ok", session={"asked": 15})
    assert over_session.stopped and "15 questions" in over_session.drawn("markdown")[0]
    under_session = run_app(QUESTION, env=KEY, reply="ok", session={"asked": 14})
    assert not under_session.stopped
    cache = {("daily_counter", (), ()): {"day": __import__("datetime").date.today().isoformat(), "count": 300}}
    assert run_app(QUESTION, env=KEY, reply="ok", cache=cache).stopped
    cache = {("daily_counter", (), ()): {"day": __import__("datetime").date.today().isoformat(), "count": 299}}
    assert not run_app(QUESTION, env=KEY, reply="ok", cache=cache).stopped


# ------------------------------------------------------------ harness hygiene


def test_fakes_and_working_directory_are_restored_after_each_run(run_app, tmp_path):
    cwd_before = Path.cwd()
    run_app(QUESTION, env=KEY, reply="ok")
    assert Path.cwd() == cwd_before
    assert not getattr(sys.modules.get("streamlit"), "_is_fake", False)
    assert not getattr(sys.modules.get("google.genai"), "_is_fake", False)
    import os
    assert "GEMINI_API_KEY" not in os.environ or os.environ["GEMINI_API_KEY"] != KEY["GEMINI_API_KEY"]


def test_the_real_notes_and_embedding_cache_are_never_touched(run_app, tmp_path):
    before = (REPO_CACHE.stat().st_mtime_ns, REPO_CACHE.stat().st_size) if REPO_CACHE.exists() else None
    run_app(QUESTION, env=ON, reply=VALID)
    after = (REPO_CACHE.stat().st_mtime_ns, REPO_CACHE.stat().st_size) if REPO_CACHE.exists() else None
    assert before == after
    assert (tmp_path / "embeddings_cache.json").exists()  # the run wrote its own cache in the temp directory
