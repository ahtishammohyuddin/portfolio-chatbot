"""Offline tests for embed_retriever.py. No API key, no network, no real data.

EmbeddingRetriever takes its embedder as a function, so a fake one with fixed
vectors stands in for Gemini. make_gemini_embedder imports google.genai inside
the function, so a tiny fake `google.genai` is put into sys.modules for those
tests only. The fakes copy the call shape the code uses; they do not prove the
real SDK behaves the same way.
"""

import json
import math
import sys
import types
from pathlib import Path

import pytest

from embed_retriever import (
    DEFAULT_DIMS,
    DEFAULT_MIN_SCORE,
    EmbeddingRetriever,
    _dot,
    _normalise,
    make_gemini_embedder,
)
from rag import Chunk

MARKER = "UNIQUE-SYNTHETIC-MARKER"


class FakeEmbedder:
    """embed(texts, kind) -> vectors from a lookup table; records every call."""

    def __init__(self, table, default=(0.0, 0.0, 1.0)):
        self.table = table
        self.default = list(default)
        self.calls = []  # (kind, texts)

    def __call__(self, texts, kind):
        self.calls.append((kind, list(texts)))
        return [list(self.table.get(t, self.default)) for t in texts]

    def kinds(self):
        return [k for k, _ in self.calls]


def make_chunks():
    return [
        Chunk(f"Alpha\n{MARKER} alpha", "s.md", "Alpha"),
        Chunk("Beta\nbeta text", "s.md", "Beta"),
        Chunk("Gamma\ngamma text", "s.md", "Gamma"),
    ]


def make_table(chunks):
    return {
        chunks[0].text: [1.0, 0.0, 0.0],
        chunks[1].text: [0.0, 1.0, 0.0],
        chunks[2].text: [3.0, 4.0, 0.0],  # deliberately not unit length
        "like alpha": [10.0, 0.0, 0.0],   # deliberately not unit length
        "like beta": [0.0, 2.0, 0.0],
        "unrelated": [0.0, 0.0, 1.0],
    }


@pytest.fixture
def chunks():
    return make_chunks()


@pytest.fixture
def embedder(chunks):
    return FakeEmbedder(make_table(chunks))


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "cache.json"


def build(chunks, embedder, cache_path, **kw):
    return EmbeddingRetriever(chunks, embedder, cache_path=str(cache_path), **kw)


# ------------------------------------------------------------- vector maths


def test_normalise_gives_unit_length():
    v = _normalise([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0)


def test_normalise_zero_vector_does_not_divide_by_zero():
    assert _normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_dot_of_unit_vectors_is_cosine_similarity():
    assert _dot([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _dot([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)


# ------------------------------------------------------------------ search


def test_ranks_by_cosine_similarity_best_first(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path, min_score=0.0)
    results = r.search("like alpha", k=3)
    assert [c.heading for c, _ in results] == ["Alpha", "Gamma", "Beta"]
    assert [s for _, s in results] == pytest.approx([1.0, 0.6, 0.0])


def test_query_length_does_not_change_scores(chunks, embedder, cache_path):
    # "like alpha" is [10,0,0]; the retriever normalises it before comparing.
    r = build(chunks, embedder, cache_path, min_score=0.0)
    assert r.search("like alpha", k=1)[0][1] == pytest.approx(1.0)


def test_k_limits_results(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path, min_score=0.0)
    assert len(r.search("like alpha", k=1)) == 1
    assert len(r.search("like alpha", k=2)) == 2


def test_unrelated_question_returns_nothing_at_the_default_threshold(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path)
    assert DEFAULT_MIN_SCORE == 0.60
    assert r.search("unrelated") == []


def test_threshold_is_inclusive_and_per_call_override_wins(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path, min_score=0.0)
    score = r.search("like alpha", k=3)[1][1]  # Gamma's score (about 0.6)
    assert [c.heading for c, _ in r.search("like alpha", k=3, min_score=score)] == ["Alpha", "Gamma"]
    just_above = math.nextafter(score, 2.0)
    assert [c.heading for c, _ in r.search("like alpha", k=3, min_score=just_above)] == ["Alpha"]


def test_instance_threshold_is_used_when_no_override_is_given(chunks, embedder, cache_path):
    strict = build(chunks, embedder, cache_path, min_score=0.9)
    assert [c.heading for c, _ in strict.search("like alpha", k=3)] == ["Alpha"]
    assert len(strict.search("like alpha", k=3, min_score=0.0)) == 3


def test_documents_and_questions_are_embedded_with_the_right_kind(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path)
    r.search("like alpha")
    assert embedder.kinds() == ["document", "query"]
    assert embedder.calls[1][1] == ["like alpha"]


def test_embedder_failure_at_search_time_propagates(chunks, cache_path):
    # The retriever does not catch it; app.py is what falls back to keywords.
    good = FakeEmbedder(make_table(chunks))
    r = build(chunks, good, cache_path)

    def boom(texts, kind):
        raise RuntimeError("service down")

    r.embed_fn = boom
    with pytest.raises(RuntimeError, match="service down"):
        r.search("like alpha")


def test_no_chunks_means_no_results_and_no_document_call(embedder, cache_path):
    r = build([], embedder, cache_path)
    assert r.search("like alpha") == []
    assert "document" not in embedder.kinds()


# --------------------------------------------------------------------- cache


def test_first_run_embeds_every_chunk_and_writes_the_cache(chunks, embedder, cache_path):
    r = build(chunks, embedder, cache_path)
    assert r.texts_embedded == 3
    assert embedder.calls[0] == ("document", [c.text for c in chunks])
    assert len(json.loads(cache_path.read_text(encoding="utf-8"))) == 3


def test_second_run_uses_the_cache_and_makes_no_document_calls(chunks, embedder, cache_path):
    build(chunks, embedder, cache_path)
    again = FakeEmbedder(make_table(chunks))
    r = build(chunks, again, cache_path)
    assert r.texts_embedded == 0
    assert again.calls == []


def test_changed_chunk_text_is_the_only_one_re_embedded(chunks, embedder, cache_path):
    build(chunks, embedder, cache_path)
    edited = [chunks[0], Chunk("Beta\nbeta text EDITED", "s.md", "Beta"), chunks[2]]
    again = FakeEmbedder(make_table(chunks))
    r = build(edited, again, cache_path)
    assert r.texts_embedded == 1
    assert again.calls == [("document", ["Beta\nbeta text EDITED"])]


def test_stale_cache_entries_are_never_removed(chunks, embedder, cache_path):
    # Documents current behaviour: the cache file only grows.
    build(chunks, embedder, cache_path)
    edited = [chunks[0], Chunk("Beta\nbeta text EDITED", "s.md", "Beta"), chunks[2]]
    build(edited, FakeEmbedder(make_table(chunks)), cache_path)
    assert len(json.loads(cache_path.read_text(encoding="utf-8"))) == 4


def test_changing_model_name_or_dimensions_re_embeds_everything(chunks, embedder, cache_path):
    build(chunks, embedder, cache_path)
    other_model = build(chunks, FakeEmbedder(make_table(chunks)), cache_path, model_name="other-model")
    assert other_model.texts_embedded == 3
    other_dims = build(chunks, FakeEmbedder(make_table(chunks)), cache_path, dims=DEFAULT_DIMS // 2)
    assert other_dims.texts_embedded == 3


def test_cache_holds_hashed_keys_and_unit_vectors_but_no_chunk_text(chunks, embedder, cache_path):
    build(chunks, embedder, cache_path)
    raw = cache_path.read_text(encoding="utf-8")
    assert MARKER not in raw and "beta text" not in raw
    cache = json.loads(raw)
    assert all(len(key) == 64 and set(key) <= set("0123456789abcdef") for key in cache)
    for vec in cache.values():
        assert math.sqrt(sum(x * x for x in vec)) == pytest.approx(1.0)


def test_unreadable_cache_is_treated_as_empty_and_rewritten(chunks, embedder, cache_path):
    cache_path.write_text("{this is not json", encoding="utf-8")
    r = build(chunks, embedder, cache_path)
    assert r.texts_embedded == 3
    assert len(json.loads(cache_path.read_text(encoding="utf-8"))) == 3


def test_cache_write_leaves_no_temporary_file(chunks, embedder, cache_path):
    build(chunks, embedder, cache_path)
    assert not cache_path.with_suffix(".tmp").exists()
    assert sorted(p.name for p in cache_path.parent.iterdir()) == ["cache.json"]


# ------------------------------------------------- make_gemini_embedder (stubbed)


class FakeResponse:
    def __init__(self, vectors):
        self.embeddings = [types.SimpleNamespace(values=v) for v in vectors]


@pytest.fixture
def fake_genai(monkeypatch):
    """Put a fake `google.genai` in sys.modules; yield an object that records calls."""
    rec = types.SimpleNamespace(clients=[], embed_calls=[], reply=None)

    class HttpOptions:
        def __init__(self, timeout=None):
            self.timeout = timeout

    class EmbedContentConfig:
        def __init__(self, task_type=None, output_dimensionality=None):
            self.task_type = task_type
            self.output_dimensionality = output_dimensionality

    class Models:
        def embed_content(self, model, contents, config):
            rec.embed_calls.append(types.SimpleNamespace(model=model, contents=list(contents), config=config))
            if rec.reply is not None:
                return rec.reply(contents)
            return FakeResponse([[float(len(t)), 1.0] for t in contents])

    class Client:
        def __init__(self, api_key=None, http_options=None):
            self.models = Models()
            rec.clients.append(types.SimpleNamespace(api_key=api_key, http_options=http_options))

    genai = types.ModuleType("google.genai")
    genai.Client = Client
    genai_types = types.ModuleType("google.genai.types")
    genai_types.HttpOptions = HttpOptions
    genai_types.EmbedContentConfig = EmbedContentConfig
    genai.types = genai_types
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return rec


def test_embedder_sends_documents_in_batches_of_twenty_and_keeps_order(fake_genai):
    embed = make_gemini_embedder("test-key-not-real")
    texts = [f"text {i}" for i in range(45)]
    vectors = embed(texts, "document")
    assert [len(c.contents) for c in fake_genai.embed_calls] == [20, 20, 5]
    assert len(vectors) == 45
    assert vectors[0][0] == float(len("text 0")) and vectors[44][0] == float(len("text 44"))


def test_embedder_task_type_follows_the_kind_and_settings_are_passed_through(fake_genai):
    embed = make_gemini_embedder("test-key-not-real", model="some-model", dims=256, timeout_ms=1234)
    embed(["a"], "document")
    embed(["b"], "query")
    doc_call, query_call = fake_genai.embed_calls
    assert doc_call.config.task_type == "RETRIEVAL_DOCUMENT"
    assert query_call.config.task_type == "RETRIEVAL_QUERY"
    assert doc_call.model == "some-model" and doc_call.config.output_dimensionality == 256
    assert fake_genai.clients[0].http_options.timeout == 1234


def test_embedder_raises_when_the_vector_count_does_not_match(fake_genai):
    fake_genai.reply = lambda contents: FakeResponse([[1.0, 0.0]])  # one vector for two texts
    embed = make_gemini_embedder("test-key-not-real")
    with pytest.raises(RuntimeError, match="gemini-embedding-001"):
        embed(["one", "two"], "document")
