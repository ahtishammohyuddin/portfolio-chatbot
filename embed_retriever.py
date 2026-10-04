"""
embed_retriever.py - retrieval by MEANING instead of shared words.

The idea:
  1. An EMBEDDING turns a piece of text into a list of numbers (a vector).
     Texts about similar things get similar vectors, even with different words.
     So "study" and "education" end up close together.
  2. We embed every chunk ONCE and save the vectors to a file (the cache).
  3. At question time we embed the question too, then rank the chunks by how
     close their vectors are (cosine similarity). Closest chunks win.

Only this file knows about embeddings. rag.py (chunking, prompt) and the rest
of app.py are unchanged, because EmbeddingRetriever has the same search()
shape as the keyword Retriever.

Written in plain Python on purpose: no numpy, nothing compiled for Windows
to block.
"""

import hashlib
import json
import math
import os
from pathlib import Path

from rag import Chunk

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMS = 768          # the model can return 3072; 768 is smaller and plenty here
CACHE_FILE = "embeddings_cache.json"
DEFAULT_MIN_SCORE = 0.60    # a STARTING POINT - run compare_retrievers.py to calibrate it


# ---------------------------------------------------------------- vector maths

def _normalise(vec: list[float]) -> list[float]:
    """Scale a vector to length 1.

    Once every vector has length 1, cosine similarity is just a dot product.
    (Gemini's docs say to normalise when you ask for fewer than 3072 dims.)
    """
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- the embedder

def make_gemini_embedder(api_key: str, model: str = DEFAULT_MODEL,
                         dims: int = DEFAULT_DIMS, timeout_ms: int = 30000):
    """Return embed(texts, kind) -> list of vectors, using the Gemini API.

    kind is "document" (the chunks) or "query" (the question). The model is
    told which, because a question and the passage that answers it are not
    phrased alike, and the model compensates when it knows the roles.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),  # never hang silently
    )

    def embed(texts: list[str], kind: str) -> list[list[float]]:
        task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 20):          # small batches
            batch = texts[i:i + 20]
            resp = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=dims
                ),
            )
            got = [list(e.values) for e in resp.embeddings]
            if len(got) != len(batch):
                # gemini-embedding-2 merges a list into ONE vector. Catch that
                # loudly instead of silently ranking chunks with the wrong vectors.
                raise RuntimeError(
                    f"Asked for {len(batch)} embeddings but got {len(got)}. "
                    f"Model '{model}' may merge inputs; use gemini-embedding-001."
                )
            vectors.extend(got)
        return vectors

    return embed


# ---------------------------------------------------------------- the retriever

class EmbeddingRetriever:
    """Finds the chunks whose MEANING is closest to a question."""

    def __init__(self, chunks: list[Chunk], embed_fn, model_name: str = DEFAULT_MODEL,
                 dims: int = DEFAULT_DIMS, cache_path: str = CACHE_FILE,
                 min_score: float = DEFAULT_MIN_SCORE):
        self.chunks = chunks
        self.embed_fn = embed_fn
        self.model_name = model_name
        self.dims = dims
        self.cache_path = Path(cache_path)
        self.min_score = min_score
        self.texts_embedded = 0     # how many chunks we had to pay to embed this run
        self.vectors = self._load_or_embed()

    # -- cache: embed each chunk once, ever (until its text changes) ----------

    def _key(self, text: str) -> str:
        """A fingerprint of (model, size, text). Change any of the three and the
        old vector no longer applies, so the chunk gets re-embedded."""
        raw = f"{self.model_name}|{self.dims}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_cache(self, cache: dict) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, self.cache_path)        # swap in one step, so no half-written file

    def _load_or_embed(self) -> list[list[float]]:
        cache = self._read_cache()
        keys = [self._key(c.text) for c in self.chunks]
        missing = [i for i, k in enumerate(keys) if k not in cache]

        if missing:
            new = self.embed_fn([self.chunks[i].text for i in missing], "document")
            for i, vec in zip(missing, new):
                cache[keys[i]] = _normalise(vec)
            self.texts_embedded = len(missing)
            self._write_cache(cache)

        return [cache[k] for k in keys]

    # -- search ---------------------------------------------------------------

    def search(self, question: str, k: int = 3, min_score: float | None = None):
        """Return up to k (chunk, score) pairs, best first.

        Scores are cosine similarity: 1.0 = same meaning, near 0 = unrelated.
        Same safety net as before: below min_score we return nothing.
        """
        threshold = self.min_score if min_score is None else min_score
        q = _normalise(self.embed_fn([question], "query")[0])
        scored = [(chunk, _dot(q, vec)) for chunk, vec in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s >= threshold]
