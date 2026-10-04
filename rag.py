"""
rag.py - the RAG core, with no web UI and no AI call in it.

RAG = Retrieval-Augmented Generation. Three steps:
  1. LOAD + CHUNK : read your documents and cut them into small pieces.
  2. RETRIEVE     : given a question, find the few pieces most relevant to it.
  3. GENERATE     : hand ONLY those pieces + the question to the LLM (see app.py).

This file does steps 1 and 2, plus builds the prompt for step 3.

Step 2 here uses TF-IDF, which matches on shared WORDS. It is free, fast, and
needs no API. Later we will swap it for embeddings, which match on MEANING.
Only the Retriever class changes; everything else stays the same.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str      # what the LLM will read
    source: str    # which file it came from
    heading: str   # which section it came from


# ---------------------------------------------------------------- 1. LOAD + CHUNK

def load_chunks(data_dir: str = "data", max_words: int = 160) -> list[Chunk]:
    """Read every .md/.txt file and split it into chunks, one per '## ' section.

    Why chunk at all? An LLM can only read so much per request, and sending
    everything every time is slow and costly. Small, focused chunks mean we
    send only what is relevant.

    Each chunk keeps its section heading, so a fragment like "Python, SQL,
    pandas" still carries the label "Technical skills".
    """
    chunks: list[Chunk] = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for heading, body in _split_sections(text):
            for piece in _split_long(body, max_words):
                chunks.append(
                    Chunk(text=f"{heading}\n{piece}", source=path.name, heading=heading)
                )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections, heading, lines = [], "Introduction", []
    for line in text.splitlines():
        if line.startswith("## "):
            if lines:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # the document title is not a section
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines).strip()))
    return [(h, b) for h, b in sections if b]


def _split_long(body: str, max_words: int) -> list[str]:
    """If a section is too long, split it on line breaks into smaller pieces."""
    if len(body.split()) <= max_words:
        return [body]
    pieces, current = [], []
    for line in body.splitlines():
        if current and len(" ".join(current + [line]).split()) > max_words:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


# ---------------------------------------------------------------- 2. RETRIEVE

STOP_WORDS = set("""
a about after all also am an and any are as at be because been before being but by
can did do does doing for from had has have having he her here hers him his how i if
in into is it its just me more most my no nor not of on once only or other our out
over own same she should so some such than that the their them then there these they
this those through to too under until up very was we were what when where which while
who whom why will with would you your tell me know
""".split())


def _tokens(text: str) -> list[str]:
    """Lowercase words, minus stop words, plus two-word phrases (bigrams)."""
    words = [w for w in re.findall(r"[a-z0-9+#]+", text.lower()) if w not in STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


class Retriever:
    """Finds the chunks most similar to a question, in plain Python.

    TF-IDF turns each chunk into a table of word weights. A word that is rare
    across your documents (like "RFID") gets a high weight; a common one gets a
    low weight. Cosine similarity then scores how close the question's table is
    to each chunk's table.

    (This is the same maths scikit-learn does. Writing it by hand means there
    is nothing compiled to install, and you can see every step.)
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        counts = [Counter(_tokens(c.text)) for c in chunks]

        # IDF: how rare is each word across all chunks?
        n = len(chunks)
        df = Counter(term for c in counts for term in c)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}

        self.vectors = [self._weigh(c) for c in counts]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        """Turn word counts into normalised TF-IDF weights."""
        vec = {
            t: (1 + math.log(c)) * self.idf[t]
            for t, c in counts.items()
            if t in self.idf  # ignore words never seen in the documents
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, question: str, k: int = 3, min_score: float = 0.08):
        """Return up to k (chunk, score) pairs, best first.

        min_score is a safety net: if nothing is a good match we return an
        empty list, and the bot says "I don't have that" instead of guessing.
        """
        q = self._weigh(Counter(_tokens(question)))
        scores = [
            sum(w * vec.get(t, 0.0) for t, w in q.items()) for vec in self.vectors
        ]
        ranked = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)
        return [(self.chunks[i], float(s)) for i, s in ranked[:k] if s >= min_score]


# ---------------------------------------------------------------- 3. PROMPT

SYSTEM_PROMPT = """You are the portfolio assistant for Ahtisham Mohyuddin. \
You answer questions from recruiters and visitors about his background, \
skills, projects and experience.

Rules:
- Answer ONLY from the CONTEXT provided. Do not use outside knowledge about him.
- If the context does not contain the answer, say you don't have that \
information about Ahtisham and suggest the visitor contact him directly. \
Never guess. Never use the words "context", "provided" or "documents" \
when you say this.
- Missing from the notes is not the same as false. Say you have no \
information about it. Never say that he does not have a skill, \
qualification or experience, because you cannot know that.
- Never invent skills, job titles, dates, employers, numbers or projects.
- Refer to him in the third person ("Ahtisham", "he").
- Keep answers short and plain: two to four sentences, no bullet lists, \
no markdown. They may be read aloud later.
"""


def build_prompt(question: str, results) -> str:
    """Put the retrieved chunks and the question into one prompt."""
    context = "\n\n".join(f"[{i}] {c.text}" for i, (c, _) in enumerate(results, 1))
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


NO_ANSWER = (
    "I don't have that information about Ahtisham. "
    "It's best to contact him directly and ask."
)
