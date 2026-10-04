"""
compare_retrievers.py - measure keyword search (TF-IDF) against embeddings.

Run:  python compare_retrievers.py

It asks both retrievers the same questions and reports how often each finds
the right section. It also tries a range of cut-off scores and tells you which
one to use. That number goes in your README.

Three kinds of question, because they test different things:
  IN_SCOPE   the answer is in the document   -> the right section should be in the top 3
  OFF_TOPIC  nothing to do with the document -> the retriever should return nothing
  NEAR_TOPIC close to the document, but the fact is absent (CGPA, Oracle...).
             Retrieval cannot know a fact is missing; the model's rules handle that.
             Shown for information, not scored.
"""

import time
import tomllib

from embed_retriever import (DEFAULT_DIMS, DEFAULT_MODEL, EmbeddingRetriever,
                             make_gemini_embedder)
from rag import Retriever, load_chunks

# (question, headings that count as a correct hit - any one is enough)
IN_SCOPE = [
    ("Who is Ahtisham?",                          ["Summary"]),
    ("Where did he study?",                       ["Education"]),
    ("What university did he attend?",            ["Education"]),
    ("What databases does he know?",              ["databases"]),
    ("Which programming languages can he use?",   ["languages and frameworks"]),
    ("Tell me about the MERIDIA ERP",             ["MERIDIA"]),
    ("Did he build anything with RFID?",          ["School Bus"]),
    ("What was his final year project?",          ["School Bus"]),
    ("What does he do at DevVibe?",               ["DevVibe"]),
    ("Does he speak Urdu?",                       ["Languages"]),
    ("What certifications does he have?",         ["Certifications"]),
    ("What did he do in IT support in the UAE?",  ["Bait Al Hali"]),
    ("Has he used AI video tools?",               ["AI content and web", "DevVibe"]),
    ("How does he stop the AI making up numbers?", ["MERIDIA"]),
    ("What kind of job is he looking for?",       ["Summary"]),
    ("Has he built a chatbot?",                   ["AI Tools Portfolio"]),
]

OFF_TOPIC = [
    "What is the capital of France?",
    "What is his favourite football team?",
    "Give me a recipe for biryani",
    "Who won the cricket world cup?",
]

NEAR_TOPIC = [
    "What is his CGPA?",
    "Has he worked with Oracle?",
    "What is his salary?",
]

THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(0, 25)]   # 0.30 ... 0.90


def rank_all(retriever, question):
    """Every chunk with its score, best first (no cut-off applied)."""
    return retriever.search(question, k=len(retriever.chunks), min_score=-1.0)


def hit(ranked, expected, threshold, top):
    return any(
        any(e.lower() in c.heading.lower() for e in expected)
        for c, s in ranked[:top] if s >= threshold
    )


def score_at(threshold, in_ranked, off_ranked):
    in_top1 = sum(hit(r, exp, threshold, 1) for r, exp in in_ranked)
    in_top3 = sum(hit(r, exp, threshold, 3) for r, exp in in_ranked)
    rejected = sum(1 for r in off_ranked if not r or r[0][1] < threshold)
    return in_top1, in_top3, rejected


def report(name, retriever, thresholds, fixed=None):
    in_ranked = [(rank_all(retriever, q), exp) for q, exp in IN_SCOPE]
    off_ranked = [rank_all(retriever, q) for q in OFF_TOPIC]
    n_in, n_off = len(IN_SCOPE), len(OFF_TOPIC)

    print(f"\n=== {name} ===")
    print("\nIn-scope questions (top result and its score):")
    for (q, exp), (ranked, _) in zip(IN_SCOPE, in_ranked):
        c, s = ranked[0]
        ok = "ok " if any(e.lower() in c.heading.lower() for e in exp) else "MISS"
        print(f"  {ok} {s:5.2f}  {q:<45} -> {c.heading[:38]}")

    print("\nOff-topic questions (best score; these should fall below the cut-off):")
    for q, ranked in zip(OFF_TOPIC, off_ranked):
        print(f"  {ranked[0][1]:5.2f}  {q}")

    print("\nNear-topic, fact absent (information only):")
    for q in NEAR_TOPIC:
        r = rank_all(retriever, q)
        print(f"  {r[0][1]:5.2f}  {q:<32} -> {r[0][0].heading[:38]}")

    if fixed is not None:
        t1, t3, rej = score_at(fixed, in_ranked, off_ranked)
        print(f"\nAt cut-off {fixed}: top-1 {t1}/{n_in}, in top 3 {t3}/{n_in}, "
              f"off-topic rejected {rej}/{n_off}")
        return fixed, (t1, t3, rej)

    print("\nCut-off sweep   (in top 3 + off-topic rejected = total)")
    best_total, best = -1, []
    for t in thresholds:
        t1, t3, rej = score_at(t, in_ranked, off_ranked)
        total = t3 + rej
        print(f"  {t:5.3f}   top-1 {t1:2d}/{n_in}   top-3 {t3:2d}/{n_in}   "
              f"rejected {rej}/{n_off}   total {total}/{n_in + n_off}")
        if total > best_total:
            best_total, best = total, [t]
        elif total == best_total:
            best.append(t)
    chosen = best[len(best) // 2]
    print(f"\nBest total {best_total}/{n_in + n_off} at cut-offs {best[0]} to {best[-1]}.")
    print(f"Suggested EMBED_MIN_SCORE = {chosen}")
    return chosen, score_at(chosen, in_ranked, off_ranked)


def paced(embed_fn, delay=0.7):
    """Wait a moment before each call, so ~25 quick requests don't trip a
    free-tier rate limit."""
    def inner(texts, kind):
        time.sleep(delay)
        return embed_fn(texts, kind)
    return inner


def main():
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    key = secrets["GEMINI_API_KEY"]
    model = secrets.get("GEMINI_EMBED_MODEL", DEFAULT_MODEL)

    chunks = load_chunks("data")
    print(f"{len(chunks)} chunks. Embedding model: {model}")

    keyword = Retriever(chunks)
    semantic = EmbeddingRetriever(chunks, paced(make_gemini_embedder(key, model)),
                                  model_name=model, dims=DEFAULT_DIMS)
    print(f"Embedded {semantic.texts_embedded} new chunks this run "
          f"(0 means they were already cached).")

    report("KEYWORD SEARCH (TF-IDF)", keyword, THRESHOLDS, fixed=0.08)
    report("EMBEDDINGS (meaning)", semantic, THRESHOLDS)


if __name__ == "__main__":
    main()
