# Running the tests

From the repository root:

```
python -m pytest
```

Only `pytest` is needed. Nothing else is installed for the tests, and they
make no network calls and need no API key. pytest exits with a non-zero code
if any test fails, or if the strict expected failure below starts passing.

On the audit machine the exact command is:

```
cd C:\Users\USER\ChatbotAudit\portfolio-chatbot
C:\Users\USER\ChatbotAudit\.venv\Scripts\python.exe -m pytest
```

## What is covered

| File | Covers | Data used |
|---|---|---|
| `test_retrieval.py` | The eight original retrieval cases against the real notes in `data/` | Real knowledge base, read in-process only. Skipped if `data/` has no notes. Failure messages never print its text. |
| `test_rag_units.py` | Chunking, the keyword retriever, the 0.08 floor, prompt building, system-prompt and refusal-text guards | Small synthetic notes about an invented person |
| `test_embed_retriever.py` | `EmbeddingRetriever` ranking, threshold, cache behaviour; `make_gemini_embedder` batching and error handling | Fake vectors and a fake `google.genai` module |

## "Where did he study?"

The keyword retriever cannot match "study" to "Education". This is expressed as two tests:

- `test_current_keyword_behaviour_study_question_finds_nothing` passes and pins today's behaviour.
- `test_desired_future_study_question_finds_the_education_section` is a **strict expected failure** for the desired behaviour (retrieval by meaning). If it ever passes, the suite fails so the pair can be updated.

## Not covered

- `app.py` (needs Streamlit, and the usage limits and the fallback between retrievers live there, run at import).
- The real Gemini API and the real embeddings, including whether the 0.60 threshold is right.
- Citation checking. The chatbot does not verify citations yet.
- The fakes copy the call shape the code uses. They do not prove the real SDK behaves the same way.
