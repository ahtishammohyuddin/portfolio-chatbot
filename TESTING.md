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
| `test_grounding.py` | Citation validation and answer orchestration in `grounding.py`: JSON parsing, citation references, quote matching (one line, whole words, meaningful words, passage wording shown), refusals, `GROUNDING_MODE`, the off and on answer paths with a stubbed model, prompt-building errors | Synthetic notes only; stubbed model function |
| `test_app_wiring.py` | `app.py` end to end under a fake Streamlit and a fake Gemini: mode off keeps the original prompt and Markdown, mode on shows plain text with the "Quotes found in his notes:" label, invalid replies, model errors, no key, old session messages, usage limits, embedding fallback | Synthetic notes in a temporary directory; fakes restored after every run; sockets blocked |
| `test_embed_retriever.py` | `EmbeddingRetriever` ranking, threshold, cache behaviour; `make_gemini_embedder` batching and error handling | Fake vectors and a fake `google.genai` module |

## "Where did he study?"

The keyword retriever cannot match "study" to "Education". This is expressed as two tests:

- `test_current_keyword_behaviour_study_question_finds_nothing` passes and pins today's behaviour.
- `test_desired_future_study_question_finds_the_education_section` is a **strict expected failure** for the desired behaviour (retrieval by meaning). If it ever passes, the suite fails so the pair can be updated.

## Not covered

- Real Streamlit. `test_app_wiring.py` runs `app.py` under a fake `streamlit` module, so layout, wrapping of `st.text` and Streamlit's own caching and session behaviour are not tested.
- The real Gemini API and the real embeddings, including whether the 0.60 threshold is right.
- Whether a real Gemini model follows the JSON and quoting rules. Only a live test with an API key can show that; `GROUNDING_MODE` is `off` by default until then.
- Fact verification. Citation validation proves each quote exists word for word in a passage retrieved for that question. It does not prove the answer follows from the quotes, and it does not stop prompt injection.
- Quote matching policy (deterministic, see `grounding.py`): a quote must lie inside ONE line of the passage, so list items, wrapped lines and paragraphs are never merged (a quote that crosses a line wrap is refused); it must start and end on whole words (letter or digit edges; punctuation edges and hyphens are not checked); it needs at least two words of two or more letters or digits and 12 normalised characters; visitors see the passage's own wording, not the model's. Text written without spaces (such as Chinese) has no words by this rule and is refused.
- Invalid Unicode. A reply whose answer or any citation quote cannot be encoded as strict UTF-8 (for example a lone UTF-16 surrogate arriving as a JSON escape) is refused with the standard refusal, whatever else is valid and whether or not strict mode is on, so nothing that could break the page is drawn or kept in the session. Valid non-ASCII text and emoji are unaffected. This check applies to grounding mode only; the original (off) path is unchanged.
- Refusal wording. An answer that opens with refusal wording ("I don't have that information", "The notes do not", and similar fixed English phrases) is refused even with a valid quote. Only the opening is checked: a refusal worded differently, or placed after a factual first sentence, is not detected.
- The fakes copy the call shape the code uses. They do not prove the real SDK behaves the same way.
