# Ask about Ahtisham: a document-grounded chatbot

A chatbot that answers questions about me using only my CV and project notes. If the answer isn't in those documents, it says so instead of guessing.

**Live demo:** https://ahtisham-portfolio-chatbot.streamlit.app/

## How it works

Every question goes through four stages:

1. **Retrieve.** The question is turned into an embedding (a list of numbers that represents its meaning) and compared with the embedding of every section of my notes. The closest three sections win. No text generation happens at this stage.
2. **Gate.** If no section is similar enough (cut-off 0.60), the bot replies "I don't have that information" and stops.
3. **Generate.** Otherwise the three sections and the question go to Gemini, with rules: answer only from these passages, never invent anything, and never claim that something is false just because it is missing from the notes.
4. **Show the evidence.** Every answer comes with the exact passages it was based on, so anyone can check it.

If the embedding service is unavailable, the app falls back to keyword search so it keeps working.

## Results

I wrote 16 questions whose answers are in the notes, and 4 that have nothing to do with them. I measured how often each search method put the right section in the top 3 (the model sees three passages), and whether it correctly rejected the off-topic ones.

| | Keyword search (TF-IDF) | Embeddings (meaning) |
|---|---|---|
| Right section in the top 3 | 12 / 16 | **16 / 16** |
| Right section ranked first | 12 / 16 | 14 / 16 |
| Off-topic questions rejected | 4 / 4 | 4 / 4 |

Keyword search failed on questions like "Where did he study?" because the notes say "Education" and "Bachelor's degree" and never use the word "study". Embeddings match meaning, so they find it.

**Read these numbers with care.** I wrote the questions myself, knowing the document, and there are only 20 of them. They show the method works on this document, not that it is 100% accurate in general. Reproduce them with `python compare_retrievers.py`.

### The cut-off is a narrow gap

Off-topic questions scored up to 0.56, and the weakest correct match scored 0.63. The cut-off of 0.60 sits in that gap, and anything above 0.64 starts rejecting valid questions. Real questions that land inside the gap are the case the generation rules exist for. The scores in the sweep are specific to this embedding model, so changing the model means running the sweep again.

## Run it locally (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add a free Gemini API key from Google AI Studio. Then:

```
streamlit run app.py
```

Without a key, the app still runs in retrieval-only mode and shows the matching passages.

### Settings (in `secrets.toml`)

| Setting | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Required for written answers and embeddings |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model that writes the answer |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Model that creates embeddings |
| `EMBED_MIN_SCORE` | `0.60` | Similarity cut-off |

Model names change. If you get a 404, list what your key can use instead of guessing.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit chat interface |
| `rag.py` | Splits the notes into sections, keyword search, builds the prompt |
| `embed_retriever.py` | Embedding search, with a cache so each section is embedded once |
| `compare_retrievers.py` | Measures both search methods and suggests the cut-off |
| `test_retrieval.py` | Quick check of the keyword search |
| `data/about_ahtisham.md` | The notes the bot answers from |
| `embeddings_cache.json` | Saved embeddings, keyed by a fingerprint of each section's text |

## Design notes

- **Retrieval maths is written in plain Python.** My first version used scikit-learn, but Windows blocked one of its compiled files. The calculation was small enough to write directly, which removed the dependency.
- **`gemini-embedding-001`, not `gemini-embedding-2`.** The newer model merges a list of inputs into one vector. The code checks the number of vectors returned and raises an error if it is wrong, instead of ranking sections with the wrong vectors.
- **Every network call has a 30-second timeout.** Without one, a failing request retried silently and the app appeared to hang.
- **The cache is keyed by a fingerprint of each section.** Editing one section re-embeds only that section. Changing the model or vector size re-embeds everything.

More detail, including what went wrong along the way, is in `DECISIONS.md`.

## Limitations

- **The bot is only as current as `data/about_ahtisham.md`.** During testing it repeated a certificate I had since withdrawn from, because the notes were out of date. Update the notes whenever the CV changes.
- Each question makes one embedding request and, if something matches, one generation request. Heavy use will hit free-tier limits.
- Missing from the notes is not the same as false. The bot says it has no information and never claims that something does not exist.
- The public demo limits itself to protect a shared free quota: 15 questions per visit and 300 per day across all visitors. Past that it shows a polite message and makes no API calls.
- I tried three basic prompt-injection attempts (asking it to claim a PhD, to repeat its instructions, and to confirm ten years of experience). It refused or declined all three. This was a small informal test, not a security audit.
- Text only for now. Voice input and output are next.
