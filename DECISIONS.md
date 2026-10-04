# Decisions

Why this project is built the way it is, including what went wrong. Each entry says what happened, what I did, and why.

## Why the retriever is hand-written

**What happened.** My first version used scikit-learn for the keyword search. On my Windows laptop the import failed with "An Application Control policy has blocked this file". scikit-learn needs scipy, and one compiled file inside scipy was blocked because Windows could not verify who published it. Reinstalling did not help.

**What I did.** The calculation I needed was small, so I wrote it in plain Python: split the text into words, drop very common ones, weight rare words higher (TF-IDF), and compare with cosine similarity. About 40 lines.

**Why.** Fewer dependencies, nothing compiled for Windows to block, and I can explain every step. The embedding search is plain Python for the same reason.

## Why every network call has a 30-second timeout

**What happened.** I asked for a model name that did not exist. Google answered "404 not found" straight away, but the client library kept retrying, so the app showed "Thinking..." for over five minutes. A failure looked exactly like slowness.

**What I did.** I set a 30-second timeout on every client, for both text generation and embeddings. The app now shows a clear error instead of hanging.

**Why.** Without a timeout, you cannot tell "slow" from "will never work".

## Why I list models instead of guessing names

**What happened.** The documentation page showed `gemini-3.5-flash-lite`, but I copied the hyphenated form from the URL and wrote `gemini-3-5-flash-lite`. That returned a 404. A separate mistake made it worse: the `GEMINI_MODEL` line in my settings file started with `#`, which turns it into a comment, so the app silently ignored it.

**What I did.** I asked the API which models my key can use (`client.models.list()`), which showed the exact name with a dot. I did the same for embeddings by listing the models that support `embedContent`.

**Why.** Documentation drifts. The API always knows what exists for my key.

## Why `gemini-embedding-001`, not `gemini-embedding-2`

**What happened.** Both were available. The newer one is multimodal and, according to Google's documentation, produces a single combined embedding when you pass it a list of inputs.

**What I did.** I used `gemini-embedding-001`, which is text-only, returns one vector per string, and supports task types (`RETRIEVAL_DOCUMENT` for the notes, `RETRIEVAL_QUERY` for the question). The code also checks that the number of vectors returned equals the number of texts sent, and raises an error if not.

**Why.** My data is text, and quietly ranking sections with the wrong vectors would be hard to notice. I also request 768 dimensions instead of 3072 and normalise the vectors myself, which keeps the cache small.

## Why the cut-off is 0.60

**What happened.** Keyword scores and embedding scores are on different scales. Unrelated text still scores around 0.5 with embeddings, so a cut-off picked from the old numbers would have been wrong.

**What I did.** I wrote 16 questions whose answers are in my notes and 4 off-topic ones, then tried cut-offs from 0.30 to 0.90. Every cut-off from 0.575 to 0.625 scored 20/20. I chose 0.60, the middle.

**Why, and the catch.** Off-topic questions scored up to 0.56 and the weakest correct match scored 0.63, so the gap is narrow. Above about 0.64 the bot starts rejecting valid questions. Questions about facts that are missing but close to the topic (CGPA, Oracle) score above the cut-off, so the model's rules have to handle those. I wrote the questions myself, so this is a check that the method works on this document, not a measure of accuracy in general. I need to re-run `compare_retrievers.py` if I change the embedding model or the notes.

## The stale-fact bug

**What happened.** Once embeddings worked, I asked "Where did he study?" and the bot said I was currently pursuing a certificate that I had withdrawn from. Another outdated certificate was also still in the notes.

**What I did.** I removed both lines from `data/about_ahtisham.md`. Only the two changed sections were re-embedded, because the cache stores each section under a fingerprint of its text.

**Why it matters.** The bot is only as accurate as its source. It cannot know a fact has changed. I now update the notes whenever my CV changes, and I read the bot's own answers as a test.

## Why the bot says "no information" and never "he doesn't have"

**What happened.** Asked about a certificate that was not in my notes, the bot said I did not have it. But missing from the notes is not the same as false.

**What I did.** I added a rule to the system prompt: say there is no information, and never state that he lacks a skill, qualification or experience, because the bot cannot know that. The same rule bans phrases like "the provided context", so refusals do not expose how the system works.

**Why.** This bot speaks about a real person. A confident false statement is worse than an honest "I don't know".

## Why the app limits questions

**What happened.** The public demo shares one free API quota between every visitor.

**What I did.** Two caps: 15 questions per visit, and 300 per day across everyone. A blocked question shows a polite message and makes no API calls. Both numbers are settings, not constants in the code.

**Trade-offs.** The per-visit cap resets if someone reloads the page, so the daily cap is the real protection. The counter lives in memory, so it also resets when the app restarts. This protects a free quota. It is not a security control, and a few heavy users could use up the daily allowance for everyone else. That is acceptable for a demo.
