"""
app.py - the Streamlit chat UI around the RAG core.

Flow for every question:
  question -> retrieve relevant chunks -> build prompt -> Gemini -> answer
Retrieval now works by MEANING (embeddings). If the embedding service is
unavailable, it falls back to keyword search so the bot keeps working.
If nothing matches well, Gemini is not asked to write an answer at all.
"""

import os

import streamlit as st

from embed_retriever import (DEFAULT_DIMS, DEFAULT_MIN_SCORE, DEFAULT_MODEL as EMBED_MODEL,
                             EmbeddingRetriever, make_gemini_embedder)
from rag import NO_ANSWER, SYSTEM_PROMPT, Retriever, build_prompt, load_chunks

# Free text-generation model. Set GEMINI_MODEL in secrets.toml to change it.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

st.set_page_config(page_title="Ask about Ahtisham", page_icon="💬")
st.title("Ask about Ahtisham")
st.caption("An AI assistant that answers only from his CV and project notes.")


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets (deployed) or environment variables (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets file locally; fall through to the environment
    return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
MODEL = get_secret("GEMINI_MODEL", DEFAULT_MODEL)
EMBED = get_secret("GEMINI_EMBED_MODEL", EMBED_MODEL)
EMBED_MIN_SCORE = float(get_secret("EMBED_MIN_SCORE", str(DEFAULT_MIN_SCORE)))


@st.cache_resource
def get_retrievers(api_key: str, embed_model: str, min_score: float):
    """Build both retrievers once. Embeddings are None if they can't be built."""
    chunks = load_chunks("data")
    keyword = Retriever(chunks)
    semantic = None
    if api_key:
        try:
            semantic = EmbeddingRetriever(
                chunks, make_gemini_embedder(api_key, embed_model),
                model_name=embed_model, dims=DEFAULT_DIMS, min_score=min_score,
            )
        except Exception:
            semantic = None  # restart the app after fixing the cause
    return keyword, semantic


def retrieve(question: str):
    """Return (results, mode). Prefer meaning; fall back to keywords."""
    keyword, semantic = get_retrievers(API_KEY, EMBED, EMBED_MIN_SCORE)
    if semantic is not None:
        try:
            return semantic.search(question, k=3), "meaning"
        except Exception:
            pass  # embedding call failed this time; use keywords instead
    return keyword.search(question, k=3), "keywords"


def ask_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=30000),
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,  # low = stick to the facts, don't get creative
        ),
    )
    return (response.text or "").strip()


def show_sources(sources, mode):
    if not sources:
        return
    with st.expander(f"Passages used (search by {mode})"):
        for s in sources:
            st.markdown(f"**{s['heading']}** (score {s['score']:.2f})")
            st.text(s["text"])


if not API_KEY:
    st.info(
        "No GEMINI_API_KEY found, so this is running in retrieval-only mode: "
        "it shows the matching passages instead of a written answer."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"), m.get("mode", "keywords"))

question = st.chat_input("Ask about his skills, projects or experience...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results, mode = retrieve(question)
    sources = [
        {"heading": c.heading, "score": s, "text": c.text} for c, s in results
    ]

    if not results:
        answer = NO_ANSWER
    elif not API_KEY:
        answer = "Closest matches from his notes are shown below."
    else:
        try:
            with st.spinner("Thinking..."):
                answer = ask_gemini(build_prompt(question, results)) or NO_ANSWER
        except Exception:  # rate limit, bad key, network...
            answer = "Sorry, the AI service is unavailable right now. Please try again shortly."

    # Draw the answer now, in this same run.
    with st.chat_message("assistant"):
        st.markdown(answer)
        show_sources(sources, mode)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "mode": mode}
    )
