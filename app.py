"""
PS36 - Research Paper Q&A  (Streamlit UI)
Run:  streamlit run app.py
"""
import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import (
    PRESETS,
    PROMPTS,
    Gemini,
    PaperQA,
    build_prompt,
    friendly_error,
    load_pdf,
    retrieval_query,
    wants_overview,
)

load_dotenv()
CACHE_DIR = Path(__file__).parent / ".cache"
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]

st.set_page_config(page_title="Research Paper Q&A", page_icon="📄", layout="wide")
for key, default in {"messages": [], "qa": None, "paper_hash": None, "pending": None}.items():
    st.session_state.setdefault(key, default)


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources used ({len(sources)} excerpts)"):
        for s in sorted(sources, key=lambda s: s["page"]):
            score = "intro / abstract" if s["score"] is None else f"similarity {s['score']:.2f}"
            st.markdown(f"**Page {s['page']}** · {score}")
            st.caption(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))


def transcript(paper_title: str, messages: list[dict]) -> str:
    out = [f"# Q&A — {paper_title}\n"]
    for m in messages:
        out.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    typed_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Loaded from .env" if env_key else "Paste your key",
        help="Free key: https://aistudio.google.com/apikey",
    )
    api_key = typed_key.strip() or env_key

    default_model = os.getenv("GEMINI_MODEL", MODELS[0])
    models = MODELS if default_model in MODELS else [default_model, *MODELS]
    model = st.selectbox("Model", models, index=models.index(default_model))
    top_k = st.slider("Excerpts per answer (top-k)", 3, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    prompt_name = st.selectbox("System prompt", list(PROMPTS), help="Switch to v1 to compare against the baseline prompt.")

    st.divider()
    uploaded = st.file_uploader("Research paper (PDF)", type=["pdf"])
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

# --------------------------------------------------------------------------- main
st.title("📄 Research Paper Q&A")
st.caption("Upload a paper and ask questions. Answers are grounded in the paper's text, with page citations.")

if not api_key:
    st.info("Add your Gemini API key in the sidebar or in `.env` to start — free at https://aistudio.google.com/apikey")
    st.stop()
if uploaded is None:
    st.info("⬅️ Upload a research paper (PDF) from the sidebar.")
    st.stop()

gemini = Gemini(api_key)

data = uploaded.getvalue()
if st.session_state.paper_hash != hashlib.sha256(data).hexdigest():
    try:
        paper = load_pdf(data, uploaded.name)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, text=f"Indexing {len(paper.chunks)} chunks…")
    try:
        qa = PaperQA.build(
            paper, gemini, cache_dir=CACHE_DIR,
            on_progress=lambda done, total: bar.progress(done / total, text=f"Embedding chunks… {done}/{total}"),
        )
    except Exception as e:
        bar.empty()
        st.error(friendly_error(e))
        st.stop()
    bar.empty()
    st.session_state.update(qa=qa, paper_hash=paper.sha256, messages=[], pending=None)

qa: PaperQA = st.session_state.qa
paper = qa.paper

c1, c2, c3 = st.columns([4, 1, 1])
c1.markdown(f"**{paper.title}**  \n`{paper.name}`")
c2.metric("Pages", paper.n_pages)
c3.metric("Chunks indexed", len(paper.chunks))

st.markdown("**Quick questions**")
for col, (label, (preset_q, preset_intro)) in zip(st.columns(len(PRESETS)), PRESETS.items()):
    if col.button(label, key=f"preset_{label}"):
        st.session_state.pending = (preset_q, preset_intro)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

typed = st.chat_input("Ask anything about the paper…")
question, include_intro = None, False
if typed and typed.strip():
    question = typed.strip()
elif st.session_state.pending:
    question, include_intro = st.session_state.pending
    st.session_state.pending = None

if question:
    history = list(st.session_state.messages)
    include_intro = include_intro or wants_overview(question)
    sources: list[dict] = []
    error = False

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the paper…"):
                hits = qa.retrieve(retrieval_query(question, history), gemini, k=top_k, include_intro=include_intro)
            prompt = build_prompt(paper.title, hits, question, history)
            answer = st.write_stream(gemini.stream(PROMPTS[prompt_name], prompt, model, temperature))
            if not isinstance(answer, str) or not answer.strip():
                answer = "_The model returned an empty response — try rephrasing the question._"
                st.markdown(answer)
            sources = [{"page": h.chunk.page, "score": h.score, "text": h.chunk.text} for h in hits]
            show_sources(sources)
        except Exception as e:
            answer, error = f"⚠️ {friendly_error(e)}", True
            st.error(answer)

    st.session_state.messages += [
        {"role": "user", "content": question, "error": error},
        {"role": "assistant", "content": answer, "sources": sources, "error": error},
    ]

if st.session_state.messages:
    st.sidebar.download_button(
        "⬇️ Download chat (.md)",
        data=transcript(paper.title, st.session_state.messages),
        file_name=f"qa_{Path(paper.name).stem}.md",
        mime="text/markdown",
    )
