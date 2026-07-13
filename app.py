"""
Real Estate AI Assistant — Streamlit dashboard.

Run locally:
    streamlit run app.py

First run will build the vector index automatically if it doesn't exist yet.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.auth import login_view, logout_button
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_all_documents
from src.rag.chain import RAGChain
from src.rag.retriever import Retriever

RAW_DIR = Path("data/raw")
PERSIST_DIR = Path("chroma_db")

st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Global styling — keep it clean, modern, and simple per the spec
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #f7f8fa; }
        [data-testid="stChatMessage"] { border-radius: 14px; }
        .source-pill {
            display:inline-block; background:#eef2ff; color:#4338ca;
            border-radius:999px; padding:2px 10px; font-size:12px;
            margin:2px 6px 2px 0; border:1px solid #e0e7ff;
        }
        .app-header { text-align:center; padding: 6px 0 18px 0; }
        .app-header h1 { margin-bottom:0; }
        .app-header p { color:#6b7280; margin-top:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached, expensive resources: build/load the index once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_rag_chain() -> RAGChain:
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GENERATOR_BACKEND"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

    retriever = Retriever(persist_dir=PERSIST_DIR)
    if retriever.is_empty():
        docs = load_all_documents(RAW_DIR)
        chunks = chunk_documents(docs)
        retriever.add_chunks(chunks)
    return RAGChain(retriever)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for s in sources:
            st.markdown(
                f"<span class='source-pill'>{s['category']}</span>"
                f"**{s['source']}** &nbsp;·&nbsp; relevance {s['relevance_score']:.2f}",
                unsafe_allow_html=True,
            )


def chat_view() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🏠 Real Estate AI Assistant</h1>
            <p>Ask about projects, pricing, floor plans, RERA, policies & more</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### 👋 Hi, {st.session_state.get('username', 'guest')}")
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        logout_button()
        st.divider()
        st.caption(
            "Knowledge base covers brochures, pricing, RERA docs, FAQs, "
            "policies, loan info, registration & possession guides for "
            "3 builders / 6 projects."
        )
        with st.expander("💡 Try asking"):
            st.markdown(
                "- What projects does Skyline Horizon Developers offer?\n"
                "- What's the payment plan for Horizon Business Park?\n"
                "- What documents do I need to register a unit?\n"
                "- What is your cancellation & refund policy?\n"
                "- What's the possession date for [project], and what "
                "amenities does it have?"
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    question = st.chat_input("Ask about a project, policy, price, or process...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                try:
                    chain = get_rag_chain()
                    # build plain-role history for the generator (exclude sources)
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = chain.answer(question, history=history)
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "answer": (
                            "Something went wrong while generating a response. "
                            f"Details: `{exc}`. If this is an API key error, check "
                            "that GROQ_API_KEY (or your chosen provider's key) is "
                            "set in Streamlit secrets."
                        ),
                        "sources": [],
                        "grounded": False,
                    }
            st.markdown(result["answer"])
            render_sources(result["sources"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


def main() -> None:
    if not login_view():
        return
    chat_view()


if __name__ == "__main__":
    main()
