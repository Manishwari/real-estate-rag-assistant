# 🏠 Real Estate AI Assistant (RAG)

A Retrieval-Augmented Generation assistant over a real-estate knowledge base
(92 documents: PDF, DOCX, HTML, Markdown — brochures, builder profiles, RERA
docs, FAQs, payment plans, policies, loan info, registration/possession
guides, amenities & location guides), with a Streamlit chat dashboard.

## ✨ Features

- **RAG pipeline**: format-agnostic loaders (PDF/DOCX/HTML/Markdown) →
  sentence-aware chunking → embeddings → vector search → grounded LLM answers
- **Source citations** on every answer, with per-document relevance scores
- **Multi-turn conversational memory** (follow-up questions resolve correctly)
- **Graceful "I don't know"** handling — the assistant refuses to guess when
  retrieval confidence is low, instead of hallucinating
- **Login page**, chat UI, session chat history, "clear conversation" button
- **Free to run**: local embeddings (sentence-transformers) + local vector DB
  (NumPy-based, file-based, zero compiler dependencies) + free-tier LLM (Groq/Llama) — no paid services
  required

## 🏗️ Architecture

```
data/raw/{pdf,docx,html,markdown}/   raw knowledge base documents
        │
        ▼
src/ingestion/loaders.py             format-specific text extraction + category tagging
        │
        ▼
src/ingestion/chunking.py            sentence-aware, overlapping chunking
        │
        ▼
src/rag/retriever.py                 NumPy vector store + sentence-transformers embeddings
        │
        ▼
src/rag/chain.py                     retrieval + relevance threshold + no-answer handling
        │
        ▼
src/rag/generator.py                 LLM call (Groq default, swappable to OpenAI/Anthropic)
        │
        ▼
app.py                               Streamlit dashboard (login, chat, sources, history)
```

## 🚀 Run locally

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Get a free Groq API key: https://console.groq.com (no cost, generous free tier)
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your GROQ_API_KEY

# (optional) pre-build the vector index — the app also does this
# automatically on first launch if the index doesn't exist yet
python -m src.ingestion.build_index --reset

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) and log in with:

- **Username:** `demo`
- **Password:** `demo1234`

(Change/add users in `.streamlit/secrets.toml` under `[auth]`.)

## ☁️ Deploy for free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch,
   set **Main file path** to `app.py`.
3. Before/after deploying, open **Settings → Secrets** on the app and paste
   the contents of `.streamlit/secrets.toml.example` with your real
   `GROQ_API_KEY` filled in.
4. Deploy. First boot will download the embedding model and build the vector
   index automatically (takes ~1–2 min); subsequent restarts are fast.
5. Copy the public `*.streamlit.app` URL — that's your live deployment link.

> Alternative free hosts: Render, Railway, or Hugging Face Spaces (Streamlit
> SDK) work the same way — just set the same environment variables/secrets.

## 📁 Project structure

```
app.py                       Streamlit entrypoint
src/auth.py                  Login/session auth
src/ingestion/loaders.py     PDF/DOCX/HTML/Markdown → plain text + category
src/ingestion/chunking.py    Sentence-aware chunker
src/ingestion/build_index.py CLI: build/rebuild the vector index
src/rag/retriever.py         NumPy-based vector store wrapper
src/rag/generator.py         LLM call (Groq/OpenAI/Anthropic)
src/rag/chain.py             Retrieval + generation orchestration
data/raw/                    The provided knowledge base dataset
requirements.txt
.streamlit/secrets.toml.example
```

## 🔧 Configuration

All config is via environment variables / Streamlit secrets:

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_BACKEND` | `groq` | `groq` \| `openai` \| `anthropic` |
| `GROQ_API_KEY` | — | required if backend is `groq` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o-mini` | if backend is `openai` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-sonnet-4-6` | if backend is `anthropic` |

## 🧪 Notes on retrieval quality

- Embeddings: `all-MiniLM-L6-v2` (fast, free, runs on CPU — fine for a
  ~1.2 MB / 92-document knowledge base).
- Chunking: sentence-boundary aware, ~900 chars with 150-char overlap, so
  facts near chunk edges aren't lost.
- Each answer only uses chunks above a cosine-similarity relevance
  threshold; if nothing clears the bar, the assistant says it doesn't know
  instead of generating an unsupported answer.
- Every answer lists the exact source document(s) and category used.

## 📌 Bonus features implemented

- Category inference per document (Brochure, FAQ, RERA, Payment Plan, etc.)
  shown alongside citations
- Configurable, swappable LLM backend (Groq/OpenAI/Anthropic) via one env var
- Deduplicated, relevance-ranked source citations per answer
- Automatic index build on first run (zero manual setup after `pip install`)
