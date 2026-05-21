A Flask-based RAG (Retrieval-Augmented Generation) chatbot for customer support associates at an online bond brokerage platform. It answers bond taxation queries using a Neon Postgres knowledge base and a local AI model (Ollama or LM Studio).

---

## Project Structure

```
rag_chatbot/
├── app.py                 # Main Flask application
├── db.py                  # Neon Postgres search (full-text + ILIKE fallback)
├── llm.py                 # LLM client (Ollama & LM Studio via OpenAI-compatible API)
├── ingest_chunks.py       # One-time script to load markdown chunks into Postgres
├── requirements.txt
├── .env.example           # Copy to .env and fill in your credentials
└── templates/
    └── index.html         # Chat UI
```

---

## Quick Setup

### 1. Install dependencies

```bash
cd rag_chatbot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `DATABASE_URL` — your Neon Postgres connection string
- `OLLAMA_BASE_URL` — usually `http://localhost:11434/v1`
- `LM_STUDIO_BASE_URL` — e.g. `http://192.168.0.100:1234/v1`

### 3. Ingest the knowledge base chunks

Extract the `taxation_rag_markdown_chunks.zip` to a folder, then run:

```bash
python ingest_chunks.py --chunks_dir ./taxation_rag_markdown_chunks
```

This upserts all 20 chunk files into the `rag_chunks` table in Neon Postgres.

### 4. Start the chatbot

```bash
python app.py
```

Visit: http://localhost:1707

---

## How It Works

```
Support Associate types a query
        ↓
Flask /api/chat receives it
        ↓
db.py: Full-text search on Neon Postgres (tsvector)
  → Falls back to ILIKE if no results
  → Returns top 4 most relevant chunks
        ↓
llm.py: Builds a prompt with:
  - System instructions (empathetic support tone)
  - Top chunks as context
  - Conversation history (last 6 turns)
        ↓
Calls selected provider (Ollama or LM Studio)
via OpenAI-compatible /v1/chat/completions API
        ↓
Returns Markdown answer + source chunk titles
        ↓
index.html: Renders Markdown inline with source pills
```

---

## AI Model Selector

The header has a dropdown to switch between:
- **Ollama** — `gpt-oss:120b` (green dot)
- **LM Studio** — `google/gemma-4-e4b` (amber dot)

Both use the OpenAI-compatible `/v1/chat/completions` endpoint.

---

## Neon Postgres Schema

The app uses the `rag_chunks` table with full-text search via a `tsvector` column:
- `search_vector` — auto-generated from `search_text`
- `ts_rank()` used for relevance scoring
- GIN indexes for fast search
