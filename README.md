---
title: Simple RAG
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
---

# Simple RAG

A learning project to understand and build a Retrieval Augmented Generation (RAG) pipeline from scratch, progressively adding complexity.

## What is RAG?

LLMs like GPT-4 are trained on general data and have a knowledge cutoff. RAG solves two problems:
- Answering questions from **your own documents** the LLM has never seen
- Grounding answers in a specific source so the LLM doesn't hallucinate

Instead of sending all your documents to the LLM every time (expensive, hits token limits), RAG retrieves only the relevant chunks and sends those.

```
Your documents → split into chunks → embed as vectors → store in ChromaDB
                                                               ↓
User query → embed → similarity search → relevant chunks → LLM → answer
```

## How this app works

1. **Documents** — `.txt` files in `docs/` are the knowledge base
2. **Embeddings** — `HuggingFaceEmbeddings` (all-mpnet-base-v2) converts text into 768-dimensional vectors locally, no API needed
3. **ChromaDB** — stores the vectors on disk, loaded on restart without re-embedding
4. **Retrieval** — hybrid search combining BM25 (keyword) + MMR (semantic vector search)
5. **Re-ranking** — FlashrankRerank re-scores retrieved chunks for better relevance
6. **Agentic loop** — LangGraph agent grades chunks, rewrites query and retries if needed
7. **Generation** — relevant chunks passed to `gpt-4o-mini`, answers only from context with source citations

## Project structure

```
simple-rag/
├── agent.py                      # RAG logic — LangGraph agentic pipeline
├── api.py                        # FastAPI — exposes RAG as HTTP endpoints
├── ui.py                         # Streamlit — chat UI, calls FastAPI
├── scraper.py                    # downloads and cleans Wikipedia docs
├── docs/                         # knowledge base (txt files)
│   ├── transformer.txt           # Wikipedia: Transformer architecture
│   ├── large_language_model.txt  # Wikipedia: Large language models
│   └── cognitive_psychology.txt  # Wikipedia: Cognitive psychology
├── chroma_db/                    # auto-generated, persisted vector store
├── requirements.txt
├── .env                          # API keys, never commit
└── .gitignore
```

## Setup

```bash
git clone <repo>
cd simple-rag

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

Download and clean docs:
```bash
python3 scraper.py
```

Run — requires two terminals:
```bash
# Terminal 1 — FastAPI backend
uvicorn api:app --reload

# Terminal 2 — Streamlit UI
streamlit run ui.py
```

First run builds the vector store (~30 seconds). Every run after loads instantly.

Streamlit opens at `http://localhost:8501`. FastAPI docs at `http://localhost:8000/docs`.

To add new documents, drop `.txt` files into `docs/`, delete `chroma_db/`, and rerun.

## Key concepts

| Concept | What it does in this app |
|---|---|
| Embeddings | Converts text to numbers that capture meaning |
| Cosine similarity | Measures how similar two vectors are by angle, not size |
| Chunking | Splits large docs into pieces so retrieval is precise |
| Chunk overlap | Prevents meaning loss at chunk boundaries |
| Persistent ChromaDB | Avoids re-embedding on every restart |
| Score threshold | Filters out irrelevant chunks before sending to LLM |
| Prompt template | Instructs LLM to answer only from provided context |

## Tuning knobs

- `chunk_size` — larger chunks = more context per chunk, but less precise retrieval
- `chunk_overlap` — higher overlap = less meaning lost at boundaries, but more redundancy
- `k` in `similarity_search_with_score(k=3)` — how many chunks to retrieve
- score threshold `1.2` — lower = stricter (fewer results), higher = looser (more results)

## Roadmap

**Level 2 — Real documents**
- [x] Load from txt files
- [x] Text chunking with overlap
- [x] Persistent ChromaDB

**Level 3 — Better retrieval**
- [x] Metadata filtering by filename/topic
- [x] MMR (Maximal Marginal Relevance) — avoid redundant chunks
- [x] Hybrid search — keyword + vector combined

**Level 4 — Better generation**
- [x] Conversation memory across questions
- [x] Source citations in answers
- [x] Streaming responses

**Current: Level 5 — Production patterns**
- [x] Query rewriting
- [x] Re-ranking with a second model
- [x] Agentic RAG
