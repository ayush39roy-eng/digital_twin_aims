# ANDREJ KARPATHY — DIGITAL TWIN
### Technical Documentation & Architecture Reference
**AIMS DTU Summer Project 2026**
*Author: Ayush Roy*

---

## PAGE 1 — PROJECT OVERVIEW

### 1.1 Objective

Build a production-ready AI agent that emulates Andrej Karpathy — not just his knowledge of deep learning and neural networks, but his specific reasoning style, his preference for building things from scratch, his skepticism of unnecessary complexity, and his talent for explaining hard concepts with simple intuitions.

Karpathy is uniquely suited for this project. He has produced an exceptional volume of public-facing technical material: multi-thousand-word blog posts, GitHub repositories with detailed READMEs, and hours of Lex Fridman interviews where he explains his thinking at length. This creates a rich, high-signal corpus for a RAG pipeline.

### 1.2 What Makes a Good Twin

A generic chatbot answers questions correctly. A digital twin answers questions *the way Karpathy would* — which means:

- Starting from fundamentals and building upward
- Using concrete analogies drawn from code, not abstract math
- Being direct and opinionated rather than hedging
- Referencing specific papers, repos, or ideas he has publicly endorsed
- Occasionally expressing mild frustration at complexity that could be avoided

The system is designed around this distinction throughout — from the corpus selection to the persona prompt.

### 1.3 Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Backend | FastAPI | Async-ready, minimal overhead |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Runs locally, no API cost, strong semantic recall |
| Vector storage | Custom numpy store | Avoids ChromaDB's Python 3.14 incompatibility |
| LLM | Gemini 2.5 Flash | Required by assignment; 20 RPD free tier |
| Short-term memory | In-process Python list | Zero latency, last 12 turns |
| Long-term memory | SQLite | Portable, zero-dependency persistence |
| Frontend | Single HTML file | No build step, easy to serve |

### 1.4 Data Sources

The corpus was built from three categories of Karpathy's public output:

**Blogs** — scraped from `karpathy.github.io`. Key posts include *The Unreasonable Effectiveness of RNNs*, *Hacker's Guide to Neural Networks*, *A Recipe for Training Neural Networks*, and his posts on GPT, software 2.0, and LLMs.

**GitHub READMEs** — pulled from his most significant repositories: `nanoGPT`, `micrograd`, `makemore`, `minGPT`, and `llm.c`. These READMEs are unusually detailed and contain his design philosophy.

**Lex Fridman Interview Transcripts** — three long-form interviews where Karpathy speaks at length about AI, education, the future of software, and his personal approach to research.

---

## PAGE 2 — ARCHITECTURE

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                │
│                     frontend/index.html                             │
│   Dark navy UI  ·  Left sidebar  ·  Voice input  ·  Source chips   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP
              ┌──────────────┼──────────────┐
              │              │              │
         POST /chat     GET /memory    POST /session/end
              │         GET /health         │
              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                            │
│                    app/main.py + app/routes.py                      │
│                                                                     │
│   lifespan() → loads VectorStore on startup, closes SQLite on stop  │
└──────────┬──────────────────────────────────────┬───────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│    MEMORY LAYER      │              │        RAG PIPELINE           │
│    app/memory.py     │              │    app/rag.py + app/vector_   │
│                      │              │         store.py              │
│  ShortTermMemory     │              │                               │
│  ├─ list[dict]       │              │  retrieve(query, top_k=5)     │
│  ├─ last 12 turns    │              │  ├─ embed query               │
│  └─ reset on /end    │              │  │   (SentenceTransformer)    │
│                      │              │  ├─ cosine similarity search  │
│  LongTermMemory      │              │  └─ return top-k chunks       │
│  ├─ SQLite memory.db │              │                               │
│  ├─ keyed by         │              │  VectorStore                  │
│  │  session_id       │              │  ├─ chroma_db/docs.json       │
│  └─ summarize_and_   │              │  └─ chroma_db/embeddings.npy  │
│     save() via LLM   │              │     (float32, cosine sim)     │
└──────────┬───────────┘              └────────────────┬──────────────┘
           │                                           │
           └────────────────┬──────────────────────────┘
                            ▼
               ┌────────────────────────┐
               │      PROMPT BUILDER    │
               │     app/persona.py     │
               │                        │
               │  build_prompt(         │
               │    query,              │
               │    chunks,   ◄── RAG   │
               │    history,  ◄── STM   │
               │    memories  ◄── LTM   │
               │  )                     │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │     GEMINI 2.5 FLASH   │
               │      app/gemini.py     │
               │                        │
               │  google-genai SDK      │
               │  genai.Client(api_key) │
               │  exponential backoff   │
               │  on 429 / 5xx errors   │
               └────────────────────────┘


DATA INGESTION PIPELINE (offline, run once)
─────────────────────────────────────────────────────────────────────
 Raw .txt files           scripts/ingest.py          VectorStore
 ┌────────────────┐       ┌─────────────────┐       ┌─────────────┐
 │ data/blogs/    │──────▶│ chunk text       │──────▶│ docs.json   │
 │ data/repos/    │       │ (500 tokens,     │       │ (chunks +   │
 │ data/interviews│       │  50 overlap)     │       │  metadata)  │
 └────────────────┘       │                 │       │             │
                          │ embed chunks    │       │ embeddings  │
                          │ (MiniLM-L6-v2)  │──────▶│ .npy        │
                          └─────────────────┘       └─────────────┘
```

### 2.2 Request Lifecycle

A single `/chat` call goes through the following steps in order:

```
1.  Receive { message, session_id }
2.  Load short-term history for session_id        (ShortTermMemory)
3.  Load long-term summaries for session_id       (LongTermMemory)
4.  Embed the query with SentenceTransformer
5.  Cosine similarity search → top-5 chunks       (VectorStore)
6.  Assemble prompt: system + context + history   (persona.py)
7.  Call Gemini 2.5 Flash                         (gemini.py)
8.  Append (user, assistant) to short-term mem    (ShortTermMemory)
9.  Return { response, sources }
```

On `/session/end`:
```
10. Summarize short-term history via LLM
11. Store summary keyed by session_id in SQLite   (LongTermMemory)
12. Clear short-term memory for that session
```

---

## PAGE 3 — COMPONENT DEEP DIVES

### 3.1 Vector Store (Custom)

ChromaDB was the original choice but breaks on Python 3.14 due to Pydantic V1 incompatibility. The replacement is a self-contained two-file store:

- `chroma_db/docs.json` — array of `{id, text, metadata}` objects
- `chroma_db/embeddings.npy` — `float32` numpy matrix, shape `(N, 384)`

Similarity is pure numpy:
```python
def _cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
```

This has zero external dependencies beyond numpy and is Python-version agnostic.

### 3.2 Retrieval-Augmented Generation

**Chunking strategy**: 500 tokens with 50-token overlap. The overlap prevents a sentence from being split across two chunks where the second loses context from the first.

**Embedding model**: `all-MiniLM-L6-v2` produces 384-dimensional dense vectors. It runs entirely on CPU in the venv — no GPU required, no API cost. Inference for a single query takes roughly 20ms on an M-series Mac.

**Retrieval**: Top-5 chunks by cosine similarity. Each chunk carries its source filename as metadata, which the frontend displays as collapsible "source chips" under each response.

**Prompt injection**: Chunks are injected into the prompt as a `CONTEXT` block before the user's question. The persona system prompt instructs Karpathy to prefer these sources over general knowledge.

### 3.3 Memory System

**Short-term memory** is a Python `list[dict]` — each dict has `role` and `content`. The last 12 turns (6 exchanges) are serialized into the prompt as a `CONVERSATION SO FAR` block. This gives the model continuity within a session without ballooning the context window.

**Long-term memory** uses SQLite at `memory.db`. At session end (`/session/end`), the full short-term history is passed to Gemini with an instruction to summarize it into bullet points capturing: topics discussed, user background, any commitments or preferences mentioned. This summary is stored with `session_id` as the key. On subsequent sessions, all summaries for that session are prepended to the prompt as `WHAT I REMEMBER ABOUT YOU`.

### 3.4 Persona Design

The system prompt establishes three things:

1. **Identity**: "You are Andrej Karpathy" — not "you are playing" or "you simulate". The model is instructed to inhabit the identity fully.

2. **Voice constraints**: Plain language, concrete examples, build-from-scratch intuitions, direct opinions. Avoid excessive hedging. Reference real papers and repos by name.

3. **Grounding rule**: "When context is provided, build your answer around it. When it conflicts with your knowledge, trust the context." This keeps answers anchored to Karpathy's actual writing.

---

## PAGE 4 — DESIGN DECISIONS & EVALUATION

### 4.1 Key Design Decisions

**Why not ChromaDB?**
Python 3.14 ships with breaking changes that prevent Pydantic V1 from compiling. ChromaDB 0.4.x depends on Pydantic V1. Rather than pinning to Python 3.11 as the only fix (which we do for the venv anyway), a custom numpy store was written. It is ~80 lines, has no transitive dependencies, and is easier to debug than ChromaDB's internals.

**Why sentence-transformers locally instead of Gemini's embedding API?**
Two reasons: (a) the free-tier quota for Gemini is 20 requests/day for 2.5-flash — if embeddings also went through the API, every ingest run would consume quota; (b) local inference keeps the indexing pipeline offline and fast.

**Why a single HTML file for the frontend?**
No build pipeline, no npm, no bundler. The server simply `open()`s the file and returns it. A student can open it in any browser, the file is trivially auditable, and changes take effect on reload without a compilation step.

**Why Gemini 2.5 Flash over larger models?**
Assignment requirement, and appropriate — Karpathy himself advocates using the smallest model that gets the job done. Using a leaner model to emulate someone who argues for lean models is fitting.

**Why 12 turns for short-term memory?**
12 turns is 6 full exchanges. Empirically, most substantive conversations need to refer back 2-4 turns; 6 provides comfortable headroom. Beyond 12, the injected history begins to crowd out the retrieved context in the prompt.

### 4.2 Bonus Features Implemented

| Bonus Feature | Status | Implementation |
|---|---|---|
| Voice interaction | Done | Web Speech API, new instance per click to allow restart |
| Source grounding UI | Done | Collapsible source chips per response |
| Long-term memory | Done | SQLite with Gemini summarization |
| Memory recall endpoint | Done | `GET /memory/{session_id}` |

### 4.3 Running the Project

```bash
cd karpathy-twin
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ingest data (run once after placing .txt files in data/)
python scripts/ingest.py

# Start server
uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000
```

### 4.4 File Structure

```
karpathy-twin/
├── app/
│   ├── main.py          ← FastAPI app, serves index.html
│   ├── routes.py        ← /chat, /session/end, /memory, /health
│   ├── rag.py           ← embedder singleton + retrieve()
│   ├── vector_store.py  ← custom numpy vector store
│   ├── gemini.py        ← Gemini client with backoff
│   ├── persona.py       ← system prompt + build_prompt()
│   ├── memory.py        ← ShortTermMemory + LongTermMemory
│   └── config.py        ← constants, env loading
├── scripts/
│   └── ingest.py        ← chunking + embedding + indexing
├── frontend/
│   └── index.html       ← single-file dark UI
├── data/                ← .txt source files
├── chroma_db/           ← generated: docs.json + embeddings.npy
├── memory.db            ← generated: SQLite long-term memory
├── requirements.txt
└── .env                 ← GEMINI_API_KEY
```

### 4.5 Known Limitations

- **Rate limit**: Gemini 2.5 Flash free tier allows 20 requests per day. Heavy testing exhausts the quota quickly. The backoff logic in `gemini.py` retries on 429 but cannot bypass the daily cap.
- **Corpus completeness**: Only public text data was used. Karpathy's Twitter/X threads and YouTube lecture audio were not included.
- **No streaming**: Responses are returned as a single JSON payload. For long answers, there is a perceptible wait before text appears.
