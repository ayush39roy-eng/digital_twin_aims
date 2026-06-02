# Karpathy Digital Twin

A RAG-powered digital twin of Andrej Karpathy built on his blogs, GitHub READMEs, and Lex Fridman interview transcripts. Uses Gemini 2.5 Flash for generation and ChromaDB + sentence-transformers for retrieval.

## Setup

```bash
git clone <repo>
cd karpathy-twin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_key_here
```

## Data collection

Run the scrape scripts in any order:
```bash
python scripts/scrape_blogs.py
python scripts/scrape_github.py
python scripts/scrape_lex.py
```

Then ingest into ChromaDB:
```bash
python scripts/ingest.py
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Notes

- The `data/` and `chroma_db/` directories are gitignored — you must run the scrape + ingest steps locally.
- To re-ingest from scratch, delete `chroma_db/` and re-run `ingest.py`.
- Long-term memory is stored in `memory.db` (SQLite) and persists across server restarts.
