import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from app.vector_store import VectorStore
from app.config import CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

WORDS_PER_CHUNK = int(CHUNK_SIZE * 0.75)
WORDS_OVERLAP = int(CHUNK_OVERLAP * 0.75)


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        start += chunk_words - overlap_words
    return chunks


def main():
    store = VectorStore(CHROMA_PATH)

    if store.count() > 0:
        print(f"Vector store already has {store.count()} documents — skipping ingest.")
        print("Delete chroma_db/ to re-ingest.")
        return

    txt_files = list(DATA_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} text files in {DATA_DIR}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_docs, all_ids, all_metas = [], [], []
    chunk_counter = 0

    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, WORDS_PER_CHUNK, WORDS_OVERLAP)
        source = filepath.relative_to(DATA_DIR.parent).as_posix()
        print(f"  {source}: {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metas.append({"source": source, "chunk_index": idx})
            chunk_counter += 1

    print(f"Embedding {len(all_docs)} chunks...")
    batch_size = 64
    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        embeddings = embedder.encode(batch_docs).tolist()
        store.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas, embeddings=embeddings)
        print(f"  Stored {min(i + batch_size, len(all_docs))}/{len(all_docs)}")

    print(f"Ingest complete — {store.count()} chunks saved to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
