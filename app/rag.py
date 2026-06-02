from sentence_transformers import SentenceTransformer
from app.vector_store import VectorStore
from app.config import CHROMA_PATH, TOP_K

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_store = VectorStore(CHROMA_PATH)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    embedding = _embedder.encode(query).tolist()
    return _store.query(embedding, n_results=top_k)


def collection_count() -> int:
    return _store.count()
