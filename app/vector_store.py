import json
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self, path: str):
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._docs_path = self._dir / "documents.json"
        self._emb_path = self._dir / "embeddings.npy"
        self._docs: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    def _load(self):
        if self._docs_path.exists():
            self._docs = json.loads(self._docs_path.read_text())
        if self._emb_path.exists() and self._docs:
            self._embeddings = np.load(str(self._emb_path))

    def _save(self):
        self._docs_path.write_text(json.dumps(self._docs))
        if self._embeddings is not None:
            np.save(str(self._emb_path), self._embeddings)

    def count(self) -> int:
        return len(self._docs)

    def add(self, documents: list[str], ids: list[str], metadatas: list[dict], embeddings: list[list[float]]):
        existing_ids = {d["id"] for d in self._docs}
        new_docs, new_embs = [], []
        for doc, id_, meta, emb in zip(documents, ids, metadatas, embeddings):
            if id_ not in existing_ids:
                self._docs.append({"id": id_, "text": doc, "metadata": meta})
                new_embs.append(emb)

        if not new_embs:
            return

        arr = np.array(new_embs, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = arr
        else:
            self._embeddings = np.vstack([self._embeddings, arr])
        self._save()

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        if not self._docs or self._embeddings is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms
        scores = normed @ q_norm

        top_k = min(n_results, len(self._docs))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return [
            {"text": self._docs[i]["text"], "source": self._docs[i]["metadata"].get("source", "unknown")}
            for i in indices
        ]
