"""
Vector store + retrieval layer.

Implemented as a small, dependency-light NumPy-based vector store instead
of ChromaDB. This is a deliberate choice: ChromaDB's HNSW index
(chroma-hnswlib) is a C++ extension that needs a compiler toolchain to
build on many Windows machines (and on some free hosting containers),
which makes local setup and deployment unnecessarily fragile. For a
knowledge base this size (a few hundred chunks), brute-force cosine
similarity over a NumPy matrix is just as fast in practice and has zero
native-build requirements — pip install just works everywhere.

Embeddings: sentence-transformers (local, free, no API key).
Persistence: a single .npz file (vectors) + a .json file (texts/metadata).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, persist_dir: str | Path, embedding_model: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.persist_dir / "vectors.npz"
        self._meta_path = self.persist_dir / "meta.json"

        self._model = SentenceTransformer(embedding_model)
        self._embeddings: np.ndarray | None = None
        self._records: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._vectors_path.exists() and self._meta_path.exists():
            data = np.load(self._vectors_path)
            self._embeddings = data["embeddings"]
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._records = json.load(f)
        else:
            self._embeddings = np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
            self._records = []

    def _save(self) -> None:
        np.savez_compressed(self._vectors_path, embeddings=self._embeddings)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f)

    # ------------------------------------------------------------------
    # public API (same shape as before, so chain.py needs no changes)
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        return self._embeddings is None or self._embeddings.shape[0] == 0

    def reset(self) -> None:
        dim = self._model.get_sentence_embedding_dimension()
        self._embeddings = np.zeros((0, dim), dtype=np.float32)
        self._records = []
        self._save()

    def add_chunks(self, chunks, batch_size: int = 64) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        new_records = [
            {
                "doc_id": c.doc_id,
                "source": c.source,
                "format": c.format,
                "category": c.category,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in chunks
        ]

        if self._embeddings.shape[0] == 0:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])
        self._records.extend(new_records)
        self._save()

    def query(self, query_text: str, top_k: int = 5, category: str | None = None):
        """
        Returns a list of dicts: {text, source, category, doc_id, relevance_score}
        sorted by relevance (most relevant first). Cosine similarity since
        both query and stored embeddings are L2-normalized.
        """
        if self.is_empty():
            return []

        query_vec = self._model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)[0]

        candidate_idx = range(len(self._records))
        if category:
            candidate_idx = [i for i in candidate_idx if self._records[i]["category"] == category]
            if not candidate_idx:
                return []
            sub_embeddings = self._embeddings[candidate_idx]
        else:
            candidate_idx = list(candidate_idx)
            sub_embeddings = self._embeddings

        scores = sub_embeddings @ query_vec  # cosine similarity (both normalized)
        top_n = min(top_k, len(candidate_idx))
        top_local_idx = np.argsort(-scores)[:top_n]

        out = []
        for local_i in top_local_idx:
            global_i = candidate_idx[local_i]
            rec = self._records[global_i]
            out.append(
                {
                    "text": rec["text"],
                    "source": rec["source"],
                    "category": rec["category"],
                    "doc_id": rec["doc_id"],
                    "relevance_score": round(float(scores[local_i]), 4),
                }
            )
        return out

    def count(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[0]
