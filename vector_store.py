"""
vector_store.py
Builds a FAISS index from text chunks using SentenceTransformers embeddings.
Supports similarity search to retrieve relevant chunks for a query.
"""

from typing import List
import numpy as np


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: List[str] = []
        self.index = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype("float32")

    def build(self, chunks: List[str]):
        """Embed all chunks and build the FAISS index."""
        import faiss
        self.chunks = chunks
        embeddings = self._embed(chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query: str, k: int = 4) -> List[str]:
        """Return top-k most relevant chunks for a query."""
        if self.index is None or len(self.chunks) == 0:
            raise RuntimeError("Vector store is empty. Call build() first.")
        query_vec = self._embed([query])
        distances, indices = self.index.search(query_vec, min(k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
