"""Dense retrieval with normalized vectors; exact search suits this small corpus."""
import numpy as np


class EmbeddingRetriever:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", encoder=None):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name, device="cpu")
        self.chunks, self.encoder = chunks, encoder
        self.matrix = encoder.encode([c.text for c in chunks], normalize_embeddings=True)

    def retrieve(self, query, top_k=1):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ vector
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in indices]
