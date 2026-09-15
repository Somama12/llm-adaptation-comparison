"""
Simple TF-IDF based retriever for RAG baseline.
Chunks a markdown handbook by section and retrieves the most relevant
chunk(s) for a given query using cosine similarity over TF-IDF vectors.

This is a dependency-light retrieval baseline (sklearn only). It can later
be swapped for an embedding-based retriever (sentence-transformers + FAISS)
without changing the harness interface.
"""
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    section_id: str
    title: str
    text: str


def chunk_markdown(path: str) -> list[Chunk]:
    """Split a markdown handbook into chunks by ### subsection headers."""
    with open(path, "r") as f:
        content = f.read()

    # Split on level-3 headers (###), keep header with its body
    pattern = r"(### .+?)(?=\n### |\n## |\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    chunks = []
    for m in matches:
        lines = m.strip().split("\n")
        header = lines[0].replace("### ", "").strip()
        # Strip [UPDATED] tag for section_id matching, keep for display
        section_id = re.sub(r"\s*\[UPDATED\]\s*", "", header).strip()
        body = "\n".join(lines[1:]).strip()
        chunks.append(Chunk(section_id=section_id, title=header, text=body))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [c.text for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 1) -> list[tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in ranked_idx]


if __name__ == "__main__":
    # Quick smoke test
    chunks = chunk_markdown("data/handbook_v1.md")
    print(f"Parsed {len(chunks)} chunks from handbook_v1.md")
    for c in chunks[:3]:
        print(f"  - {c.section_id}: {c.text[:60]}...")

    retriever = TfidfRetriever(chunks)
    results = retriever.retrieve("How often must passwords be reset?", top_k=1)
    for chunk, score in results:
        print(f"\nQuery result (score={score:.3f}): {chunk.section_id}")
        print(f"  {chunk.text}")
