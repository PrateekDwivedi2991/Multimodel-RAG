import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


from src.config import INDEX_DIR,TOP_K_BM25
# from config import INDEX_DIR,TOP_K_BM25



class BM25Retriever:
    """BM25 keyword search across all department chunks."""

    def __init__(self):
        self.documents = []
        self.bm25 = None
        self._load_and_build()

    def _load_and_build(self):
        bm25_path = os.path.join(INDEX_DIR, "bm25_corpus.json")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 data not found at {bm25_path}. Run ingest.py first."
            )

        with open(bm25_path, "r") as f:
            data = json.load(f)

        self.documents = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in data
        ]

        # Tokenize for BM25
        tokenized = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

        print(f"  [BM25Retriever] Built index with {len(self.documents)} chunks")

    def retrieve(self, query: str, top_k: int = TOP_K_BM25):
        """
        Search using BM25 keyword matching.
        Returns list of (Document, score) tuples.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches with positive score
                results.append((self.documents[idx], float(scores[idx])))

        return results
