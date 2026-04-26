import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import RERANKER_MODEL,RERANKER_TOP_K

from sentence_transformers import CrossEncoder

class CrossEncocderReranker:

    def __init__(self):
        print(f"Loading cross-encoder model: {RERANKER_MODEL}.....")

        self.model = CrossEncoder(RERANKER_MODEL)
        print("Cross-encoder model loaded successfully.")

    def rerank(self, query: str, doc_score_pairs: list,top_k: int = RERANKER_TOP_K) -> list:
        
        if not doc_score_pairs:
            return []
        
        docs=[doc for doc, _ in doc_score_pairs]

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]
        return top_docs
    
    