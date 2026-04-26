import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from openai import OpenAI
from langchain_core.documents import Document

from src.config import OPENAI_API_KEY,RAG_FUSION_QUERIES,RAG_FUSION_MODEL

class RAGFusion:
    """Generates query variations and fuses retrieval results with RRF."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_queries(self, original_query: str) -> list[str]:
        """
        Generate variations of the original query.
        Returns [original_query, variation_1, variation_2, ...]
        """
        prompt = f"""You are a helpful assistant that generates search query variations.
Given the original query, generate {RAG_FUSION_QUERIES} alternative versions that:
- Capture the same intent but use different wording
- Cover different aspects or perspectives of the question
- Use synonyms and related terminology

Original query: "{original_query}"

Return ONLY the alternative queries, one per line. No numbering, no explanations."""

        response = self.client.chat.completions.create(
            model=RAG_FUSION_MODEL,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        variations = response.choices[0].message.content.strip().split("\n")
        variations = [v.strip() for v in variations if v.strip()]

        # Always include original as first query
        all_queries = [original_query] + variations[:RAG_FUSION_QUERIES]

        return all_queries

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: List of lists, each containing (Document, score) tuples
            k: RRF constant (default 60)

        Returns:
            Single merged and deduplicated list of (Document, rrf_score) tuples
        """
        doc_scores = {}  # content_hash -> {doc, score}

        for ranked_list in ranked_lists:
            # Skip None or empty lists
            if ranked_list is None or not ranked_list:
                continue
                
            for rank, (doc, _original_score,*_) in enumerate(ranked_list):
                # Use content as dedup key
                key = hash(doc.page_content)

                rrf_score = 1.0 / (k + rank + 1)

                if key in doc_scores:
                    doc_scores[key]["score"] += rrf_score
                else:
                    doc_scores[key] = {"doc": doc, "score": rrf_score}

        # Sort by RRF score descending
        merged = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

        return [(item["doc"], item["score"]) for item in merged]
