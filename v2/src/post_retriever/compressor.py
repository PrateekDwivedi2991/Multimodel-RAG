"""
V2 Context Compressor
- Filters out irrelevant sentences within each chunk
- Uses embedding similarity to keep only query-relevant content
- Reduces token count by 40-60%
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config import COMPRESSION_SIMILARITY_THRESHOLD, OPENAI_API_KEY,EMBEDDING_MODEL, OPENAI_API_KEY, COMPRESSION_SIMILARITY_THRESHOLD


class ContextCompressor:
    """Filters document content to keep only query-relevant sentences."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY,
        )

    def compress(self, query: str, doc_score_pairs: list,
                 threshold: float = COMPRESSION_SIMILARITY_THRESHOLD) -> list:
        """
        Compress documents by removing irrelevant sentences.

        Args:
            query: User's question
            doc_score_pairs: List of (Document, score) tuples
            threshold: Minimum cosine similarity to keep a sentence

        Returns:
            List of (Document, score) with trimmed content.
            Documents with no relevant sentences are removed entirely.
        """
        if not doc_score_pairs:
            return []

        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        query_vec = np.array(query_embedding)

        compressed = []

        for doc, score in doc_score_pairs:
            # Split into sentences
            sentences = self._split_sentences(doc.page_content)
            if not sentences:
                continue

            # Embed all sentences
            sentence_embeddings = self.embeddings.embed_documents(sentences)

            # Filter by similarity
            relevant_sentences = []
            for sent, sent_emb in zip(sentences, sentence_embeddings):
                similarity = self._cosine_similarity(query_vec, np.array(sent_emb))
                if similarity >= threshold:
                    relevant_sentences.append(sent)

            # Only keep docs with relevant content
            if relevant_sentences:
                new_doc = Document(
                    page_content=" ".join(relevant_sentences),
                    metadata=doc.metadata,
                )
                compressed.append((new_doc, score))

        return compressed

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split text into sentences (simple heuristic)."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
