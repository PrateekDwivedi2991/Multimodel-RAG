"""
V2 RAG Pipeline
Full chain: RAG Fusion → Ensemble Retrieval → Rerank → LOTR → Compress → Generate

Pipeline:
  Query
    → RAG Fusion (4 query variations)
    → Per-dept vector search + BM25 (24 retrieval calls)
    → RRF merge + dedup (~20-25 docs)
    → Cross-encoder reranker (top 10)
    → LongContextReorder (attention optimization)
    → Context compressor (trim noise)
    → LLM generates answer
"""


import os 
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.post_retriever.reranker import CrossEncocderReranker
from src.post_retriever.compressor import ContextCompressor
from src.post_retriever.reorder import LongContextReorder
from src.generator import SimpleGenerator
from src.retreivers.ensemble import EnsembleRetriever

class RAGPipeline:
    def __init__(
        self,
        use_rag_fusion: bool = True,
        use_reranker: bool = True,
        use_lotr: bool = True,
        use_compressor: bool = True,
    ):
        self.use_rag_fusion = use_rag_fusion
        self.use_reranker = use_reranker
        self.use_lotr = use_lotr
        self.use_compressor = use_compressor

        # Initialize components
        self.ensemble = EnsembleRetriever()
        self.reranker = CrossEncocderReranker() if use_reranker else None
        self.reorder = LongContextReorder() if use_lotr else None
        self.compressor = ContextCompressor() if use_compressor else None
        self.generator = SimpleGenerator()

        # For stats
        self._last_pipeline_stats = {}

    def ask(self,query:str)->dict:

        stats = {"query": query, "timings": {}}
        t_start = time.time()

        t=time.time()
        merged_docs = self.ensemble.retrieve(query,use_rag_fusion=self.use_rag_fusion)
        # Ensure merged_docs is a list
        merged_docs = merged_docs if isinstance(merged_docs, list) else []
        stats["timings"]["ensemble_retrieval"] = round(time.time() - t, 3)
        stats["docs_after_retrieval"] = len(merged_docs)


        if self.reranker and merged_docs:
            t = time.time()
            merged_docs = self.reranker.rerank(query, merged_docs)
            stats["timings"]["reranker"] = round(time.time() - t, 3)
            stats["docs_after_rerank"] = len(merged_docs)
        if self.reorder and merged_docs:
            t = time.time()
            merged_docs = self.reorder.reorder(merged_docs)
            stats["timings"]["lotr"] = round(time.time() - t, 3)


        if self.compressor and merged_docs:
            t = time.time()
            merged_docs = self.compressor.compress(query, merged_docs)
            stats["timings"]["compressor"] = round(time.time() - t, 3)
            stats["docs_after_compress"] = len(merged_docs)

        # ── Step 5: Build context string ──
        context = self._format_context(merged_docs)
        stats["context_chars"] = len(context)

        # ── Step 6: Generate answer ──
        t = time.time()
        answer = self.generator.generate(query, context)
        stats["timings"]["generation"] = round(time.time() - t, 3)

        stats["total_time"] = round(time.time() - t_start, 3)
        self._last_pipeline_stats = stats

        # Build sources list
        sources = self._extract_sources(merged_docs)

        return {
            "answer": answer,
            "sources": sources,
            "stats": stats,
        }

    def ask_stream(self, query: str):
        """Streaming version - retrieves first, then streams generation."""
        # Steps 1-4: blocking retrieval pipeline
        merged_docs = self.ensemble.retrieve(query, use_rag_fusion=self.use_rag_fusion)

        if self.reranker and merged_docs:
            merged_docs = self.reranker.rerank(query, merged_docs)

        if self.reorder and merged_docs:
            merged_docs = self.reorder.reorder(merged_docs)

        if self.compressor and merged_docs:
            merged_docs = self.compressor.compress(query, merged_docs)

        context = self._format_context(merged_docs)
        self._last_sources = self._extract_sources(merged_docs)

        # Step 5: stream generation
        for token in self.generator.generate_stream(query, context):
            yield token

    @property
    def last_sources(self):
        return getattr(self, "_last_sources", [])

    @property
    def last_stats(self):
        return self._last_pipeline_stats

    @staticmethod
    def _format_context(doc_score_pairs: list) -> str:
        """Format documents into a context string with source labels."""
        if not doc_score_pairs:
            return "No documents found to format as context."
        
        parts = []
        for i, (doc, score,*_) in enumerate(doc_score_pairs, 1):
            source = doc.metadata.get("doc_name", "Unknown")
            dept = doc.metadata.get("dept", "Unknown")
            parts.append(
                f"[Source {i}: {source} | Dept: {dept.upper()}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    
    @staticmethod
    def _extract_sources(doc_score_pairs: list) -> list:
        """Extract source metadata for display."""
        if not doc_score_pairs:
            return []
        
        sources = []
        seen = set()
        for doc, score in doc_score_pairs:
            key = doc.metadata.get("doc_name", "Unknown")
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": key,
                    "dept": doc.metadata.get("dept", "Unknown"),
                    "score": float(score),
                    "preview": doc.page_content[:150] + "...",
                })
        return sources


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = RAGPipeline(
        use_rag_fusion=True,
        use_reranker=True,
        use_lotr=True,
        use_compressor=False,  # Skip compressor for faster testing
    )

    test_queries = [
        "What is the expense approval process?",
        "How does the deployment process work?",
        "What are the NDA requirements?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = pipeline.ask(q)
        print(f"\nA: {result['answer'][:300]}...")
        print(f"\nSources:")
        for s in result["sources"][:5]:
            print(f"  [{s['dept'].upper()}] {s['source']} (score: {s['score']:.4f})")
        print(f"\nStats: {result['stats']['timings']}")
        print(f"Docs: retrieval={result['stats'].get('docs_after_retrieval', '?')} "
              f"→ rerank={result['stats'].get('docs_after_rerank', '?')} "
              f"→ compress={result['stats'].get('docs_after_compress', '?')}")
