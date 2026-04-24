import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import simple_retriever
from src.generator import simple_generator


class RAGPipeline:
    def __init__(self):
        self.retriever = simple_retriever()
        self.generator = simple_generator()

    def ask(self, query:str)->dict:
        # Step 1: Retrieve relevant documents
        retrieved_context = self.retriever.get_context(query)
        retrieved_sources=self.retriever.get_sources(query)
        # Step 2: Generate answer based on retrieved documents
        answer = self.generator.generate(query, retrieved_context)



        return {
            "answer": answer,
            "sources": retrieved_sources,
            "context": retrieved_context,
        }
    def ask_stream(self, query: str):
        # Step 1: Retrieve (blocking, fast)
        context = self.retriever.get_context(query)
        self._last_sources = self.retriever.get_sources(query)

        # Step 2: Stream generation
        for token in self.generator.generate_stream(query, context):
            yield token

    @property
    def last_sources(self):
        """Get sources from the last streaming call."""
        return getattr(self, "_last_sources", [])
    
if __name__ == "__main__":
    pipeline = RAGPipeline()

    queries = [
        "What is the annual leave policy?",
        "How does the performance review process work?",
        "What are the expense approval thresholds?",
        "Tell me about the health insurance plans",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = pipeline.ask(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source']} (score: {s['score']:.3f})")
