import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import DEPT_INDEX_PATHS, EMBEDDING_MODEL, OPENAI_API_KEY, TOP_K_PER_DEPT,DEPT_FOLDERS,DEPARTMENTS

class MultiDeptVectorRetriever:
    """Loads and searches per-department FAISS indexes."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,openai_api_key=OPENAI_API_KEY)
        self.stores = {}
        self._load_indexes()

    def _load_indexes(self):
        for dept in DEPARTMENTS:
            path = DEPT_INDEX_PATHS[dept]
            if os.path.exists(path):
                self.stores[dept] = FAISS.load_local(
                    path, self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"  [VectorRetriever] Loaded {dept} index")
            else:
                print(f"  [VectorRetriever] {dept} index not found, skipping")

    def retrieve(self, query: str, top_k: int = TOP_K_PER_DEPT):
        """
        Search all department indexes and return combined results.
        Returns list of (Document, score, dept) tuples.
        """
        all_results = []
        for dept, store in self.stores.items():
            results = store.similarity_search_with_score(query, k=top_k)
            for doc, score in results:
                all_results.append((doc, score, dept))
        return all_results

    def get_retrievers(self):
        """Return individual LangChain retrievers for ensemble."""
        retrievers = {}
        for dept, store in self.stores.items():
            retrievers[dept] = store.as_retriever(
                search_kwargs={"k": TOP_K_PER_DEPT}
            )
        return retrievers

    @property
    def total_chunks(self):
        return sum(s.index.ntotal for s in self.stores.values())
