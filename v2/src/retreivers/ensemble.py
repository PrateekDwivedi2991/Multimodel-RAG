import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.retreivers.bm25_retriever import BM25Retriever
from src.retreivers.vector_retriever import MultiDeptVectorRetriever
from src.retreivers.rag_fusion import RAGFusion
from src.config import TOP_K_PER_DEPT,TOP_K_BM25

class EnsembleRetriever:

    def __init__(self):
        print("Initializing Ensemble Retriever...")
        self.vector_retriever=MultiDeptVectorRetriever()
        self.bm25_retriever=BM25Retriever()
        self.rag_fusion=RAGFusion()
        print("  Ensemble Retriever ready!\n")

    def retrieve(self, query:str,use_rag_fusion:bool=True):

        if use_rag_fusion:
            print("Generating query variations for RAG Fusion...")
            queries=self.rag_fusion.generate_queries(query)

        else:
            queries=[query]

        all_ranked_lists=[]

        for q in queries:
            print(f"Retrieving for query: '{q}'")
            vector_results=self.vector_retriever.retrieve(q, top_k=TOP_K_PER_DEPT)
            bm25_results=self.bm25_retriever.retrieve(q, top_k=TOP_K_BM25)
            # Ensure results are lists, default to empty list if None
            vector_results = vector_results if vector_results is not None else []
            bm25_results = bm25_results if bm25_results is not None else []
            print(f"  Retrieved {len(vector_results)} vector results and {len(bm25_results)} BM25 results.")
            # print(f"  Sample vector result: '{vector_results[0][0].page_content[:100]}...' with score {vector_results[0][1]:.4f}")
            # print(f"  Sample BM25 result: '{bm25_results[0][0].page_content[:100]}...' with score {bm25_results[0][1]:.4f}")
            all_ranked_lists.append(vector_results)
            all_ranked_lists.append(bm25_results)

        merged = RAGFusion.reciprocal_rank_fusion(all_ranked_lists)
        # Ensure merged is a list, default to empty list if None
        merged = merged if merged is not None else []

        return merged
    
    @property
    def total_chunks(self):
        return self.vector_retriever.total_chunks
    

if __name__=="__main__":
    retriever=EnsembleRetriever()
    query="What are the key responsibilities of the HR department?"
    results=retriever.retrieve(query)
    print(f"\nTotal query retrieved: {len(results)}")
    print("\nFinal Merged Results:")
    for doc, score in results:
        print(f"Score: {score:.4f} | Content: {doc.page_content[:100]}...")
        