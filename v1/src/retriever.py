import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import FAISS_INDEX_DIR, EMBEDDING_MODEL, OPENAI_API_KEY,TOP_K, SYSTEM_PROMPT, LLM_MODEL, TEMPERATURE

class simple_retriever:
    def __init__(self):
        self.vectorstore = None
        self._load_index()


    def _load_index(self):
        if not os.path.exists(FAISS_INDEX_DIR):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_DIR}. Please run the ingest script first.")
        
        print(f"Loading FAISS index from {FAISS_INDEX_DIR}...")
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        self.vectorstore=FAISS.load_local(FAISS_INDEX_DIR, embeddings,allow_dangerous_deserialization=True)

        print("FAISS index loaded successfully.")

    def retrieve(self, query:str,top_k:int=TOP_K):
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results
    
    def get_context(self,query:str, top_k:int=TOP_K)->str:

        results=self.retrieve(query,top_k)
        
        context_parts=[]
        
        for i ,(doc,score) in enumerate(results,1):

            source = doc.metadata.get("doc_name", "Unknown")
            dept = doc.metadata.get("dept", "Unknown")
            context_parts.append(
                f"[Source {i}: {source} | Dept: {dept} | Relevance: {score:.3f}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(self,query:str,top_k:int=TOP_K)->list:
        results=self.retrieve(query,top_k)
        print(results)
        sources=[]
        for doc, score in results:
            sources.append({
                "source": doc.metadata.get("doc_name", "Unknown"),
                "dept": doc.metadata.get("dept", "Unknown"),
                "score": float(score),
                "preview": doc.page_content[:150] + "...",
            })
        return sources
    

if __name__ == "__main__":
    retriever = simple_retriever()

    test_query = "What is the annual leave entitlement?"
    print(f"\nQuery: {test_query}\n")

    sources = retriever.get_sources(test_query)
    for i, src in enumerate(sources, 1):
        print(f"  {i}. [{src['dept']}] {src['source']} (score: {src['score']:.3f})")
        print(f"     {src['preview']}\n")