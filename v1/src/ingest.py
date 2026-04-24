import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


from src.config import HR_DOCS_DIR, FAISS_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,OPENAI_API_KEY



def load_documents():
    loader = DirectoryLoader(HR_DOCS_DIR, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")
    return documents

def chunk_documents(docs):
    """Split documents into chunks."""
    print(f"\n  Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    
    for i,chunk in enumerate(chunks):
        source_file=os.path.basename(chunk.metadata.get("source","unknown"))
        chunk.metadata["source_file"]=source_file
        chunk.metadata["doc_name"]="HR"
        chunk.metadata["chunk_id"]=i

    print(f"  Created {len(chunks)} chunks from {len(docs)} documents")

    print(f"\n  Sample chunk (#{0}):")
    print(f"  Source: {chunks[0].metadata['doc_name']}")
    print(f"  Content: {chunks[0].page_content[:200]}...")

    return chunks

def embed_and_save(chunks):

    print(f"\n Embedding {len(chunks)} chunks using {EMBEDDING_MODEL} model ....")
    embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    start=time.time()
    vectorstore=FAISS.from_documents(chunks, embeddings)
    elapsed=time.time()-start

    print(f"Embedded in elapsed time: {elapsed:.2f} seconds")

    os.makedirs(os.path.dirname(FAISS_INDEX_DIR), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"Saved FAISS index to {FAISS_INDEX_DIR}")

    return vectorstore

def run_ingest():
    print("Starting HR document ingestion...")
    documents = load_documents()
    chunks = chunk_documents(documents)
    vectorstore = embed_and_save(chunks)
    print(f"  DONE! {len(chunks)} chunks indexed and saved.")
    print("Ingestion complete.")

    return vectorstore


if __name__ == "__main__":
    run_ingest()