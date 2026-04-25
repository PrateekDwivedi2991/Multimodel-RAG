import os 
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import DEPT_FOLDERS, DEPT_INDEX_PATHS, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, OPENAI_API_KEY,INDEX_DIR,DEPARTMENTS

def load_department_docs(dept,folder):
    if not os.path.exists(folder):
        print(f" [{dept}] Folder not found:{folder}, skipping")
        return []
    
    loader=DirectoryLoader(folder,glob="**/*.txt",show_progress=False,loader_cls=TextLoader)
    docs=loader.load()
    print(f"  [{dept}] Loaded {len(docs)} documents")
    return docs

def chunk_documents(docs, dept):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
        chunk.metadata["dept"] = dept
        chunk.metadata["doc_name"] = source_file
        chunk.metadata["chunk_id"] = f"{dept}_{i}"
    return chunks


def build_department_index(dept,chunks,embeddings):

    if not chunks:
        return None
    
    vectorstore=FAISS.from_documents(chunks,embeddings)
    index_path=DEPT_INDEX_PATHS[dept]
    os.makedirs(os.path.dirname(index_path),exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"  [{dept}] Saved FAISS index with {len(chunks)} chunks to {index_path}")
    return vectorstore

def save_all_chunks_for_bm25(all_chunks):
    bm25_path=os.path.join(INDEX_DIR,"bm25_corpus.json")
    os.makedirs(INDEX_DIR, exist_ok=True)
    serialized = [{"content": c.page_content, "metadata": c.metadata} for c in all_chunks]
    with open(bm25_path, "w") as f:
        json.dump(serialized, f)
    print(f"\n  [BM25] Saved {len(serialized)} chunks → {bm25_path}")


def run_ingestion():
    print("=" * 60)
    print("  V2 MULTI-DEPARTMENT INGESTION PIPELINE")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    all_chunks = []
    start = time.time()

    for dept in DEPARTMENTS:
        folder = DEPT_FOLDERS[dept]
        print(f"\n  Processing: {dept.upper()}")
        docs = load_department_docs(dept, folder)
        if not docs:
            continue
        chunks = chunk_documents(docs, dept)
        print(f"  [{dept}] Chunked into {len(chunks)} pieces")
        build_department_index(dept, chunks, embeddings)
        all_chunks.extend(chunks)

    save_all_chunks_for_bm25(all_chunks)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  DONE! {len(all_chunks)} total chunks across {len(DEPARTMENTS)} departments")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_ingestion()

