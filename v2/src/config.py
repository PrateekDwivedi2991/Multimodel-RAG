
import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

DEPARTMENTS = ["hr", "finance", "legal", "engineering", "marketing"]

DEPT_FOLDERS = {
    "hr": os.path.join(DATA_DIR, "hr_docs"),
    "finance": os.path.join(DATA_DIR, "finance_docs"),
    "legal": os.path.join(DATA_DIR, "legal_docs"),
    "engineering": os.path.join(DATA_DIR, "engineering_docs"),
    "marketing": os.path.join(DATA_DIR, "marketing_docs"),
}

DEPT_INDEX_PATHS = {
    dept: os.path.join(INDEX_DIR, f"{dept}_index")
    for dept in DEPARTMENTS
}

# Legacy V1 compat
HR_DOCS_DIR = DEPT_FOLDERS["hr"]
FAISS_INDEX_DIR = DEPT_INDEX_PATHS["hr"]

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding ---
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Retrieval ---
TOP_K_PER_DEPT = 3
TOP_K_BM25 = 5
TOP_K_FINAL = 10

# --- RAG Fusion ---
RAG_FUSION_QUERIES = 3
RAG_FUSION_MODEL = "gpt-4o-mini"

# --- Ensemble Weights ---
VECTOR_WEIGHT = 0.15
BM25_WEIGHT = 0.25

# --- Cross-encoder Reranker ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 10

# --- Contextual Compression ---
COMPRESSION_SIMILARITY_THRESHOLD = 0.76

# --- Generation ---
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# --- Prompt ---
SYSTEM_PROMPT = """You are a company knowledge assistant with access to documents from multiple departments: HR, Finance, Legal, Engineering, and Marketing.

Answer questions based ONLY on the provided context.

Rules:
1. Only use information from the context below to answer.
2. If the context doesn't contain the answer, say "I don't have enough information in the available documents."
3. ALWAYS cite which department and document the information comes from.
4. If information comes from multiple departments, synthesize it and cite all sources.
5. Be concise and professional.

Context:
{context}
"""
