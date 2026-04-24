import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HR_DOCS_DIR=os.path.join(os.path.dirname(__file__), "..","data","hr_docs")
FAISS_INDEX_DIR=os.path.join(os.path.dirname(__file__), "..","indexes","hr_index")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 5

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

SYSTEM_PROMPT = """You are an HR department assistant. Answer questions based ONLY on the provided context.

Rules:
1. Only use information from the context below to answer.
2. If the context doesn't contain the answer, say "I don't have enough information to answer that based on the available HR documents."
3. Cite which document the information comes from when possible.
4. Be concise and professional.

Context:
{context}
"""