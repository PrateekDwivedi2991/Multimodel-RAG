import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import TOP_K_PER_DEPT,TOP_K_BM25
print(TOP_K_BM25)


print("testing ensemble retriever...")