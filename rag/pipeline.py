from rag.ingestion import load_pdfs_with_cache
from rag.indexing import build_index_from_embeddings
from rag.config import INDEX_PATH
import os


def ingest_documents():
    print("📂 Starting document ingestion...")

    message = load_pdfs_with_cache()

    print("✅ Ingestion finished successfully.")
    print(message)

    return message


def update_faiss_index():
    print("🔄 Updating FAISS index...")

    message = load_pdfs_with_cache()

    print("✅ FAISS index updated successfully.")
    print(message)

    return message
