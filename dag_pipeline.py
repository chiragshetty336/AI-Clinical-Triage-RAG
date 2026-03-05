from ingestion import load_pdfs_with_cache
from embedding import generate_embeddings
from indexing import update_faiss_index


def task_detect_and_load():
    print("Task 1: Detecting new PDFs...")
    return load_pdfs_with_cache()


def task_embedding(chunks):
    print("Task 2: Generating embeddings...")
    return generate_embeddings(chunks)


def task_indexing(embeddings):
    print("Task 3: Updating FAISS...")
    update_faiss_index(embeddings)


def run_pipeline():
    chunks, embeddings, metadata = task_detect_and_load()
    task_indexing(embeddings)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
