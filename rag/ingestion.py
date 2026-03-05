import os
import fitz
import numpy as np
import pickle
import faiss
from rag.config import DATA_PATH, CACHE_PATH, MODEL_NAME


def extract_text_from_pdf(pdf_path):
    pages_data = []
    try:
        doc = fitz.open(pdf_path)
        for page_number, page in enumerate(doc):
            text = page.get_text()
            pages_data.append((page_number + 1, text))
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages_data


def chunk_text(text, max_words=300):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len((current_chunk + " " + para).split()) < max_words:
            current_chunk += " " + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def load_pdfs_with_cache():
    from sentence_transformers import SentenceTransformer

    print("📂 Starting document ingestion...")

    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    all_embeddings = []
    metadata = []

    os.makedirs(CACHE_PATH, exist_ok=True)

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            cache_file = os.path.join(CACHE_PATH, file + ".pkl")

            if os.path.exists(cache_file):
                print(f"Loading cache for: {file}")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                all_chunks.extend(cached_data["chunks"])
                all_embeddings.extend(cached_data["embeddings"])
                metadata.extend(cached_data["metadata"])

            else:
                print(f"Processing new file: {file}")

                pages = extract_text_from_pdf(path)

                file_chunks = []
                file_metadata = []

                for page_number, page_text in pages:
                    chunks = chunk_text(page_text)
                    for chunk in chunks:
                        file_chunks.append(chunk)
                        file_metadata.append({"source": file, "page": page_number})

                if file_chunks:
                    embeddings = model.encode(file_chunks, show_progress_bar=True)
                    embeddings = np.array(embeddings).astype("float32")

                    with open(cache_file, "wb") as f:
                        pickle.dump(
                            {
                                "chunks": file_chunks,
                                "embeddings": embeddings,
                                "metadata": file_metadata,
                            },
                            f,
                        )

                    all_chunks.extend(file_chunks)
                    all_embeddings.extend(embeddings)
                    metadata.extend(file_metadata)

    # -------------------------
    # 🔥 CREATE FAISS INDEX
    # -------------------------

    all_embeddings = np.array(all_embeddings).astype("float32")

    dimension = all_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(CACHE_PATH, "medical_faiss.index"))

    # Save metadata separately
    with open(os.path.join(CACHE_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "chunks": all_chunks,
                "metadata": metadata,
            },
            f,
        )

    print(f"✅ Stored {len(all_chunks)} chunks in FAISS")

    return f"Ingestion completed. {len(all_chunks)} chunks stored."
