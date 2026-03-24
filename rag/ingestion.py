import os
import fitz
import numpy as np
import pickle
import faiss
from rag.config import DATA_PATH, CACHE_PATH, MODEL_NAME
from sentence_transformers import SentenceTransformer


# =========================
# 🔥 CLEANING FUNCTION
# =========================
def clean_chunk(text: str):
    text_lower = text.lower()

    if len(text.split()) < 40:
        return None

    # ❌ REMOVE TRIAGE / TRAINING CONTENT (CRITICAL)
    bad_keywords = [
        "triage",
        "esi",
        "emergency severity index",
        "etat",
        "module",
        "airway and breathing practice",
        "learning objectives",
    ]

    if any(k in text_lower for k in bad_keywords):
        return None

    # ❌ REMOVE PROCEDURAL TEXT
    if any(
        k in text_lower
        for k in ["procedure", "technique", "step", "practice", "equipment"]
    ):
        return None

    # ✅ FORCE CLINICAL CONTENT
    must_have = [
        "diagnosis",
        "treatment",
        "management",
        "symptoms",
        "causes",
        "clinical",
        "syndrome",
        "disease",
    ]

    if not any(k in text_lower for k in must_have):
        return None

    return text.strip()


# =========================
# 🔥 BETTER CHUNKING
# =========================
def chunk_text(text, max_words=120):  # 🔥 reduced size
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


# =========================
# PDF EXTRACTION
# =========================
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


# =========================
# MAIN INGESTION
# =========================
def load_pdfs_with_cache():

    print("📂 Starting document ingestion...")

    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    all_embeddings = []
    metadata = []

    os.makedirs(CACHE_PATH, exist_ok=True)

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):

            path = os.path.join(DATA_PATH, file)
            print(f"Processing: {file}")

            pages = extract_text_from_pdf(path)

            for page_number, page_text in pages:

                chunks = chunk_text(page_text)

                for chunk in chunks:
                    cleaned = clean_chunk(chunk)

                    if cleaned:
                        all_chunks.append(cleaned)
                        metadata.append({"source": file, "page": page_number})

    print(f"✅ Clean chunks: {len(all_chunks)}")

    # =========================
    # EMBEDDINGS
    # =========================
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize embeddings (🔥 important)
    faiss.normalize_L2(embeddings)

    # =========================
    # FAISS INDEX
    # =========================
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  # 🔥 changed to cosine similarity
    index.add(embeddings)

    faiss.write_index(index, os.path.join(CACHE_PATH, "medical_faiss.index"))

    # Save metadata
    with open(os.path.join(CACHE_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": metadata}, f)

    print(f"✅ Stored {len(all_chunks)} clean chunks")

    return f"Ingestion completed. {len(all_chunks)} chunks stored."


if __name__ == "__main__":
    load_pdfs_with_cache()
