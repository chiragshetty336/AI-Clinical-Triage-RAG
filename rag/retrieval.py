import numpy as np
import faiss
from rag.config import MODEL_NAME
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)


def search(query, index, chunks, metadata, top_k=10):

    q_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_embedding)

    scores, ids = index.search(q_embedding, top_k)

    results = []
    sources = []

    query_words = set(query.lower().split())

    for rank, idx in enumerate(ids[0]):

        chunk = chunks[idx]
        source = metadata[idx]

        semantic_score = scores[0][rank]

        chunk_words = set(chunk.lower().split())
        keyword_overlap = len(query_words.intersection(chunk_words))

        combined_score = semantic_score + 0.02 * keyword_overlap

        if combined_score > 0.35:  # 🔥 filter weak chunks
            results.append((combined_score, chunk, source))

    results = sorted(results, key=lambda x: x[0], reverse=True)

    top_chunks = [r[1] for r in results]
    top_sources = [r[2] for r in results]

    confidence = float(scores[0][0])

    return top_chunks, top_sources, confidence
