import numpy as np
import faiss
from rag.config import MODEL_NAME


def search(query, index, chunks, metadata, top_k=10):
    # 🔥 Load model INSIDE function (Airflow Safe)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    q_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_embedding)

    scores, ids = index.search(q_embedding, top_k)

    boosted_results = []
    query_words = set(query.lower().split())

    for rank, idx in enumerate(ids[0]):
        chunk = chunks[idx]
        source = metadata[idx]

        semantic_score = scores[0][rank]

        chunk_words = set(chunk.lower().split())
        keyword_overlap = len(query_words.intersection(chunk_words))

        combined_score = semantic_score + 0.02 * keyword_overlap

        boosted_results.append((combined_score, chunk, source))

    boosted_results = sorted(boosted_results, key=lambda x: x[0], reverse=True)

    results = [item[1] for item in boosted_results]
    sources = [item[2] for item in boosted_results]

    confidence = float(scores[0][0])

    return results, sources, confidence
