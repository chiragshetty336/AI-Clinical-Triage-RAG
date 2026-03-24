from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


def expand_query(query):
    q = query.lower()

    if "chest pain" in q:
        return (
            query
            + " acute coronary syndrome myocardial infarction diagnosis treatment emergency"
        )

    if "breathing" in q:
        return (
            query + " respiratory failure hypoxia airway management emergency treatment"
        )

    if "unconscious" in q:
        return (
            query + " cardiac arrest loss of consciousness emergency resuscitation CPR"
        )

    if "fever" in q:
        return query + " infection causes treatment diagnosis mild illness"

    return query


class HybridRetriever:

    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_docs = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, index, metadata, top_k=15):

        expanded_query = expand_query(query)

        # =======================
        # FAISS
        # =======================
        q_embedding = model.encode([expanded_query]).astype("float32")
        faiss.normalize_L2(q_embedding)

        scores, ids = index.search(q_embedding, top_k)

        # =======================
        # BM25
        # =======================
        tokenized_query = expanded_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # normalize BM25
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
        )

        # =======================
        # HYBRID MERGE
        # =======================
        hybrid_scores = {}

        for i, idx in enumerate(ids[0]):
            hybrid_scores[idx] = 0.7 * scores[0][i]  # 🔥 stronger FAISS

        for idx, score in enumerate(bm25_scores):
            if idx in hybrid_scores:
                hybrid_scores[idx] += 0.3 * score

        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # =======================
        # OUTPUT
        # =======================
        final_docs = []
        final_meta = []

        for idx, score in sorted_results:
            doc = self.chunks[idx]

            # 🔥 final safety filter
            if len(doc.split()) < 40:
                continue

            final_docs.append(doc)
            final_meta.append(metadata[idx])

            if len(final_docs) >= top_k:
                break

        confidence = float(sorted_results[0][1]) if sorted_results else 0.0

        return final_docs, final_meta, confidence
