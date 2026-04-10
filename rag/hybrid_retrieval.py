from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


# ─────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────
def expand_query(query):
    q = query.lower()

    if "chest pain" in q:
        return query + " acute coronary syndrome myocardial infarction emergency"

    if "breathing" in q:
        return query + " respiratory failure hypoxia airway emergency"

    if "unconscious" in q:
        return query + " cardiac arrest CPR emergency resuscitation"

    if "fever" in q:
        return query + " infection diagnosis treatment mild illness"

    return query


# ─────────────────────────────────────────
# Hybrid Retriever
# ─────────────────────────────────────────
class HybridRetriever:

    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_docs = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, index, metadata, top_k=15):

        expanded_query = expand_query(query)

        # =======================
        # FAISS SEARCH
        # =======================
        q_embedding = model.encode([expanded_query]).astype("float32")
        faiss.normalize_L2(q_embedding)

        scores, ids = index.search(q_embedding, top_k)

        # =======================
        # BM25 SEARCH
        # =======================
        tokenized_query = expanded_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        if np.max(bm25_scores) != np.min(bm25_scores):
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
                np.max(bm25_scores) - np.min(bm25_scores)
            )
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # =======================
        # HYBRID MERGE
        # =======================
        hybrid_scores = {}

        # 🔥 FAISS results
        for i, idx in enumerate(ids[0]):

            if idx == -1:  # ✅ FIX (FAISS edge case)
                continue

            i_idx = int(idx)

            if i_idx < len(self.chunks):
                hybrid_scores[i_idx] = 0.7 * float(scores[0][i])

        # 🔥 BM25 results
        for i_idx, score in enumerate(bm25_scores):
            if i_idx < len(self.chunks):
                if i_idx in hybrid_scores:
                    hybrid_scores[i_idx] += 0.3 * float(score)
                else:
                    hybrid_scores[i_idx] = 0.3 * float(score)

        # =======================
        # SORT RESULTS
        # =======================
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # =======================
        # FINAL FILTER + OUTPUT
        # =======================
        final_docs = []
        final_meta = []

        for idx, score in sorted_results:

            if idx >= len(self.chunks):
                continue

            doc = self.chunks[idx]

            # 🔥 SOFTEN FILTER (FIX)
            if len(doc.split()) < 10:
                continue

            final_docs.append(doc)

            if idx < len(metadata):
                final_meta.append(metadata[idx])
            else:
                final_meta.append({})

            if len(final_docs) >= top_k:
                break

        # 🔥 FALLBACK (CRITICAL FIX)
        if not final_docs:
            for idx, score in sorted_results[:top_k]:
                i = int(idx)

                if i < len(self.chunks):
                    final_docs.append(self.chunks[i])
                    final_meta.append(metadata[i] if i < len(metadata) else {})

        # =======================
        # CONFIDENCE
        # =======================
        confidence = float(sorted_results[0][1]) if sorted_results else 0.0

        return final_docs, final_meta, confidence
