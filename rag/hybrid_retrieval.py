from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


def expand_query(query):
    q = query.lower()

    if "chest pain" in q:
        return query + " cardiac coronary myocardial infarction heart attack angina"

    if "shortness of breath" in q or "breathing" in q:
        return query + " respiratory distress airway oxygen hypoxia"

    if "fever" in q:
        return query + " infection sepsis temperature"

    return query


def rewrite_query(query):
    q = query.lower()

    if "chest pain" in q:
        return "chest pain cardiac emergency acute coronary syndrome myocardial infarction heart attack"

    if "shortness of breath" in q:
        return "respiratory distress airway breathing emergency oxygen hypoxia"

    return query


def clean_chunk(text, query):
    text_lower = text.lower()
    query_lower = query.lower()

    # -----------------------
    # 1️⃣ REMOVE JUNK
    # -----------------------
    if any(x in text_lower for x in ["chapter", "case study", "discussion", "answers"]):
        return None

    # -----------------------
    # 2️⃣ REMOVE PROCEDURES
    # -----------------------
    if any(
        x in text_lower
        for x in ["insertion", "procedure", "technique", "step", "insert", "device"]
    ):
        return None

    # -----------------------
    # 3️⃣ REMOVE VERY SHORT
    # -----------------------
    if len(text.strip()) < 100:
        return None

    # -----------------------
    # 4️⃣ REMOVE IRRELEVANT DOMAINS (🔥 NEW)
    # -----------------------
    if "pediatric" in text_lower and "child" not in query_lower:
        return None

    if "pregnancy" in text_lower and "pregnan" not in query_lower:
        return None

    if "covid" in text_lower and "covid" not in query_lower:
        return None

    # -----------------------
    # 5️⃣ INTENT-AWARE FILTER
    # -----------------------

    if "chest pain" in query_lower:
        if not any(
            x in text_lower
            for x in ["chest pain", "cardiac", "coronary", "heart", "myocardial"]
        ):
            return None

    if any(x in query_lower for x in ["breathing", "shortness of breath"]):
        if not any(
            x in text_lower for x in ["breathing", "respiratory", "airway", "oxygen"]
        ):
            return None

    if "fever" in query_lower:
        if not any(x in text_lower for x in ["fever", "infection", "temperature"]):
            return None

    # -----------------------
    # 6️⃣ CLEAN TEXT
    # -----------------------
    text = " ".join(text.split())

    return text


class HybridRetriever:

    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_docs = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, index, metadata, top_k=10):

        query = rewrite_query(query)

        # =======================
        # FAISS SEARCH
        # =======================
        expanded_query = expand_query(query)
        q_embedding = model.encode([expanded_query]).astype("float32")
        faiss.normalize_L2(q_embedding)

        scores, ids = index.search(q_embedding, top_k)

        faiss_scores = scores[0]
        if len(faiss_scores) > 0:
            faiss_scores = (faiss_scores - np.min(faiss_scores)) / (
                np.max(faiss_scores) - np.min(faiss_scores) + 1e-8
            )

        # =======================
        # BM25 SEARCH
        # =======================
        expanded_query = expand_query(query)
        tokenized_query = expanded_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
        )

        # =======================
        # HYBRID SCORING
        # =======================
        hybrid_scores = {}

        for i, idx in enumerate(ids[0]):
            hybrid_scores[idx] = 0.6 * faiss_scores[i]  # ✅ stronger semantic

        for idx, score in enumerate(bm25_scores):
            if idx in hybrid_scores:
                hybrid_scores[idx] += 0.4 * score
            else:
                hybrid_scores[idx] = 0.4 * score

        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # =======================
        # FINAL OUTPUT
        # =======================
        final_docs = []
        final_meta = []
        final_scores = []

        for idx, score in sorted_results:
            cleaned = clean_chunk(self.chunks[idx], query)

            if cleaned:
                final_docs.append(cleaned)
                final_meta.append(metadata[idx])
                final_scores.append(score)

            if len(final_docs) >= top_k:
                break

        confidence = float(final_scores[0]) if final_scores else 0.0

        return final_docs, final_meta, confidence
