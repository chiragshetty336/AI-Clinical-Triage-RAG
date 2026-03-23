from sentence_transformers import CrossEncoder


class MedicalReranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def medical_filter(self, query, doc):
        query_lower = query.lower()
        doc_lower = doc.lower()

        # ❌ Remove unrelated topics
        if "pediatric" in doc_lower and "child" not in query_lower:
            return False

        if "pregnancy" in doc_lower and "pregnan" not in query_lower:
            return False

        if "covid" in doc_lower and "covid" not in query_lower:
            return False

        # ✅ Keep relevant
        return True

    def rerank(self, query, documents, metadata, top_k=5):

        # -----------------------
        # 1️⃣ FILTER FIRST (🔥 NEW)
        # -----------------------
        filtered_docs = []
        filtered_meta = []

        for doc, meta in zip(documents, metadata):
            if self.medical_filter(query, doc):
                filtered_docs.append(doc)
                filtered_meta.append(meta)

        # fallback if everything removed
        if len(filtered_docs) == 0:
            filtered_docs = documents
            filtered_meta = metadata

        # -----------------------
        # 2️⃣ CROSS-ENCODER
        # -----------------------
        pairs = [(query, doc) for doc in filtered_docs]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(filtered_docs, filtered_meta, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        top_docs = [r[0] for r in ranked[:top_k]]
        top_meta = [r[1] for r in ranked[:top_k]]

        return top_docs, top_meta
