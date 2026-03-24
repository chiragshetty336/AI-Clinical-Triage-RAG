from sentence_transformers import CrossEncoder


class MedicalReranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents, metadata, top_k=5):

        pairs = [(query, doc) for doc in documents]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, metadata, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        # 🔥 STRICT FILTER
        filtered_docs = []
        filtered_meta = []

        for doc, meta, score in ranked:
            if score > 0.35:  # 🔥 important threshold
                filtered_docs.append(doc)
                filtered_meta.append(meta)

        # fallback
        if len(filtered_docs) == 0:
            filtered_docs = [r[0] for r in ranked[:top_k]]
            filtered_meta = [r[1] for r in ranked[:top_k]]

        return filtered_docs[:top_k], filtered_meta[:top_k]
