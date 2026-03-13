from sentence_transformers import CrossEncoder


class MedicalReranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents, metadata, top_k=5):

        pairs = [(query, doc) for doc in documents]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, metadata, scores), key=lambda x: x[2], reverse=True
        )

        top_docs = [r[0] for r in ranked[:top_k]]
        top_meta = [r[1] for r in ranked[:top_k]]

        return top_docs, top_meta
