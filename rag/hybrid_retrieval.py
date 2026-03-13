from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


class HybridRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        # Tokenize documents for BM25
        tokenized_docs = [chunk.lower().split() for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, index, metadata, top_k=10):

        # -----------------------
        # VECTOR SEARCH (FAISS)
        # -----------------------

        q_embedding = model.encode([query]).astype("float32")

        # normalize for cosine similarity
        faiss.normalize_L2(q_embedding)

        scores, ids = index.search(q_embedding, top_k)

        vector_results = []
        vector_meta = []

        for idx in ids[0]:
            vector_results.append(self.chunks[idx])
            vector_meta.append(metadata[idx])

        # use FAISS similarity as confidence
        vector_confidence = float(scores[0][0])

        # -----------------------
        # BM25 KEYWORD SEARCH
        # -----------------------

        tokenized_query = query.lower().split()

        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_bm25_ids = np.argsort(bm25_scores)[::-1][:top_k]

        bm25_results = []
        bm25_meta = []

        for idx in top_bm25_ids:
            bm25_results.append(self.chunks[idx])
            bm25_meta.append(metadata[idx])

        # -----------------------
        # MERGE RESULTS
        # -----------------------

        combined_docs = vector_results + bm25_results
        combined_meta = vector_meta + bm25_meta

        # remove duplicates
        seen = set()
        unique_docs = []
        unique_meta = []

        for doc, meta in zip(combined_docs, combined_meta):

            if doc not in seen:
                unique_docs.append(doc)
                unique_meta.append(meta)
                seen.add(doc)

        return unique_docs[:top_k], unique_meta[:top_k], vector_confidence
