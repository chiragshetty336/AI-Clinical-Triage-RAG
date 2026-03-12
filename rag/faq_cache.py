import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_FILE = "data/faq_cache.pkl"

model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")


def load_cache():

    if not os.path.exists(CACHE_FILE):
        return []

    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def save_cache(data):

    os.makedirs("data", exist_ok=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


def search_cache(query, threshold=0.85):

    cache = load_cache()

    if len(cache) == 0:
        return None

    query_emb = model.encode([query])[0]

    best_score = 0
    best_item = None

    for item in cache:

        score = np.dot(query_emb, item["embedding"])

        if score > best_score:
            best_score = score
            best_item = item

    if best_score > threshold:
        print("⚡ FAQ cache hit")
        return best_item

    return None


def store_cache(query, answer, triage_level, sources):

    cache = load_cache()

    emb = model.encode([query])[0].tolist()

    cache.append(
        {
            "query": query,
            "embedding": emb,
            "answer": answer,
            "triage_level": triage_level,
            "sources": sources,
        }
    )

    save_cache(cache)
