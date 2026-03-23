import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# File to store cache
CACHE_FILE = "data/faq_cache.pkl"

# Embedding model
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")


# Load cache from file
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return []

    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


# Save cache to file
def save_cache(data):
    os.makedirs("data", exist_ok=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


# ✅ FIXED search function (NO ERROR NOW)
def search_cache(query, context=None, threshold=0.85):
    cache = load_cache()

    if len(cache) == 0:
        return None

    # Convert query to embedding
    query_emb = model.encode([query])[0]

    best_score = 0
    best_item = None

    # Find best match
    for item in cache:
        stored_emb = np.array(item["embedding"])

        # ✅ SAFETY CHECK
        if stored_emb.shape != query_emb.shape:
            continue

        score = np.dot(query_emb, stored_emb)

        if score > best_score:
            best_score = score
            best_item = item

    # Return only if above threshold
    if best_score > threshold:
        print("⚡ FAQ cache hit")
        return best_item

    return None


# Store new result in cache
def store_cache(query, answer, triage_level, sources, embedding=None):
    cache = load_cache()

    # If embedding is not provided, generate it
    if embedding is None:
        emb = model.encode([query])[0].tolist()
    else:
        emb = embedding

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
