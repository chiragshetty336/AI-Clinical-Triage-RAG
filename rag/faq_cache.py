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
