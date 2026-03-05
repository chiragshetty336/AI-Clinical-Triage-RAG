import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data/guidelines")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data/embeddings_cache")

INDEX_PATH = os.path.join(CACHE_PATH, "medical_faiss.index")

MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
