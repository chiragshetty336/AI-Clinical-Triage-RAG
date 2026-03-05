import faiss
from rag.config import INDEX_PATH


def build_index_from_embeddings(embeddings):
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    return index


def load_index():
    return faiss.read_index(INDEX_PATH)
