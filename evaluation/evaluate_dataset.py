import json
from sentence_transformers import SentenceTransformer, util
from rag.agent import medical_agent
from rag.indexing import load_index
from rag.vitals_triage import calculate_vital_triage
from rag.config import INDEX_PATH
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")


# load dataset
with open("evaluation/queries.json") as f:
    queries = json.load(f)


# load RAG index
index = load_index()

metadata_path = os.path.join("data/embeddings_cache", "metadata.pkl")

with open(metadata_path, "rb") as f:
    metadata_store = pickle.load(f)

chunks = metadata_store["chunks"]
metadata = metadata_store["metadata"]


results = []

for q in queries:

    query = q["query"]
    ground_truth = q["ground_truth"]
    expected_triage = q["expected_triage"]

    # run RAG system
    result = medical_agent(query, index, chunks, metadata)

    # run vital triage
    triage_vitals = calculate_vital_triage(
        heart_rate=q.get("heart_rate"),
        oxygen=q.get("oxygen"),
        temperature=q.get("temperature"),
        systolic_bp=q.get("systolic_bp"),
    )

    # override triage if vitals critical
    if triage_vitals == "RED":
        result["triage_level"] = "RED"

    rag_answer = result["answer"]
    predicted_triage = result["triage_level"]

    embeddings = model.encode([rag_answer, ground_truth], convert_to_tensor=True)

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    print("\n---------------------------")
    print("Query:", query)
    print("Expected triage:", expected_triage)
    print("Predicted triage:", predicted_triage)
    print("Similarity:", similarity)

    triage_correct = predicted_triage == expected_triage

    results.append(
        {"query": query, "triage_correct": triage_correct, "similarity": similarity}
    )


avg_similarity = sum(r["similarity"] for r in results) / len(results)
triage_accuracy = sum(r["triage_correct"] for r in results) / len(results)

print("Average similarity:", avg_similarity)
print("Triage accuracy:", triage_accuracy)
