import json
from rag.agent import medical_agent
from rag.indexing import load_index
from rag.vitals_triage import calculate_vital_triage
import pickle
import os


def evaluate_answer(predicted_answer, ground_truth, predicted_triage, expected_triage):

    pred = predicted_answer.lower()
    gt = ground_truth.lower()

    score = 0

    # =========================
    # 1️⃣ TRIAGE (weight: 0.4)
    # =========================
    if predicted_triage == expected_triage:
        score += 0.4

    # =========================
    # 2️⃣ REASONING MATCH (0.3)
    # =========================
    gt_keywords = gt.split()

    overlap = sum(1 for word in gt_keywords if word in pred)

    if overlap >= 3:
        score += 0.3
    elif overlap == 2:
        score += 0.2
    elif overlap == 1:
        score += 0.1

    # =========================
    # 3️⃣ EMERGENCY LANGUAGE (0.2)
    # =========================
    severity_words = ["emergency", "critical", "urgent", "immediate"]

    if any(word in pred for word in severity_words):
        score += 0.2

    # =========================
    # 4️⃣ PENALTY (IMPORTANT)
    # =========================
    # Penalize vague answers
    if len(pred.split()) < 20:
        score -= 0.1

    return round(max(score, 0), 3)


# =========================
# LOAD DATA
# =========================
with open("evaluation/queries.json") as f:
    queries = json.load(f)


# =========================
# LOAD INDEX
# =========================
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

    result = medical_agent(query, index, chunks, metadata)

    triage_vitals = calculate_vital_triage(
        heart_rate=q.get("heart_rate"),
        oxygen=q.get("oxygen"),
        temperature=q.get("temperature"),
        systolic_bp=q.get("systolic_bp"),
    )

    if triage_vitals == "RED":
        result["triage_level"] = "RED"

    predicted_triage = result["triage_level"]
    rag_answer = result["answer"]

    triage_correct = predicted_triage == expected_triage

    score = evaluate_answer(rag_answer, ground_truth, predicted_triage, expected_triage)

    print("\n===========================")
    print("Query:", query)
    print("Expected:", expected_triage)
    print("Predicted:", predicted_triage)
    print("Score:", score)
    print("===========================\n")

    results.append({"query": query, "triage_correct": triage_correct, "score": score})


avg_score = sum(r["score"] for r in results) / len(results)
triage_accuracy = sum(r["triage_correct"] for r in results) / len(results)

print("\n📊 FINAL RESULTS")
print("Triage Accuracy:", round(triage_accuracy, 3))
print("Average Score:", round(avg_score, 3))
