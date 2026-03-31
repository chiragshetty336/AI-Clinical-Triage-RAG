import json
from rag.agent import medical_agent
from rag.indexing import load_index
import pickle
import os

# load your dataset
with open("evaluation/queries.json") as f:
    queries = json.load(f)

# load GPT answers
with open("evaluation/gpt_answers.json") as f:
    gpt_data = json.load(f)

gpt_map = {item["query"]: item["gpt_answer"] for item in gpt_data}

# load index
index = load_index()

metadata_path = os.path.join("data/embeddings_cache", "metadata.pkl")

with open(metadata_path, "rb") as f:
    metadata_store = pickle.load(f)

chunks = metadata_store["chunks"]
metadata = metadata_store["metadata"]


def simple_score(answer):
    answer = answer.lower()
    score = 0

    # clarity
    if len(answer.split()) > 10:
        score += 0.3

    # emergency awareness
    if any(w in answer for w in ["emergency", "immediate", "urgent"]):
        score += 0.3

    # actionability
    action_words = [
        "call",
        "seek",
        "go",
        "monitor",
        "rest",
        "hydrate",
        "drink",
        "medication",
    ]

    if any(w in answer for w in action_words):
        score += 0.4

    return score


your_scores = []
gpt_scores = []

for q in queries:
    query = q["query"]

    # YOUR MODEL
    result = medical_agent(query, index, chunks, metadata)
    your_answer = result["answer"]

    # GPT
    gpt_answer = gpt_map.get(query, "")

    your_score = simple_score(your_answer)
    gpt_score = simple_score(gpt_answer)

    your_scores.append(your_score)
    gpt_scores.append(gpt_score)

    print("\n==========================")
    print("Query:", query)

    print("\n--- YOUR MODEL ---")
    print(your_answer)
    print("Score:", your_score)

    print("\n--- GPT ---")
    print(gpt_answer)
    print("Score:", gpt_score)

    winner = "YOUR MODEL" if your_score > gpt_score else "GPT"

    print("\nWinner:", winner)
    print("==========================\n")


print("📊 FINAL COMPARISON")
print("Your Model Avg:", sum(your_scores) / len(your_scores))
print("GPT Avg:", sum(gpt_scores) / len(gpt_scores))
