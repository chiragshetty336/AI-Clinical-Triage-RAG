"""
evaluation/benchmark.py
Mistral + Groq vs Base Answer Benchmark
"""

import json
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_compare import compare_llms, compute_similarity


# 🔥 SAME BASE ANSWERS (KEEP CONSISTENT WITH DASHBOARD)
BASE_ANSWERS = [
    "Chest pain radiating to arm with hypotension indicates acute coronary syndrome. Immediate ECG, oxygen, aspirin and cardiology referral required.",
    "Worst headache with neck stiffness suggests subarachnoid hemorrhage or meningitis. CT scan and lumbar puncture required.",
    "Asthma with low oxygen indicates acute exacerbation. Oxygen, bronchodilators, steroids needed.",
    "Child with fever and sore throat likely infection. Throat exam, hydration, and antibiotics if bacterial.",
    "Mild ankle sprain managed with rest, ice, compression and elevation.",
    "Mild viral respiratory infection managed with rest and fluids.",
    "Cardiac arrest requires immediate CPR and defibrillation.",
    "Diabetes with high glucose needs evaluation for DKA and insulin management.",
]


# ─── Benchmark Queries ─────────────────────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Patient presents with crushing chest pain radiating to left arm, sweating, heart rate 130, BP 90/60.",
        "expected": "RED",
    },
    {
        "id": 2,
        "query": "Worst headache of life with neck stiffness and photophobia",
        "expected": "RED",
    },
    {
        "id": 3,
        "query": "Asthma patient oxygen 92% difficulty speaking",
        "expected": "YELLOW",
    },
    {"id": 4, "query": "Child with fever and sore throat", "expected": "YELLOW"},
    {"id": 5, "query": "Mild ankle sprain no swelling", "expected": "GREEN"},
    {"id": 6, "query": "Mild cough 3 days no fever", "expected": "GREEN"},
    {"id": 7, "query": "Unresponsive patient no pulse CPR started", "expected": "RED"},
    {"id": 8, "query": "Diabetes glucose 380 conscious", "expected": "YELLOW"},
]


# ─── RUN BENCHMARK ─────────────────────────────────────────────


def run_benchmark(output_dir="evaluation/results"):

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    print("\n===== BENCHMARK START =====\n")

    for i, item in enumerate(BENCHMARK_QUERIES):

        print(f"[{i+1}] {item['query']}")

        compare = compare_llms(item["query"])

        mistral_ans = compare["mistral"]["answer"]
        groq_ans = compare["gpt4"]["answer"]

        base = BASE_ANSWERS[i]

        # 🔥 COMPUTE AGAINST BASE
        m_score = compute_similarity(base, mistral_ans)
        g_score = compute_similarity(base, groq_ans)

        record = {
            "id": item["id"],
            "query": item["query"],
            "expected": item["expected"],
            # MISTRAL
            "mistral_triage": m_score.get("triage_level_b"),
            "mistral_correct": m_score.get("triage_level_b") == item["expected"],
            "mistral_composite": m_score.get("composite_score"),
            "mistral_semantic": m_score.get("semantic_similarity"),
            # GROQ
            "groq_triage": g_score.get("triage_level_b"),
            "groq_correct": g_score.get("triage_level_b") == item["expected"],
            "groq_composite": g_score.get("composite_score"),
            "groq_semantic": g_score.get("semantic_similarity"),
        }

        results.append(record)

        print(f"  Mistral score: {record['mistral_composite']}")
        print(f"  Groq score: {record['groq_composite']}\n")

    # ─── SUMMARY ─────────────────────────────────────────

    summary = {
        "mistral_accuracy": sum(r["mistral_correct"] for r in results) / len(results),
        "groq_accuracy": sum(r["groq_correct"] for r in results) / len(results),
        "avg_mistral_score": sum(r["mistral_composite"] for r in results)
        / len(results),
        "avg_groq_score": sum(r["groq_composite"] for r in results) / len(results),
    }

    print("\n===== SUMMARY =====")
    print(summary)

    # ─── SAVE JSON ─────────────────────────────────────

    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    # ─── SAVE CSV ─────────────────────────────────────

    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", json_path)

    return json_path


if __name__ == "__main__":
    run_benchmark()
