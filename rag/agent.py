from unittest import result

import requests
from rag.triage import classify_triage
from rag.retrieval import search
from rag.generation import generate_answer
from rag.evaluation import calculate_faithfulness


# ==============================
# MEDICAL QUERY NORMALIZATION
# ==============================

MEDICAL_SYNONYMS = {
    "low bp": "hypotension",
    "low blood pressure": "hypotension",
    "high bp": "hypertension",
    "heart racing": "tachycardia",
    "fast pulse": "tachycardia",
    "slow pulse": "bradycardia",
    "infection spreading": "sepsis",
    "blood infection": "sepsis",
    "patient crashing": "septic shock",
    "oxygen low": "hypoxia",
    "breathing fast": "tachypnea",
    "very sick from infection": "septic shock",
}


def normalize_query(query):
    query_lower = query.lower()
    expanded_terms = []

    for casual, medical in MEDICAL_SYNONYMS.items():
        if casual in query_lower:
            expanded_terms.append(medical)

    if expanded_terms:
        query = query + " " + " ".join(expanded_terms)

    return query


# ==============================
# INTENT CLASSIFICATION
# ==============================


def classify_intent(query):
    prompt = f"""
You are a strict medical triage classifier.

Classify the user query into ONE category only:

1. Educational
2. Clinical Non-Emergency
3. Emergency Critical

Rules:
- If asking explanation/theory → Educational
- If describing patient but stable → Clinical Non-Emergency
- If describing unstable active patient → Emergency Critical

Return ONLY the category name.

Query:
{query}
"""

    try:
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={"model": "phi3:mini", "prompt": prompt, "stream": False},
            timeout=30,
        )

        classification = response.json()["response"].strip()

        if "Emergency" in classification:
            return "Emergency Critical"
        elif "Educational" in classification:
            return "Educational"
        else:
            return "Clinical Non-Emergency"

    except Exception:
        return "Clinical Non-Emergency"


# ==============================
# MEDICAL AGENT
# ==============================


def medical_agent(query, index, chunks, metadata):

    print("\n🧠 Agent analyzing query...")

    # 🔹 Step 0: Normalize query for better retrieval
    normalized_query = normalize_query(query)

    # 🔹 Step 1: Intent Classification
    intent = classify_intent(query)
    triage_level = classify_triage(query)

    print(f"🧠 Intent: {intent}")
    print(f"🚑 TRIAGE LEVEL: {triage_level}")

    if triage_level == "RED":
        retrieval_k = 15
        is_emergency = True

    elif triage_level == "YELLOW":
        retrieval_k = 10
        is_emergency = False

    else:
        retrieval_k = 6
        is_emergency = False

    if is_emergency:
        print("🚨 Emergency pattern detected.")

    # 🔹 Step 2: Retrieval (use normalized query here)
    results, sources, confidence = search(
        normalized_query, index, chunks, metadata, top_k=retrieval_k
    )

    context = "\n\n".join(results[:5])

    # 🔹 Step 3: Generation (use original human query)
    answer = generate_answer(context, query)

    # 🔹 Step 4: Faithfulness Evaluation
    faithfulness = calculate_faithfulness(answer, context)

    # 🔹 Step 5: Safety Check
    safety_flag = False

    if confidence < 0.4:
        print("⚠ Low semantic similarity detected.")
        safety_flag = True

    if faithfulness < 35:
        print("⚠ Low faithfulness score detected.")
        safety_flag = True

    # 🔹 Optional fallback message
    if safety_flag:
        answer += "\n\n⚠ The answer may not be fully grounded in the retrieved guidelines. Please verify clinically."

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "faithfulness": faithfulness,
        "safety_flag": safety_flag,
        "emergency": is_emergency,
        "triage_level": triage_level,
    }
