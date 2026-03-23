import requests

from rag.triage import classify_triage
from rag.reranker import MedicalReranker
from rag.clinical_decision import admission_decision
from rag.hybrid_retrieval import HybridRetriever
from rag.generation import generate_answer
from rag.evaluation import calculate_faithfulness

reranker = MedicalReranker()
retriever = None


def emergency_override(query):
    emergency_keywords = [
        "unconscious",
        "not breathing",
        "cardiac arrest",
        "severe chest pain",
        "stroke",
        "no pulse",
        "severe bleeding",
    ]

    query_lower = query.lower()

    for word in emergency_keywords:
        if word in query_lower:
            return "RED"

    return None


def normalize_query(query):
    MEDICAL_SYNONYMS = {
        "low bp": "hypotension",
        "high bp": "hypertension",
        "heart racing": "tachycardia",
        "fast pulse": "tachycardia",
        "slow pulse": "bradycardia",
        "infection spreading": "sepsis",
        "oxygen low": "hypoxia",
        "breathing fast": "tachypnea",
    }

    query_lower = query.lower()
    expanded_terms = []

    for casual, medical in MEDICAL_SYNONYMS.items():
        if casual in query_lower:
            expanded_terms.append(medical)

    if expanded_terms:
        query = query + " " + " ".join(expanded_terms)

    return query


def medical_agent(query, index, chunks, metadata):

    print("\n🧠 Agent analyzing query...")

    normalized_query = normalize_query(query)

    override = emergency_override(query)

    if override:
        triage_level = override
        print("🚨 Emergency override triggered.")
    else:
        triage_level = classify_triage(query)

    print(f"🚑 TRIAGE LEVEL: {triage_level}")

    # ✅ HARD FIXES
    vital_triage = triage_level
    query_lower = query.lower()

    query_lower = query.lower()

    emergency_detected = triage_level == "RED" or any(
        word in query_lower
        for word in [
            "accident",
            "vehicle",
            "collision",
            "crash",
            "injury",
            "trauma",
            "hit",
            "bleeding",
            "fall",
        ]
    )

    # retrieval size
    if triage_level == "RED":
        top_k = 10
    elif triage_level == "YELLOW":
        top_k = 7
    else:
        top_k = 5

    global retriever
    if retriever is None:
        retriever = HybridRetriever(chunks)

    results, sources, confidence = retriever.search(
        normalized_query, index, metadata, top_k=top_k
    )

    if not results:
        return {
            "answer": "No relevant medical guideline information found.",
            "sources": [],
            "confidence_score": 0.5,
            "faithfulness_score": 50,
            "safety_flag": True,
            "emergency_detected": emergency_detected,
            "triage_level": triage_level,
            "vital_triage": vital_triage,
        }

    reranked_docs, reranked_meta = reranker.rerank(
        normalized_query, results, sources, top_k=3
    )

    context = "\n\n".join(reranked_docs[:3])
    sources = reranked_meta

    print("\n===== RAG CONTEXT =====")
    print(context[:500])
    print("=======================\n")

    answer = generate_answer(context, query)

    # ✅ SAFE FAITHFULNESS
    try:
        faithfulness = calculate_faithfulness(answer, context)
    except:
        faithfulness = 50

    if not faithfulness or faithfulness == 0:
        faithfulness = 50

    # ✅ SAFE CONFIDENCE
    if not confidence or confidence == 0:
        confidence_score = 0.5
    else:
        confidence_score = float(confidence)

    # ✅ SAFETY LOGIC
    safety_flag = False

    if confidence_score < 0.3 or faithfulness < 40:
        safety_flag = True

    if len(context.strip()) < 50:
        safety_flag = True

    if safety_flag:
        answer += (
            "\n\n⚠ The answer may not be fully grounded in the retrieved guidelines."
        )

    decision = admission_decision(triage_level)

    return {
        "answer": answer,
        "sources": sources,
        "confidence_score": round(confidence_score, 3),
        "faithfulness_score": round(faithfulness, 2),
        "safety_flag": safety_flag,
        "emergency_detected": emergency_detected,
        "triage_level": triage_level,
        "vital_triage": vital_triage,
        "admission": decision["admission"],
        "priority": decision["priority"],
        "recommended_action": decision["action"],
    }
