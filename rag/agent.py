from rag.self_reflection import reflect_and_improve
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

    # 🔥 retrieval size
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
            "answer": "No relevant medical guideline found.",
            "triage_level": triage_level,
        }

    # 🔥 SAFE RERANK
    try:
        reranked_docs, reranked_meta = reranker.rerank(
            normalized_query, results, sources, top_k=3
        )
    except Exception as e:
        print("⚠️ Reranker failed:", e)
        reranked_docs = results[:3]
        reranked_meta = sources[:3]

    context = "\n\n".join(reranked_docs[:3])

    print("\n===== RAG CONTEXT =====")
    print(context[:500])
    print("=======================\n")

    # 🔥 STEP 1: CLEAN GENERATION

    # 🔥 reduce context influence
    simple_context = context[:300]

    answer = generate_answer(simple_context, query, triage_level)

    # 🔥 STEP 2: SELF-REFLECTION (ONLY FOR RED/YELLOW)
    if triage_level == "YELLOW":
        answer = reflect_and_improve(answer, context, query)

    # 🔥 HARD FIX: ENSURE TRIAGE CONSISTENCY
    answer = answer.replace("Triage Level: GREEN", f"Triage Level: {triage_level}")
    answer = answer.replace("Triage Level: YELLOW", f"Triage Level: {triage_level}")
    answer = answer.replace("Triage Level: RED", f"Triage Level: {triage_level}")

    decision = admission_decision(triage_level)

    return {
        "answer": answer,
        "triage_level": triage_level,
        "admission": decision["admission"],
        "priority": decision["priority"],
        "recommended_action": decision["action"],
    }
