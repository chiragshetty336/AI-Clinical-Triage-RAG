"""
llm_compare.py — Mistral (local) + Groq (free API)
"""

import os, time, requests
from dotenv import load_dotenv
from groq import Groq  # ✅ ADDED

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # ✅ ADDED

SYSTEM_PROMPT = (
    "You are a clinical triage assistant. "
    "Provide a clear structured clinical recommendation. "
    "State the triage level (RED / YELLOW / GREEN) and recommended next steps."
)

# ── Mistral via Ollama (UNCHANGED) ────────────────────────────────────────────


def query_mistral(prompt: str, context: str = "") -> dict:
    full = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            answer = data.get("response", "").strip()
        return {
            "answer": answer,
            "model": OLLAMA_MODEL,
            "latency_s": round(time.time() - start, 2),
            "error": None,
        }
    except Exception:
        try:
            resp2 = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{full}",
                    "stream": False,
                },
                timeout=120,
            )
            resp2.raise_for_status()
            answer = resp2.json().get("response", "").strip()
            return {
                "answer": answer,
                "model": OLLAMA_MODEL,
                "latency_s": round(time.time() - start, 2),
                "error": None,
            }
        except Exception as e2:
            return {
                "answer": "",
                "model": OLLAMA_MODEL,
                "latency_s": 0,
                "error": str(e2),
            }


# ── Groq (REPLACES GEMINI) ───────────────────────────────────────────────────


def query_gpt4(prompt: str, context: str = "") -> dict:
    """Groq with automatic working model fallback"""

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return {
            "answer": "",
            "model": "groq",
            "latency_s": 0,
            "error": "GROQ_API_KEY missing",
        }

    from groq import Groq

    client = Groq(api_key=key)

    full = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n{full}"
    start = time.time()

    # ✅ ALWAYS TRY CURRENT WORKING MODELS
    MODELS = [
        "gemma2-9b-it",  # ✅ most stable right now
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
    ]

    for model_name in MODELS:
        try:
            chat = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
            )

            answer = chat.choices[0].message.content.strip()

            return {
                "answer": answer,
                "model": model_name,
                "latency_s": round(time.time() - start, 2),
                "error": None,
            }

        except Exception as e:
            continue  # try next model

    return {
        "answer": "",
        "model": "groq",
        "latency_s": 0,
        "error": "All Groq models failed or deprecated",
    }


# ── Similarity (UNCHANGED) ────────────────────────────────────────────────────


def compute_similarity(text_a: str, text_b: str) -> dict:
    scores = {}

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        m = SentenceTransformer("all-MiniLM-L6-v2")
        emb = m.encode([text_a, text_b])
        cos = float(
            np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        )
        scores["semantic_similarity"] = round(cos, 4)
    except Exception as e:
        scores["semantic_similarity"] = None
        scores["semantic_error"] = str(e)

    ta, tb = set(text_a.lower().split()), set(text_b.lower().split())
    union = ta | tb
    scores["jaccard_overlap"] = round(len(ta & tb) / len(union), 4) if union else 0.0

    try:
        from rouge_score import rouge_scorer as rs

        sc = rs.RougeScorer(["rouge1"], use_stemmer=True)
        scores["rouge1_f1"] = round(sc.score(text_a, text_b)["rouge1"].fmeasure, 4)
    except Exception:
        ref = text_a.lower().split()
        hyp = text_b.lower().split()
        rs2 = set(ref)
        m2 = sum(1 for w in hyp if w in rs2)
        p = m2 / len(hyp) if hyp else 0
        r = m2 / len(ref) if ref else 0
        scores["rouge1_f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    def get_level(t):
        u = t.upper()
        for lv in ["RED", "YELLOW", "GREEN"]:
            if lv in u:
                return lv
        return "UNKNOWN"

    la, lb = get_level(text_a), get_level(text_b)
    scores["triage_level_a"] = la
    scores["triage_level_b"] = lb
    scores["triage_agreement"] = la == lb and la != "UNKNOWN"

    wa, wb = len(text_a.split()), len(text_b.split())
    scores["length_ratio"] = round(min(wa, wb) / max(wa, wb), 4) if max(wa, wb) else 1.0

    sem = scores.get("semantic_similarity") or 0.0
    scores["composite_score"] = round(
        0.5 * sem + 0.3 * scores["rouge1_f1"] + 0.2 * scores["jaccard_overlap"], 4
    )

    return scores


# ── Main (MINOR CHANGE: now Groq used) ───────────────────────────────────────


def compare_llms(query: str, context: str = "") -> dict:
    print(f"[compare] {query[:80]}...")
    mistral = query_mistral(query, context)
    groq = query_gpt4(query, context)  # same function name, different backend

    print(f"  Mistral: {len(mistral['answer'])} chars  error={mistral['error']}")
    print(f"  Groq:    {len(groq['answer'])} chars   error={groq['error']}")

    similarity = {}
    if mistral["answer"] and groq["answer"]:
        similarity = compute_similarity(mistral["answer"], groq["answer"])

    return {
        "query": query,
        "mistral": mistral,
        "gpt4": groq,  # kept name for compatibility
        "similarity": similarity,
    }


if __name__ == "__main__":
    print("=== CHECKING GROQ KEY ===")

    key = os.getenv("GROQ_API_KEY", "NOT FOUND")
    print(f"GROQ_API_KEY = '{key[:10]}...' (length={len(key)})")
    print()

    q = "Patient has chest pain, heart rate 120, oxygen saturation 88%. What is the triage?"

    r = compare_llms(q)

    print("\n=== MISTRAL ===\n", r["mistral"]["answer"] or r["mistral"]["error"])
    print("\n=== GROQ  ===\n", r["gpt4"]["answer"] or r["gpt4"]["error"])

    print("\n=== SIMILARITY ===")
    if not r["similarity"]:
        print("Similarity not computed (one model failed)")
    else:
        for k, v in r["similarity"].items():
            print(f"  {k}: {v}")
