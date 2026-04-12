"""
llm_compare.py — UPDATED
3-Model Comparison: Mistral vs Groq vs MedAlpaca
MedAlpaca runs locally via Ollama (same as Mistral)

WHY MedAlpaca?
- Medical-domain fine-tuned (Stanford Alpaca base + medical instruction data)
- Runs locally via Ollama — no extra API key needed
- Directly comparable to Mistral (same Ollama interface)
- Stronger clinical reasoning than general Falcon/ClinicalBERT for triage Q&A
  * Falcon is a base LLM (not fine-tuned for medical Q&A)
  * ClinicalBERT is an encoder-only model — cannot generate text responses
  * MedAlpaca is instruction-tuned for medical question answering = best fit
"""

import os, time, requests
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "mistral:7b")
MEDALPACA_MODEL   = os.getenv("MEDALPACA_MODEL", "medalpaca:7b")   # ← NEW
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

# Load once (avoid repeated init overhead)
SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a clinical triage assistant. "
    "Use the provided clinical guidelines if available. "
    "Provide triage level (RED/YELLOW/GREEN) and clear steps."
)


# ─────────────────────────────────────────────────────────────
# HELPER: call Ollama-hosted model (shared by Mistral & MedAlpaca)
# ─────────────────────────────────────────────────────────────

def _query_ollama(model_name: str, prompt: str, context: str = "") -> dict:
    """Generic Ollama caller used by both Mistral and MedAlpaca."""
    full = (
        f"Clinical Guidelines:\n{context}\n\nQuestion: {prompt}"
        if context else prompt
    )
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": full},
                ],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"].strip()
        return {
            "answer":    answer,
            "model":     model_name,
            "latency_s": round(time.time() - start, 2),
            "error":     None,
        }
    except Exception as e:
        return {
            "answer":    "",
            "model":     model_name,
            "latency_s": 0,
            "error":     str(e),
        }


# ─────────────────────────────────────────────────────────────
# MODEL 1: Mistral (via Ollama, local)
# ─────────────────────────────────────────────────────────────

def query_mistral(prompt: str, context: str = "") -> dict:
    return _query_ollama(OLLAMA_MODEL, prompt, context)


# ─────────────────────────────────────────────────────────────
# MODEL 2: Groq (cloud API)
# ─────────────────────────────────────────────────────────────

def query_groq(prompt: str, context: str = "") -> dict:
    if not GROQ_API_KEY:
        return {"answer": "", "model": "groq", "latency_s": 0, "error": "Missing GROQ_API_KEY"}

    client = Groq(api_key=GROQ_API_KEY)
    full   = (
        f"Clinical Guidelines:\n{context}\n\nQuestion: {prompt}"
        if context else prompt
    )

    # Try available Groq models in order
    MODELS = ["gemma2-9b-it", "llama-3.1-8b-instant"]
    start  = time.time()

    for m in MODELS:
        try:
            chat = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": full},
                ],
            )
            return {
                "answer":    chat.choices[0].message.content.strip(),
                "model":     m,
                "latency_s": round(time.time() - start, 2),
                "error":     None,
            }
        except Exception:
            continue

    return {"answer": "", "model": "groq", "latency_s": 0, "error": "All Groq models failed"}


# ─────────────────────────────────────────────────────────────
# MODEL 3 (NEW): MedAlpaca (via Ollama, local)
# ─────────────────────────────────────────────────────────────

def query_medalpaca(prompt: str, context: str = "") -> dict:
    """
    MedAlpaca: medical instruction-tuned LLM running locally via Ollama.
    Pull command:  ollama pull medalpaca:7b
    """
    return _query_ollama(MEDALPACA_MODEL, prompt, context)


# ─────────────────────────────────────────────────────────────
# SCORING UTILITIES (unchanged)
# ─────────────────────────────────────────────────────────────

def compute_similarity(a: str, b: str) -> dict:
    scores = {}

    # Semantic similarity (cosine)
    try:
        emb = SIM_MODEL.encode([a, b])
        cos = float(
            np.dot(emb[0], emb[1])
            / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        )
        scores["semantic_similarity"] = round(cos, 4)
    except Exception:
        scores["semantic_similarity"] = 0.0

    # Jaccard / ROUGE-1 proxy
    ta, tb = set(a.lower().split()), set(b.lower().split())
    union = ta | tb
    scores["jaccard_overlap"] = round(len(ta & tb) / len(union), 4) if union else 0
    scores["rouge1_f1"]       = scores["jaccard_overlap"]

    # Composite
    scores["composite_score"] = round(
        0.5 * scores["semantic_similarity"]
        + 0.3 * scores["rouge1_f1"]
        + 0.2 * scores["jaccard_overlap"],
        4,
    )
    return scores


# ─────────────────────────────────────────────────────────────
# MAIN COMPARISON FUNCTION (now returns 3 models)
# ─────────────────────────────────────────────────────────────

def compare_llms(query: str, context: str = "") -> dict:
    """
    Run the same query against all three models and return
    answers + cross-model similarity scores.
    """
    mistral    = query_mistral(query, context)
    groq       = query_groq(query, context)
    medalpaca  = query_medalpaca(query, context)

    # Cross-model similarity pairs
    sim_m_vs_g  = compute_similarity(mistral["answer"],   groq["answer"])    if mistral["answer"] and groq["answer"]   else {}
    sim_m_vs_ma = compute_similarity(mistral["answer"],   medalpaca["answer"]) if mistral["answer"] and medalpaca["answer"] else {}
    sim_g_vs_ma = compute_similarity(groq["answer"],      medalpaca["answer"]) if groq["answer"]    and medalpaca["answer"] else {}

    return {
        "query":      query,
        "mistral":    mistral,
        "groq":       groq,          # key kept as "gpt4" below for backward compat with dashboard
        "gpt4":       groq,          # ← backward-compat alias (dashboard uses this key)
        "medalpaca":  medalpaca,     # ← NEW third model
        "similarity": {
            "mistral_vs_groq":      sim_m_vs_g,
            "mistral_vs_medalpaca": sim_m_vs_ma,
            "groq_vs_medalpaca":    sim_g_vs_ma,
        },
    }


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = (
        "A 52-year-old male with crushing chest pain, diaphoresis, "
        "BP 90/60, HR 110. Had MI 10 days ago. Triage level and immediate actions?"
    )
    result = compare_llms(test_query)

    for model_key in ("mistral", "groq", "medalpaca"):
        m = result[model_key]
        print(f"\n{'='*60}")
        print(f"MODEL: {m['model']}  |  Latency: {m['latency_s']}s")
        if m["error"]:
            print(f"ERROR: {m['error']}")
        else:
            print(m["answer"][:400])

    print("\n--- Cross-model similarity ---")
    for pair, scores in result["similarity"].items():
        print(f"{pair}: composite={scores.get('composite_score','N/A')}")
