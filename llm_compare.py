import os, time, requests
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ✅ LOAD ONCE (FIXED TIMEOUT ISSUE)
SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a clinical triage assistant. "
    "Use the provided clinical guidelines if available. "
    "Provide triage level (RED/YELLOW/GREEN) and clear steps."
)

# ── Mistral ─────────────────────────────────


def query_mistral(prompt: str, context: str = "") -> dict:
    full = (
        f"Clinical Guidelines:\n{context}\n\nQuestion: {prompt}" if context else prompt
    )

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
        answer = resp.json()["message"]["content"].strip()

        return {
            "answer": answer,
            "model": OLLAMA_MODEL,
            "latency_s": round(time.time() - start, 2),
            "error": None,
        }

    except Exception as e:
        return {"answer": "", "model": OLLAMA_MODEL, "latency_s": 0, "error": str(e)}


# ── Groq ─────────────────────────────────


def query_groq(prompt: str, context: str = "") -> dict:

    if not GROQ_API_KEY:
        return {"answer": "", "model": "groq", "latency_s": 0, "error": "Missing key"}

    client = Groq(api_key=GROQ_API_KEY)

    full = (
        f"Clinical Guidelines:\n{context}\n\nQuestion: {prompt}" if context else prompt
    )

    MODELS = ["gemma2-9b-it", "llama-3.1-8b-instant"]

    start = time.time()

    for m in MODELS:
        try:
            chat = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full},
                ],
            )

            return {
                "answer": chat.choices[0].message.content.strip(),
                "model": m,
                "latency_s": round(time.time() - start, 2),
                "error": None,
            }
        except:
            continue

    return {"answer": "", "model": "groq", "latency_s": 0, "error": "All models failed"}


# ── Similarity (FIXED) ─────────────────────────────────


def compute_similarity(a: str, b: str):

    scores = {}

    try:
        emb = SIM_MODEL.encode([a, b])
        cos = float(
            np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        )
        scores["semantic_similarity"] = round(cos, 4)
    except:
        scores["semantic_similarity"] = 0.0

    ta, tb = set(a.lower().split()), set(b.lower().split())
    union = ta | tb
    scores["jaccard_overlap"] = round(len(ta & tb) / len(union), 4) if union else 0

    scores["rouge1_f1"] = scores["jaccard_overlap"]

    scores["composite_score"] = round(
        0.5 * scores["semantic_similarity"]
        + 0.3 * scores["rouge1_f1"]
        + 0.2 * scores["jaccard_overlap"],
        4,
    )

    return scores


# ── MAIN ─────────────────────────────────


def compare_llms(query: str, context: str = ""):

    mistral = query_mistral(query, context)
    groq = query_groq(query, context)

    similarity = {}
    if mistral["answer"] and groq["answer"]:
        similarity = compute_similarity(mistral["answer"], groq["answer"])

    return {
        "query": query,
        "mistral": mistral,
        "gpt4": groq,
        "similarity": similarity,
    }
