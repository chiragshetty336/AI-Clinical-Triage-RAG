"""
api/compare_routes.py — UPDATED for 3-Model Comparison
Changes from original:
  1. Imports query_medalpaca from llm_compare
  2. /compare/ endpoint now queries all 3 models
  3. Response includes medalpaca field
  4. /compare/score-against-base now scores all 3 models
"""

import sys, os, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── UPDATED IMPORT ─────────────────────────────────────────────
from llm_compare import compare_llms, compute_similarity

from rag.agent    import medical_agent
from rag.indexing import load_index

router = APIRouter(prefix="/compare", tags=["LLM Comparison"])

# Load RAG components
index = load_index()

with open("data/embeddings_cache/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

chunks = metadata
print("✅ RAG Loaded: index + chunks + metadata (3-model mode)")

# ── BASE ANSWERS (unchanged) ─────────────────────────
BASE_ANSWERS = [
    "Chest pain radiating to arm with hypotension indicates acute coronary syndrome...",
    "Worst headache suggests hemorrhage or meningitis...",
    "Asthma exacerbation needs oxygen and bronchodilators...",
    "Child infection requires hydration and evaluation...",
    "Ankle sprain managed with RICE...",
    "Mild viral infection requires rest...",
    "Cardiac arrest requires CPR...",
    "Diabetes high glucose requires insulin...",
]

# ── REQUEST MODELS ────────────────────────────────────
class CompareRequest(BaseModel):
    query:   str
    context: Optional[str] = ""


class BaseScoreRequest(BaseModel):
    query_index:      int
    mistral_answer:   str
    groq_answer:      str
    medalpaca_answer: Optional[str] = ""   # ← NEW


# ── MAIN COMPARE (RAG + 3 MODELS) ────────────────────

@router.post("/")
async def compare_endpoint(request: CompareRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Step 1: RAG context
    try:
        rag_result = medical_agent(request.query, index, chunks, metadata)
        context    = rag_result.get("answer", "")
    except Exception as e:
        print("❌ RAG ERROR:", e)
        context = ""

    # Step 2: 3-model comparison with RAG context
    return compare_llms(request.query, context)


# ── BASE SCORING (updated to include MedAlpaca) ──────

@router.post("/score-against-base")
async def score_against_base(request: BaseScoreRequest):

    idx = request.query_index
    if idx < 0 or idx >= len(BASE_ANSWERS):
        raise HTTPException(
            status_code=400,
            detail=f"query_index must be 0 to {len(BASE_ANSWERS)-1}"
        )

    base = BASE_ANSWERS[idx]

    result = {
        "mistral_vs_base":   compute_similarity(base, request.mistral_answer),
        "groq_vs_base":      compute_similarity(base, request.groq_answer),
    }

    # Score MedAlpaca if answer provided
    if request.medalpaca_answer:
        result["medalpaca_vs_base"] = compute_similarity(base, request.medalpaca_answer)

    return result


# ── HEALTH CHECK ─────────────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "ok", "models": ["mistral", "groq", "medalpaca"]}
