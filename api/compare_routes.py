"""
api/compare_routes.py
FIXED: RAG + Benchmark + Proper loading
"""

import sys, os, pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from llm_compare import compare_llms, compute_similarity

# ✅ RAG
from rag.agent import medical_agent
from rag.indexing import load_index

router = APIRouter(prefix="/compare", tags=["LLM Comparison"])

# ── LOAD EVERYTHING CORRECTLY ─────────────────────────

index = load_index()  # ✅ only index

# ✅ LOAD CHUNKS + METADATA (VERY IMPORTANT)
with open("data/embeddings_cache/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 🔥 fallback: use metadata as chunks
chunks = metadata

print("✅ RAG Loaded: index + chunks + metadata")

# ── BASE ANSWERS (UNCHANGED) ─────────────────────────

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

# ── REQUEST MODELS ─────────────────────────


class CompareRequest(BaseModel):
    query: str
    context: Optional[str] = ""


class BaseScoreRequest(BaseModel):
    query_index: int
    mistral_answer: str
    groq_answer: str


# ── MAIN COMPARE (RAG ENABLED) ─────────────────────


@router.post("/")
async def compare_endpoint(request: CompareRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # 🔥 STEP 1: RAG AGENT
    try:
        rag_result = medical_agent(request.query, index, chunks, metadata)
        context = rag_result.get("answer", "")
    except Exception as e:
        print("❌ RAG ERROR:", e)
        context = ""

    # 🔥 STEP 2: LLM COMPARISON (WITH CONTEXT)
    return compare_llms(request.query, context)


# ── BASE SCORING ─────────────────────────


@router.post("/score-against-base")
async def score_against_base(request: BaseScoreRequest):

    idx = request.query_index

    if idx < 0 or idx >= len(BASE_ANSWERS):
        raise HTTPException(
            status_code=400, detail=f"query_index must be 0 to {len(BASE_ANSWERS)-1}"
        )

    base = BASE_ANSWERS[idx]

    return {
        "mistral_vs_base": compute_similarity(base, request.mistral_answer),
        "groq_vs_base": compute_similarity(base, request.groq_answer),
    }


# ── HEALTH CHECK ─────────────────────────


@router.get("/health")
async def health_check():
    return {"status": "ok"}
