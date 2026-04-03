from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
import pickle
import requests

from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

from rag.vitals_triage import calculate_vital_triage
from rag.config import CACHE_PATH, INDEX_PATH, MODEL_NAME

from rag.indexing import load_index
from rag.cache_db import search_cache, store_cache

try:
    from rag.agent import medical_agent
except Exception as e:
    print("❌ ERROR loading rag.agent:", e)

try:
    from evaluation.evaluate import evaluate_answers
except Exception as e:
    print("❌ ERROR loading evaluation:", e)

from typing import Optional

app = FastAPI(title="Medical RAG System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.compare_routes import router as compare_router

app.include_router(compare_router)

# ------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------
index = None
chunks = []
metadata = []

model = SentenceTransformer(MODEL_NAME)


# ------------------------------------------------
# LOAD INDEX + METADATA
# ------------------------------------------------
def load_rag_components():
    global index, chunks, metadata

    try:
        if not os.path.exists(INDEX_PATH):
            print("❌ FAISS index not found. Run ingestion first.")
            return

        print("✅ Loading FAISS index...")
        index = load_index()

        metadata_path = os.path.join(CACHE_PATH, "metadata.pkl")

        if not os.path.exists(metadata_path):
            print("❌ Metadata file not found.")
            return

        with open(metadata_path, "rb") as f:
            metadata_store = pickle.load(f)

        chunks = metadata_store.get("chunks", [])
        metadata = metadata_store.get("metadata", [])

        print(f"✅ Loaded {len(chunks)} document chunks.")

    except Exception as e:
        print("🚨 Failed to load RAG components:", str(e))


load_rag_components()


# ------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    heart_rate: Optional[int] = None
    oxygen: Optional[int] = None
    temperature: Optional[float] = None
    systolic_bp: Optional[int] = None


# ------------------------------------------------
# MAIN ENDPOINT
# ------------------------------------------------
@app.post("/query")
def query_rag(request: QueryRequest):

    if index is None:
        return {"error": "Index not ready. Run ingestion first."}

    try:
        print("\n🔎 Incoming Query:", request.question)

        # 1️⃣ Embedding
        query_embedding = model.encode([request.question])[0]

        # 2️⃣ Cache
        cached = search_cache(request.question, query_embedding)

        if cached:
            print("⚡ Cache hit")
            return {
                "triage_level": cached.get("triage_level", "GREEN"),
                "vital_triage": "UNKNOWN",
                "admission": "Cached",
                "priority": "Cached result",
                "recommended_action": "Refer to cached answer",
                "answer": cached.get("answer", ""),
                "sources": cached.get("sources", []),
                "confidence_score": 0.5,
                "faithfulness_score": 50,
                "emergency_detected": False,
                "safety_flag": False,
                "cached": True,
            }

        # 3️⃣ Vital triage
        triage_vitals = calculate_vital_triage(
            heart_rate=request.heart_rate,
            oxygen=request.oxygen,
            temperature=request.temperature,
            systolic_bp=request.systolic_bp,
        )

        # 4️⃣ RAG pipeline
        result = medical_agent(request.question, index, chunks, metadata)

        if result is None:
            return {"error": "RAG pipeline failed."}

        # 5️⃣ Merge triage
        final_triage = result.get("triage_level", "GREEN")
        if triage_vitals == "RED":
            final_triage = "RED"

        vital_triage = final_triage

        # 6️⃣ Store cache
        store_cache(
            request.question,
            query_embedding,
            result.get("answer", ""),
            final_triage,
            result.get("sources", []),
        )

        # 7️⃣ Evaluation (FIXED)
        try:
            evaluation = evaluate_answers(request.question, result.get("answer", ""))
            reference_answer = evaluation.get("reference_answer", "")  # ✅ FIXED
            similarity = round(evaluation.get("similarity_score", 0), 3)
        except Exception as e:
            print("⚠ Evaluation failed:", e)
            reference_answer = ""
            similarity = 0

        # 8️⃣ FINAL RESPONSE (FIXED)
        return {
            "triage_level": final_triage,
            "vital_triage": vital_triage,
            "admission": result.get("admission", "Unknown"),
            "priority": result.get("priority", "Unknown"),
            "recommended_action": result.get("recommended_action", ""),
            "answer": result.get("answer", ""),
            "reference_answer": reference_answer,  # ✅ FIXED
            "similarity_score": similarity,
            "sources": result.get("sources", [])[:3],
            "confidence_score": result.get("confidence_score", 0.5),
            "faithfulness_score": result.get("faithfulness_score", 50),
            "emergency_detected": result.get("emergency_detected", False),
            "safety_flag": result.get("safety_flag", False),
            "cached": False,
        }

    except Exception as e:
        print("\n🚨 API ERROR:", str(e))
        return {"error": str(e)}


# ------------------------------------------------
# AIRFLOW LOG ANALYZER (UNCHANGED)
# ------------------------------------------------
AIRFLOW_LOGS_PATH = "/opt/airflow/logs"


def get_latest_log_file():
    latest_log = None
    latest_time = 0

    for root, dirs, files in os.walk(AIRFLOW_LOGS_PATH):
        for file in files:
            if file.endswith(".log"):
                full_path = os.path.join(root, file)
                modified_time = os.path.getmtime(full_path)

                if modified_time > latest_time:
                    latest_time = modified_time
                    latest_log = full_path

    return latest_log


def extract_error_section(log_text):
    if "Traceback" in log_text:
        return log_text.split("Traceback")[-1]
    return log_text[-2000:]


def analyze_log_with_llm(error_text):

    prompt = f"""
You are a DevOps and Airflow expert.

Analyze the following Airflow task failure log.

1. Identify the root cause
2. Explain simply
3. Give exact fix steps

LOG:
{error_text}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi3", "prompt": prompt, "stream": False},
        timeout=60,
    )

    return response.json().get("response", "No response")


@app.get("/analyze-dag")
def analyze_dag():
    try:
        latest_log = get_latest_log_file()

        if not latest_log:
            return {"error": "No log files found."}

        with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()

        error_section = extract_error_section(log_text)
        analysis = analyze_log_with_llm(error_section)

        return {
            "log_file": latest_log,
            "analysis": analysis,
        }

    except Exception as e:
        return {"error": str(e)}
