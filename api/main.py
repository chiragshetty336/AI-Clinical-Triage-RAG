from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
from rag.vitals_triage import calculate_vital_triage
from rag.config import CACHE_PATH, INDEX_PATH
from rag.agent import medical_agent
from rag.indexing import load_index

app = FastAPI(title="Medical RAG System")

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
index = None
chunks = []
metadata = []


# ----------------------------
# LOAD INDEX + METADATA
# ----------------------------
def load_rag_components():
    global index, chunks, metadata

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

    chunks = metadata_store["chunks"]
    metadata = metadata_store["metadata"]

    print("✅ RAG components loaded successfully.")


# Load at startup
load_rag_components()


# ----------------------------
# REQUEST MODEL
# ----------------------------
class QueryRequest(BaseModel):
    symptoms: str

    heart_rate: int | None = None
    oxygen: int | None = None
    temperature: float | None = None
    systolic_bp: int | None = None


# ----------------------------
# MEDICAL QUERY ENDPOINT
# ----------------------------
@app.post("/query")
def query_rag(request: QueryRequest):

    if index is None:
        return {"error": "Index not ready. Run Airflow ingestion first."}

    triage_vitals = calculate_vital_triage(
        request.heart_rate, request.oxygen, request.temperature, request.systolic_bp
    )

    result = medical_agent(request.symptoms, index, chunks, metadata)

    # override triage if vitals critical
    if triage_vitals == "RED":
        result["triage_level"] = "RED"

    return {
        "answer": result["answer"],
        "vital_triage": triage_vitals,
        "triage_level": result["triage_level"],
        "sources": result["sources"][:3],
        "confidence_score": round(result["confidence"], 3),
        "faithfulness_score": result["faithfulness"],
        "emergency_detected": result["emergency"],
        "safety_flag": result["safety_flag"],
    }


# ----------------------------
# AIRFLOW LOG ANALYZER
# ----------------------------
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
    import requests

    prompt = f"""
You are a DevOps and Airflow expert.

Analyze the following Airflow task failure log.

1. Identify the root cause.
2. Explain it simply.
3. Provide exact steps to fix it.

LOG:
------------------
{error_text}
------------------
"""

    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "phi3:mini", "prompt": prompt, "stream": False},
    )

    return response.json()["response"]


@app.get("/analyze-dag")
def analyze_dag():
    latest_log = get_latest_log_file()

    if not latest_log:
        return {"error": "No log files found."}

    with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    error_section = extract_error_section(log_text)

    analysis = analyze_log_with_llm(error_section)

    return {"log_file": latest_log, "analysis": analysis}
