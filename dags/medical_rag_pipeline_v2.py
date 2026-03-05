from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from rag.ingestion import load_pdfs_with_cache
from rag.indexing import load_index
from rag.config import INDEX_PATH

import os


# -----------------------------
# TASK 1: Detect PDFs
# -----------------------------
def detect_pdfs():
    print("🔍 Checking for guideline PDFs...")

    data_path = "/opt/airflow/project/data/guidelines"

    pdfs = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

    print(f"Found {len(pdfs)} PDF files.")

    return pdfs


# -----------------------------
# TASK 2: Ingest Documents
# -----------------------------
def ingest_documents():
    print("📂 Starting document ingestion...")

    message = load_pdfs_with_cache()

    print("✅ Ingestion completed.")
    print(message)


# -----------------------------
# TASK 3: Validate FAISS Index
# -----------------------------
def validate_index():

    print("🔎 Validating FAISS index...")

    if not os.path.exists(INDEX_PATH):
        raise Exception("❌ FAISS index missing!")

    index = load_index()

    print(f"✅ Index contains {index.ntotal} vectors")


# -----------------------------
# TASK 4: Pipeline Summary
# -----------------------------
def pipeline_summary():

    print("📊 Pipeline finished successfully.")

    print(
        """
Pipeline Steps Completed:

1️⃣ PDF Detection
2️⃣ Document Ingestion
3️⃣ Embedding Generation
4️⃣ FAISS Index Update
5️⃣ Index Validation
"""
    )


# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id="medical_rag_pipeline_v2",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_detect = PythonOperator(task_id="detect_pdfs", python_callable=detect_pdfs)

    task_ingest = PythonOperator(
        task_id="ingest_documents", python_callable=ingest_documents
    )

    task_validate = PythonOperator(
        task_id="validate_index", python_callable=validate_index
    )

    task_summary = PythonOperator(
        task_id="pipeline_summary", python_callable=pipeline_summary
    )

    task_detect >> task_ingest >> task_validate >> task_summary
