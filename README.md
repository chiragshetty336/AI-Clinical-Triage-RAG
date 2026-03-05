# AI Clinical Triage RAG System

## Project Overview

This project is an **AI-powered Clinical Decision Support System** built
using a **Retrieval-Augmented Generation (RAG)** architecture. It
analyzes patient symptoms and vital signs, retrieves relevant medical
guidelines, and generates structured clinical recommendations.

The system combines **medical document retrieval**, **LLM reasoning**,
and **clinical triage logic** to assist in evaluating patient
conditions.

------------------------------------------------------------------------

# System Architecture

User Input  
↓  
FastAPI API  
↓  
Vital Signs Triage Engine  
↓  
Medical Agent  
↓  
FAISS Vector Retrieval  
↓  
LLM Clinical Reasoning (Ollama)  
↓  
Safety & Faithfulness Evaluation  
↓  
Clinical Recommendation

------------------------------------------------------------------------

# Key Features Implemented

## 1. Medical RAG Pipeline

The system uses **Retrieval-Augmented Generation** to answer medical
queries.

Steps: 1. Medical PDFs are ingested 2. Text is chunked 3. BioBERT
embeddings are generated 4. Embeddings are stored in a FAISS index 5.
Relevant context is retrieved for user queries

------------------------------------------------------------------------

## 2. Clinical Triage Classification

| Level  | Meaning                    |
|--------|----------------------------|
| RED    | Life‑threatening condition |
| YELLOW | Urgent but stable          |
| GREEN  | Mild condition             |

------------------------------------------------------------------------

## 3. Vital Signs Triage Engine

Evaluates patient vital signs:

-   Heart Rate
-   Oxygen Saturation
-   Temperature
-   Systolic Blood Pressure

Example request:

{ “symptoms”: “breathing difficulty”, “heart_rate”: 130, “oxygen”: 85,
“temperature”: 39, “systolic_bp”: 90 }

------------------------------------------------------------------------

## 4. Medical Agent Pipeline

The AI agent performs:

1.  Query normalization
2.  Intent classification
3.  Triage classification
4.  FAISS document retrieval
5.  LLM-based medical reasoning
6.  Faithfulness evaluation
7.  Safety flagging

------------------------------------------------------------------------

## 5. Safety Evaluation System

Two evaluation metrics:

**Confidence Score** Semantic similarity between query and retrieved
documents.

**Faithfulness Score** Checks whether the generated answer is grounded
in retrieved context.

------------------------------------------------------------------------

## 6. FastAPI Clinical API

Endpoint:

POST /query

Example Response:

{ “triage_level”: “RED”, “vital_triage”: “RED”, “confidence_score”:
0.82, “faithfulness_score”: 11.7, “emergency_detected”: true,
“safety_flag”: true }

------------------------------------------------------------------------

## 7. Airflow Data Pipeline

Pipeline tasks:

1.  detect_pdfs
2.  ingest_documents
3.  validate_index
4.  pipeline_summary

Flow:

Detect PDFs → Generate Embeddings → Update FAISS Index → Validate Index

------------------------------------------------------------------------

# Technologies Used

-   Python
-   FastAPI
-   FAISS
-   Sentence Transformers (BioBERT)
-   Ollama
-   Apache Airflow
-   Docker
-   PyMuPDF
-   NumPy

------------------------------------------------------------------------

# Project Structure

api/ rag/ dags/ data/ Dockerfile docker-compose.yaml requirements.txt

------------------------------------------------------------------------

# How to Run

1.  Clone repository git clone <repo_url>

2.  Start containers docker compose up –build

3.  Open API docs http://localhost:8000/docs

4.  Open Airflow http://localhost:8080

------------------------------------------------------------------------

# Current Capabilities

✔ Medical guideline retrieval  
✔ Clinical triage classification  
✔ Vital signs analysis  
✔ LLM medical reasoning  
✔ Safety evaluation  
✔ Automated Airflow pipeline

------------------------------------------------------------------------

# Disclaimer

This project is for **research and educational purposes only** and
should not be used for real medical decision making.
