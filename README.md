# 🏥 AI Clinical Triage RAG System

> **A production-grade Retrieval-Augmented Generation (RAG) system for emergency medical triage — comparing Mistral 7B, Groq (Llama-3.1-8B), and Meditron 7B against Australian clinical guidelines.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/Flask-3.0+-orange.svg)](https://flask.palletsprojects.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple.svg)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Benchmark Results](#-benchmark-results)
- [Quick Start](#-quick-start)
- [Step-by-Step Setup](#-step-by-step-setup)
- [How to Run](#-how-to-run)
- [File-by-File Explanation](#-file-by-file-explanation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Dashboard Guide](#-dashboard-guide)
- [Technologies Used](#-technologies-used)

---

## 🎯 What This Project Does

This system answers the question: **"Which LLM gives the best clinical triage decision when grounded in real medical guidelines?"**

Given a patient clinical scenario (e.g. *"52-year-old male, crushing chest pain, BP 90/60, prior MI"*), the system:

1. **Retrieves** the most relevant chunks from Australian Emergency Triage guidelines (PDF) using a FAISS vector index + BM25 hybrid search
2. **Feeds** the retrieved context + patient query to 3 different LLMs simultaneously
3. **Scores** each model's response against a gold-standard answer across 4 dimensions
4. **Displays** results side-by-side on a live benchmark dashboard with full metric reporting

### Three Models Compared
| Model | Provider | Speed | Clinical Quality |
|---|---|---|---|
| **Mistral 7B** | Local (Ollama) | Medium (11s) | High — consistent, never fails |
| **Groq (Llama-3.1-8B)** | Cloud API | Fast (2.3s) | Good — best faithfulness & coverage |
| **Meditron 7B** | Local (Ollama) | Slow (16s) | Highest ROUGE/BERTScore on RED cases |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEDICAL RAG PIPELINE                         │
│                                                                 │
│  PDF Guidelines ──► PyMuPDF ──► Chunking ──► BioBERT Embed     │
│                                                    │            │
│                                               FAISS Index       │
│                                                    │            │
│  Patient Query ──► Triage Classifier               │            │
│        │               │                           │            │
│        │          RED/YELLOW/GREEN             Hybrid Search    │
│        │          (top-k = 10/7/5)         (FAISS + BM25)      │
│        │                                           │            │
│        └───────────────────────────────► Context Assembly       │
│                                                    │            │
│                              ┌─────────────────────┤            │
│                              ▼                     ▼            │
│                         Mistral 7B          Groq API            │
│                         (Ollama)            (Llama-3.1)         │
│                              │                     │            │
│                         Meditron 7B ───────────────┘            │
│                         (Ollama)                                │
│                              │                                  │
│                         Self-Reflection ◄── YELLOW cases only   │
│                              │                                  │
│                         Score & Compare ◄── Gold Standard       │
│                              │                                  │
│                    Dashboard + API Response                      │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow
```
User Query
    │
    ▼
Query Expansion (medical synonyms)
    │
    ▼
BioBERT Embedding (pritamdeka/BioBERT-mnli-snli-scinli)
    │
    ▼
FAISS Search (cosine similarity) ──► BM25 Search
    │                                      │
    └──────────── Hybrid Merge (0.7 FAISS + 0.3 BM25) ──────────┘
                                    │
                              Cross-Encoder Reranker
                              (ms-marco-MiniLM-L-6-v2)
                                    │
                              Top-3 Context Chunks
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               Mistral 7B      Groq API        Meditron 7B
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                              Score vs Gold Standard
                              (4-dimension scoring)
                                    │
                              Dashboard Display
```

---

## 📁 Project Structure

```
MEDICAL_RAG/
│
├── rag/                          # Core RAG engine
│   ├── __init__.py
│   ├── agent.py                  # Main orchestrator — ties everything together
│   ├── cache_db.py               # FAQ similarity cache (BioBERT embeddings)
│   ├── clinical_decision.py      # Admission decisions (ICU/Ward/Outpatient)
│   ├── config.py                 # Paths, model names, constants
│   ├── db.py                     # PostgreSQL connection (Airflow)
│   ├── evaluation.py             # Faithfulness scoring
│   ├── faq_cache.py              # FAQ cache load/save
│   ├── generation.py             # LLM answer generation via Ollama
│   ├── hybrid_retrieval.py       # FAISS + BM25 hybrid retriever
│   ├── indexing.py               # FAISS index build & load
│   ├── ingestion.py              # PDF loading, chunking, embedding
│   ├── pipeline.py               # Airflow pipeline tasks
│   ├── reranker.py               # Cross-encoder reranker
│   ├── retrieval.py              # Pure FAISS retrieval
│   ├── self_reflection.py        # Self-critique loop for YELLOW triage
│   ├── triage.py                 # Rule-based + LLM triage classifier
│   └── vitals_triage.py          # Vital signs → triage level
│
├── api/                          # FastAPI REST endpoints
│   ├── api_main.py               # Main FastAPI app (port 8000)
│   └── compare_routes.py         # /compare/ endpoints for LLM comparison
│
├── dags/                         # Apache Airflow DAGs
│   ├── faq_cache_maintenance_dag.py   # Weekly cache cleanup
│   ├── medical_rag_dag.py             # Main pipeline DAG
│   └── medical_rag_pipeline_v2.py     # Updated pipeline DAG
│
├── evaluation/                   # Offline evaluation scripts
│   ├── benchmark.py              # 8-query benchmark runner
│   ├── compare_models.py         # Mistral vs GPT comparison
│   ├── evaluate_dataset.py       # Full dataset evaluation
│   ├── evaluate.py               # Gemini-based evaluation
│   ├── gpt_answers.json          # Reference GPT answers
│   └── queries.json              # Evaluation query set
│
├── data/
│   ├── embeddings_cache/         # FAISS index + metadata.pkl
│   ├── guidelines/               # Source PDF files (triage guidelines)
│   └── faq_cache.pkl             # Cached frequent queries
│
├── logs/                         # Airflow task logs
│
├── dashboard_v3.py               # 🌟 Main benchmark dashboard (Flask, port 5000)
├── triage_benchmark.py           # 6 gold-standard benchmark cases
├── rag_metrics.py                # M3–M19 metric calculators
├── llm_compare.py                # 3-model comparison logic
├── main.py                       # CLI entry point
├── dag_pipeline.py               # Standalone pipeline runner
├── dashboard.html                # Simple HTML dashboard
├── docker-compose.yaml           # Docker orchestration
├── Dockerfile                    # Airflow container
├── Dockerfile.api                # FastAPI container
├── requirements.txt              # Python dependencies
└── .env                          # API keys and config
```

---

## 📊 Benchmark Results

### RAG Pipeline Metrics (M3–M19)

| ID | Metric | Mistral 7B | Groq (Llama-3.1) | Meditron 7B | Winner |
|---|---|---|---|---|---|
| M3 | Retrieval Latency | 0.079s | 0.079s | 0.081s | — |
| M4 | Cosine Similarity | 0.432 | 0.432 | 0.432 | = |
| M6 | ROUGE-1 F1 | 0.286 | 0.324 | **0.364** | MA |
| M8 | ROUGE-L F1 | 0.138 | 0.144 | **0.179** | MA |
| M12 | BERTScore F1 | 0.835 | 0.834 | **0.849** | MA |
| M14 | Faithfulness | 37.4% | **43.0%** | 34.0% | G |
| M15 | GT Coverage | 34.9% | **39.5%** | 32.6% | G |
| M16 | E2E Latency | 11.19s | **2.28s** | 16.04s | G |
| M17 | Throughput | 0.089 q/s | **0.438 q/s** | 0.062 q/s | G |
| M18 | CPU Usage | **7.1%** | 8.0% | 7.8% | M |
| M19 | RAM Usage | 10.00 GB | 10.01 GB | **9.99 GB** | MA |

### Case-by-Case Scores (/100)

| Benchmark Case | Expected | Mistral | Groq | Meditron |
|---|---|---|---|---|
| Chest Pain Triage | 🔴 RED | 80.9 | 77.1 | 83.4 |
| Blunt Abdominal Trauma | 🔴 RED | 75.3 | 71.8 | 78.1 |
| Mild Pain (Lower Urgency) | 🟢 GREEN | 65.0 | **72.0** | 18.0* |
| Mental Health Emergency | 🟡 YELLOW | 46.5 | **48.5** | 42.0 |
| Sepsis Recognition | 🔴 RED | 86.0 | 77.3 | **88.7** |
| Respiratory Distress | 🔴 RED | **91.4** | 81.7 | 89.2 |

> *Meditron scored 18/100 on the GREEN case because it echoed the question instead of answering — a known prompt-engineering issue with this model on low-acuity cases.

### Summary
- **Best for speed**: Groq (2.28s E2E)
- **Best for RED/emergency cases**: Meditron (highest ROUGE, BERTScore)
- **Most reliable overall**: Mistral (never fails to produce valid triage)

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/chiragshetty336/AI-Clinical-Triage-RAG.git
cd AI-Clinical-Triage-RAG

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull LLM models (Ollama must be installed)
ollama pull mistral:7b
ollama pull meditron:7b

# 5. Set up .env file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Run ingestion (builds FAISS index from PDF guidelines)
python -c "from rag.ingestion import load_pdfs_with_cache; load_pdfs_with_cache()"

# 7. Start the dashboard
python dashboard_v3.py

# Open browser: http://localhost:5000
```

---

## 🔧 Step-by-Step Setup

### Prerequisites

| Tool | Version | Purpose | Install |
|---|---|---|---|
| Python | 3.10+ | Runtime | [python.org](https://python.org) |
| Ollama | Latest | Run local LLMs | [ollama.ai](https://ollama.ai) |
| Git | Any | Clone repo | [git-scm.com](https://git-scm.com) |
| Groq API Key | — | Cloud inference | [console.groq.com](https://console.groq.com) (free) |

### Step 1 — Install Ollama

Download from [https://ollama.ai](https://ollama.ai) and install.

Then in PowerShell:
```powershell
ollama serve          # starts Ollama in background
ollama pull mistral:7b
ollama pull meditron:7b
ollama list           # verify both appear
```

### Step 2 — Clone and Setup Python

```powershell
git clone https://github.com/chiragshetty336/AI-Clinical-Triage-RAG.git
cd "AI-Clinical-Triage-RAG"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Configure `.env`

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
MEDITRON_MODEL=meditron:7b
GEMINI_API_KEY=optional_for_evaluation
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) → Create API Key.

### Step 4 — Add PDF Guidelines

Place your PDF triage guidelines in:
```
data/guidelines/
```
The project uses the **Australian Emergency Triage Education Kit** PDF.

### Step 5 — Build the FAISS Index

```powershell
python main.py
```
This ingests all PDFs, creates embeddings, and builds the FAISS vector index. First run takes 2–5 minutes. Subsequent runs use cache.

### Step 6 — Run the Dashboard

```powershell
python dashboard_v3.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🚀 How to Run

### Option A: Benchmark Dashboard (Recommended)
```powershell
python dashboard_v3.py
# Opens at http://localhost:5000
```
Click any benchmark case in the sidebar to compare all 3 models instantly.

### Option B: FastAPI Backend Only
```powershell
uvicorn api.api_main:app --host 0.0.0.0 --port 8000 --reload
# API docs at http://localhost:8000/docs
```

### Option C: CLI Query
```powershell
python main.py
# Then type any medical question at the prompt
```

### Option D: Run Evaluation
```powershell
python evaluation/evaluate_dataset.py
```

### Option E: Docker (Full Stack)
```powershell
docker-compose up --build
# Airflow: http://localhost:8080  (admin/admin)
# API:     http://localhost:8000
```

---

## 📖 File-by-File Explanation

### Core RAG Engine (`rag/`)

#### `rag/config.py`
Central configuration file. Defines all file paths (index, cache, guidelines) and the embedding model name (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`). Every other file imports from here — change paths here, not in individual files.

#### `rag/ingestion.py`
Reads PDF files from `data/guidelines/` using PyMuPDF (fitz). Splits text into 120-word chunks, filters out non-clinical content (procedure descriptions, training modules), enforces clinical keywords (diagnosis, treatment, symptoms etc.), generates BioBERT embeddings for each chunk, and saves them to `data/embeddings_cache/metadata.pkl`. Uses caching — already-processed PDFs are skipped.

#### `rag/indexing.py`
Builds the FAISS index from embeddings using `IndexFlatIP` (inner product, equivalent to cosine similarity after L2 normalization). Saves the index to `data/embeddings_cache/medical_faiss.index`. `load_index()` reads it back on startup.

#### `rag/hybrid_retrieval.py`
The retrieval core. Given a query, it:
1. Expands the query with medical synonyms (`chest pain` → adds `acute coronary syndrome myocardial infarction`)
2. Runs FAISS semantic search (weight 0.7)
3. Runs BM25 keyword search (weight 0.3)
4. Merges scores and returns top-k chunks based on triage level (RED=10, YELLOW=7, GREEN=5)

#### `rag/reranker.py`
Takes the top-k retrieved chunks and reranks them using a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Filters chunks with score below 0.35. Returns top 3 most relevant chunks for the LLM context.

#### `rag/triage.py`
Classifies query into RED/YELLOW/GREEN using rule-based matching first (fast, no LLM call needed). Falls back to Mistral 7B via Ollama only if no keywords match. RED = life-threatening (unconscious, severe chest pain, not breathing), YELLOW = urgent (fever, vomiting), GREEN = minor (mild pain, small cut).

#### `rag/vitals_triage.py`
Converts vital signs (heart rate, oxygen saturation, temperature, blood pressure) into a triage score. RED if score ≥ 6, YELLOW if ≥ 3, GREEN otherwise. Overrides text-based triage if vitals indicate a more severe level.

#### `rag/generation.py`
Calls Mistral 7B via Ollama's REST API to generate the answer. Limits context to 1200 characters. Enforces a strict 3-4 line format with Triage Level, Reason, and What-to-do sections. Temperature 0.1 for consistency.

#### `rag/self_reflection.py`
For YELLOW triage cases only — sends the initial answer back to Mistral for self-critique and improvement. Checks for clarity, correctness, and appropriate tone. Adds one extra LLM call but improves output quality for borderline cases.

#### `rag/agent.py`
The orchestrator. Calls triage classifier → sets top_k → calls hybrid retriever → calls reranker → calls generation → optionally calls self-reflection → calls admission_decision → returns complete result dict.

#### `rag/cache_db.py`
Stores and retrieves cached query-answer pairs using embedding similarity. If a new query is within 0.85 cosine similarity of a cached query, returns cached answer immediately (no LLM call). Saves to `data/faq_cache.pkl`.

#### `rag/clinical_decision.py`
Maps triage level to admission decision: RED → ICU, YELLOW → Emergency Ward, GREEN → Outpatient. Returns admission type, priority description, and recommended action.

#### `rag/evaluation.py`
Calculates faithfulness score: what percentage of words in the model's answer appear in the retrieved context. Uses simple word overlap.

### Dashboard (`dashboard_v3.py`)
The main application file. A Flask web app serving at port 5000. Contains:
- `load_rag_components()` — loads FAISS index and metadata on startup
- `query_mistral()` — calls Ollama API for Mistral
- `query_groq()` — calls Groq cloud API (tries multiple models)
- `query_meditron()` — calls Ollama API for Meditron
- `run_benchmark()` — runs all 3 models on a benchmark case and scores them
- `run_custom_query()` — runs all 3 models on user-typed query
- `RESULT_MACRO` — Jinja2 HTML template for the 3-column result card
- `DASHBOARD_HTML` — full page HTML with sidebar, loading overlay, JS

### Benchmark Data (`triage_benchmark.py`)
Defines 6 clinical benchmark cases based on the Australian Triage Education Kit. Each case has: `query` (patient scenario), `base_answer` (gold standard with triage level, reasoning, key actions, keywords), `time_to_treatment`, and `disposition`. The `score_against_base()` function scores model responses across 4 dimensions totalling 100 points.

### Metrics (`rag_metrics.py`)
Implements M3–M19 metrics from the guide's benchmark suite:
- **M3**: Retrieval latency timer wrapper
- **M4**: Cosine similarity between query and retrieved chunk
- **M5**: Top-k keyword accuracy
- **M6**: ROUGE-1 F1 (word overlap)
- **M8**: ROUGE-L F1 (longest common subsequence)
- **M12**: BERTScore F1 (falls back to cosine if bert-score not installed)
- **M14**: Faithfulness (context word overlap in answer)
- **M15**: GT Coverage (gold answer words in response)
- **M16**: E2E latency (retrieval + generation time)
- **M17**: Throughput (queries per second)
- **M18**: CPU usage (psutil)
- **M19**: RAM usage (psutil)

### LLM Comparison (`llm_compare.py`)
Standalone comparison module used by the FastAPI routes. Defines `query_mistral()`, `query_groq()`, `query_meditron()`, and `compare_llms()` which calls all three and returns cross-model similarity scores (semantic similarity, Jaccard/ROUGE-1, composite).

### API (`api/`)

#### `api/api_main.py`
FastAPI application on port 8000. `/query` endpoint: takes patient question + optional vitals, runs full RAG pipeline, returns triage level, answer, confidence score, faithfulness score, sources, and admission decision. Includes cache check, vital triage override, and evaluation scoring.

#### `api/compare_routes.py`
FastAPI router mounted at `/compare/`. `POST /compare/` triggers all 3 LLM models with RAG context and returns their answers + similarity scores. `POST /compare/score-against-base` computes similarity against a base answer. Used by the frontend dashboard.

### Airflow DAGs (`dags/`)

#### `dags/medical_rag_dag.py`
Three-task Airflow DAG: detect PDFs → ingest documents → validate FAISS index. Runs on schedule to automatically update the index when new guideline PDFs are added to `data/guidelines/`.

#### `dags/faq_cache_maintenance_dag.py`
Weekly maintenance DAG: removes duplicate cached queries from the FAQ cache, prints cache statistics (total entries, triage distribution).

### Evaluation Scripts (`evaluation/`)

#### `evaluation/evaluate_dataset.py`
Runs the RAG pipeline on all queries in `queries.json`, compares predicted triage vs expected triage, calculates composite score, prints accuracy and average score.

#### `evaluation/compare_models.py`
Compares RAG pipeline answers vs GPT reference answers from `gpt_answers.json` using a simple keyword-based scoring function.

#### `evaluation/benchmark.py`
Runs 8 predefined benchmark queries through both Mistral and Groq, saves JSON + CSV results to `evaluation/results/`.

---

## 📏 Evaluation Metrics

### Scoring Formula (per benchmark case)
```
Composite Score /100 = 
    Triage Level Match     (30 pts) +
    Keyword Coverage       (40 pts) +
    Clinical Reasoning     (20 pts) +
    Action Specificity     (10 pts)
```

### Keyword Coverage (40 pts)
Each gold-standard case has 13–15 required clinical keywords (e.g. "resuscitation", "IV access", "ECG", "aspirin"). Score = (matched / total) × 40.

### Clinical Reasoning (20 pts)
Presence of reasoning terms: "assess", "monitor", "immediate", "indicates", "consistent with", "recommend", "administer" etc. 2 points per term, max 20.

### Action Specificity (10 pts)
+5 if response contains drug dosages/numbers (e.g. "300mg", "94%"), +5 if it mentions specific interventions (oxygen, IV, aspirin, salbutamol, ECG).

---

## 🖥 Dashboard Guide

### Benchmark Panel (Left Sidebar)
- Click any case button to run all 3 models against that case
- Cases are colour-coded: 🔴 RED, 🟡 YELLOW, 🟢 GREEN
- Previous runs appear in the History section

### Result Cards (Centre — 3 Columns)
Each column shows one model's response:
- **Score ring**: composite score /100 in a circular progress indicator
- **Triage chip**: detected triage level (RED/YELLOW/GREEN)
- **Score bars**: 4-dimension breakdown
- **Missed keywords**: what the model failed to include
- **Response**: first 600 chars of the model's answer

### Gold Standard Section
Shows the reference answer from the benchmark dataset with all required actions.

### RAG Sources Section
Shows which pages of the triage guidelines were retrieved, with relevance scores.

### Metrics Table (M3–M19)
Full pipeline metrics for that query. Winner column shows M/G/MA for each metric.

### Custom Query
Type any clinical scenario in the text box and click "Run Custom Query" to compare all 3 models in real time.

---

## 🛠 Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| **Embedding** | BioBERT (pritamdeka) | Clinical text embeddings |
| **Vector Index** | FAISS (IndexFlatIP) | Semantic similarity search |
| **Keyword Search** | BM25 (rank-bm25) | Lexical retrieval |
| **Reranker** | CrossEncoder (ms-marco-MiniLM) | Re-score retrieved chunks |
| **LLM 1** | Mistral 7B (Ollama) | Local medical reasoning |
| **LLM 2** | Groq (Llama-3.1-8B) | Cloud fast inference |
| **LLM 3** | Meditron 7B (Ollama) | Medical-domain fine-tuned LLM |
| **PDF Parsing** | PyMuPDF (fitz) | Extract text from guideline PDFs |
| **Web Framework** | Flask + FastAPI | Dashboard + REST API |
| **Pipeline** | Apache Airflow | Scheduled ingestion & maintenance |
| **Database** | PostgreSQL | Airflow metadata, cache storage |
| **Containerisation** | Docker + Compose | Reproducible deployment |
| **Metrics** | bert-score, sentence-transformers | ROUGE, BERTScore, similarity |

---

## 🔑 Environment Variables

```env
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxx        # Get free at console.groq.com

# Ollama (defaults work if Ollama installed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
MEDITRON_MODEL=meditron:7b

# Optional
GEMINI_API_KEY=                       # For Gemini-based evaluation
```

---

## 🐛 Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `ollama pull medalpaca:7b` fails | medalpaca not in Ollama registry | Use `ollama pull meditron:7b` instead |
| `Error: Index not ready` | FAISS index not built | Run `python main.py` first to build index |
| Mistral timeout | Ollama not running | Run `ollama serve` in a separate terminal |
| Meditron echoes question | Prompt issue on low-acuity cases | Add "Never repeat the question" to system prompt |
| `GROQ_API_KEY not set` | Missing .env | Add key to `.env` file |
| RobertaModel warnings in terminal | BERTScore loading | Normal — these are just warnings, not errors |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- **Australian Emergency Triage Education Kit** — clinical guideline source
- **EPFL** — Meditron medical LLM
- **Mistral AI** — Mistral 7B
- **Groq** — Ultra-fast LLM inference API
- **pritamdeka** — BioBERT medical embeddings
- **Facebook AI** — FAISS vector library

---

*Built as part of a final-year project in Computer Science & Engineering — demonstrating RAG-based clinical decision support.*
