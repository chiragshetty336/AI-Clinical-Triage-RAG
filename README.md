




# 🧠 AI Clinical Triage RAG System

## 📌 Project Overview

This project is an **AI-powered Clinical Triage Decision Support System** built using a **Retrieval-Augmented Generation (RAG)** architecture.

It analyzes **patient symptoms and vital signs**, retrieves relevant **medical triage guidelines**, and generates **clear, structured, and easy-to-understand recommendations**.

The system simulates real-world emergency triage by combining:

- ⚕️ Medical knowledge retrieval  
- 🤖 LLM-based reasoning  
- 🚑 Rule-based triage classification  
- 🔁 Self-reflection for improved answers  

---

# 🏗️ System Architecture

```

User Query
↓
Query Normalization
↓
Emergency Detection (Override)
↓
Triage Classification (RED / YELLOW / GREEN)
↓
Hybrid Retrieval (FAISS + BM25)
↓
Cross-Encoder Reranking
↓
Context Generation
↓
LLM Response Generation
↓
Self-Reflection (Answer Improvement)
↓
Safety & Evaluation
↓
Final Clinical Response

```

---

# 🚀 Key Features

## 🧩 1. Medical RAG Pipeline
- Medical PDFs are ingested and processed
- Text is chunked and converted into embeddings
- FAISS vector database stores embeddings
- Relevant medical context is retrieved dynamically

---

## 🚑 2. Clinical Triage Classification

| Level  | Description                     |
|--------|---------------------------------|
| 🔴 RED    | Life-threatening emergency        |
| 🟡 YELLOW | Urgent but stable condition      |
| 🟢 GREEN  | Mild / non-emergency condition   |

---

## ❤️ 3. Vital Signs Triage Engine

Evaluates:
- Heart Rate  
- Oxygen Saturation  
- Temperature  
- Systolic Blood Pressure  

👉 Automatically overrides triage if critical values are detected.

---

## 🔍 4. Hybrid Retrieval System

Combines:
- **FAISS (semantic search)**
- **BM25 (keyword search)**

👉 Improves both:
- Recall (finding relevant info)
- Precision (reducing noise)

---

## 🧠 5. Cross-Encoder Reranking

- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-ranks retrieved documents
- Keeps only highly relevant medical context

---

## 🤖 6. LLM-Based Response Generation

- Powered by **Ollama (Mistral / Phi3)**
- Produces:
  - Simple explanations
  - Clear reasoning
  - Actionable steps for patients

---

## 🔁 7. Self-Reflection RAG (Advanced Feature)

- The model reviews its own answer
- Improves:
  - clarity
  - correctness
  - safety

👉 This significantly boosts output quality and reduces hallucination

---

## 🛡️ 8. Safety & Evaluation System

### Metrics:
- **Triage Accuracy** → classification correctness  
- **Triage Score** → reasoning + clarity  

### Safety Checks:
- Low confidence detection  
- Faithfulness scoring  
- Context validation  

---

## ⚡ 9. FastAPI Clinical API

### Endpoint:
```

POST /query

````

### Example Response:
```json
{
  "triage_level": "RED",
  "vital_triage": "RED",
  "confidence_score": 0.82,
  "faithfulness_score": 78.5,
  "emergency_detected": true,
  "safety_flag": false,
  "recommended_action": "Immediate emergency care required"
}
````

---

## 🔄 10. Airflow Data Pipeline

Automates data ingestion:

1. Detect PDFs
2. Process and chunk text
3. Generate embeddings
4. Update FAISS index
5. Validate pipeline

---

# 🛠️ Tech Stack

* **Backend:** FastAPI
* **Vector Database:** FAISS
* **Embeddings:** BioBERT (`pritamdeka/...`)
* **LLM:** Ollama (Mistral / Phi3)
* **Reranker:** Cross-Encoder MiniLM
* **Retrieval:** Hybrid (FAISS + BM25)
* **Pipeline:** Apache Airflow
* **Containerization:** Docker
* **Libraries:** NumPy, PyMuPDF, Sentence Transformers

---

# 📁 Project Structure

```
api/                # FastAPI endpoints
rag/                # Core RAG pipeline
evaluation/         # Evaluation system
dags/               # Airflow pipelines
data/               # Medical datasets
Dockerfile
docker-compose.yaml
requirements.txt
```

---

# ▶️ How to Run

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd medical-triage-rag
```

## 2. Start Services

```bash
docker compose up --build
```

## 3. Access API

```
http://localhost:8000/docs
```

## 4. Run Evaluation

```bash
python -m evaluation.evaluate_dataset
```

---

# 📊 Current Performance

* ✅ Triage Accuracy: **100%**
* 🔥 Average Triage Score: **0.9**
* ✅ Clear, user-friendly outputs
* ✅ Reduced hallucination (Self-Reflection RAG)

---

# 🎯 Use Cases

* Emergency triage simulation
* Clinical decision support research
* AI in healthcare experimentation
* RAG system benchmarking

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

It is **not a medical device** and should **not be used for real clinical decisions**.

```


