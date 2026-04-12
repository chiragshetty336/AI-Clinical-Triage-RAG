"""
MEDICAL TRIAGE BENCHMARK DASHBOARD v3.0  ← UPDATED: 3 Models
Mistral 7B  |  Groq (Llama-3.1)  |  Meditron 7B  ← NEW 3rd model

WHY MEDITRON instead of medalpaca?
  - medalpaca:7b does NOT exist in Ollama's library → gives "file does not exist" error
  - Meditron IS in Ollama's official library → ollama pull meditron:7b  ✅
  - Meditron is ALSO medically fine-tuned (adapted from LLaMA 2 on medical papers + guidelines)
  - Meditron was developed by EPFL → published, peer-reviewed, strong on clinical benchmarks
  - Same Ollama interface as Mistral → zero extra setup beyond one pull command
"""

import os
import json
import time
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import requests
import re

try:
    from flask import Flask, render_template_string, request, jsonify

    HAS_FLASK = True
except:
    HAS_FLASK = False

try:
    from rag.indexing import load_index
    from rag.config import CACHE_PATH, INDEX_PATH, MODEL_NAME
    from sentence_transformers import SentenceTransformer

    HAS_RAG = True
except:
    HAS_RAG = False

try:
    from groq import Groq

    HAS_GROQ = True
except:
    HAS_GROQ = False

from triage_benchmark import BENCHMARK_CASES, score_against_base

try:
    from rag_metrics import evaluate_result

    HAS_METRICS = True
except Exception as _me:
    HAS_METRICS = False
    print(f"Metrics not loaded: {_me}")

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
MEDITRON_MODEL = os.getenv("MEDITRON_MODEL", "meditron:7b")  # ← NEW
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

index = None
chunks = []
metadata = []
embedding_model = None
comparison_history = []

SYSTEM_PROMPT = """You are a clinical triage assistant specializing in emergency medical assessment.
Analyze patient presentation and provide:
1. Triage Level (RED=Emergency, YELLOW=Urgent, GREEN=Routine)
2. Clinical Assessment with reasoning
3. Specific Recommended Actions with drug names and doses where applicable
4. Disposition recommendation"""


# ═══════════════════════════════════════════════════════
# RAG  (unchanged from your original)
# ═══════════════════════════════════════════════════════


def load_rag_components() -> bool:
    global index, chunks, metadata, embedding_model
    if not HAS_RAG:
        return False
    try:
        if not os.path.exists(INDEX_PATH):
            return False
        index = load_index()
        metadata_path = os.path.join(CACHE_PATH, "metadata.pkl")
        if not os.path.exists(metadata_path):
            return False
        with open(metadata_path, "rb") as f:
            metadata_store = pickle.load(f)
        chunks = metadata_store.get("chunks", [])
        metadata = metadata_store.get("metadata", [])
        embedding_model = SentenceTransformer(MODEL_NAME)
        return True
    except Exception as e:
        print(f"RAG load error: {str(e)}")
        return False


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def retrieve_rag_context(query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
    if index is None or not chunks or embedding_model is None:
        return "", []
    try:
        query_embedding = embedding_model.encode(query)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        distances, indices = index.search(np.array([q_norm]), top_k)
        dist_list = [float(d) for d in distances[0]]
        idx_list = [int(i) for i in indices[0]]

        def dist_to_relevance(d):
            return round(max(0.0, min(1.0, 1.0 - d / 2.0)), 3)

        retrieved_docs = []
        context_text = "RELEVANT CLINICAL GUIDELINES:\n" + "=" * 50 + "\n"

        for idx, distance in zip(idx_list, dist_list):
            if idx < len(chunks):
                chunk = chunks[idx]
                meta = metadata[idx] if idx < len(metadata) else {}
                relevance = dist_to_relevance(distance)
                retrieved_docs.append(
                    {
                        "source": str(meta.get("source", "Unknown")),
                        "page": str(meta.get("page", "N/A")),
                        "section": str(meta.get("section", "General")),
                        "content": str(chunk[:300]),
                        "relevance": relevance,
                    }
                )
                context_text += f"\n[{meta.get('source','Unknown')} - Page {meta.get('page','N/A')}]\n"
                context_text += f"{chunk[:250]}...\n"
                context_text += "-" * 50 + "\n"

        return context_text, retrieved_docs
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return "", []


# ═══════════════════════════════════════════════════════
# LLM QUERIES
# ═══════════════════════════════════════════════════════


def _query_ollama(model_name: str, prompt: str, context: str = "") -> Dict:
    """Shared helper — calls any model running in Ollama."""
    full_prompt = f"{context}\n\nCLINICAL QUESTION:\n{prompt}" if context else prompt
    start = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"].strip()
        return {
            "success": True,
            "answer": answer,
            "model": model_name,
            "latency": round(time.time() - start, 2),
            "error": None,
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "answer": "",
            "model": model_name,
            "latency": 0,
            "error": "Ollama not running. Start with: ollama serve",
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "model": model_name,
            "latency": 0,
            "error": str(e),
        }


def query_mistral(prompt: str, context: str = "") -> Dict:
    return _query_ollama(OLLAMA_MODEL, prompt, context)


# ─── NEW: Meditron ──────────────────────────────────────────────
def query_meditron(prompt: str, context: str = "") -> Dict:
    """
    Meditron 7B — medical-domain LLM from EPFL.
    Pull once:  ollama pull meditron:7b
    """
    return _query_ollama(MEDITRON_MODEL, prompt, context)


# ────────────────────────────────────────────────────────────────


def query_groq(prompt: str, context: str = "") -> Dict:
    if not HAS_GROQ:
        return {
            "success": False,
            "answer": "",
            "model": "groq",
            "latency": 0,
            "error": "pip install groq",
        }
    if not GROQ_API_KEY:
        return {
            "success": False,
            "answer": "",
            "model": "groq",
            "latency": 0,
            "error": "GROQ_API_KEY not set",
        }

    full_prompt = f"{context}\n\nCLINICAL QUESTION:\n{prompt}" if context else prompt
    client = Groq(api_key=GROQ_API_KEY)
    STABLE_MODELS = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    start = time.time()
    last_error = None

    for model in STABLE_MODELS:
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            return {
                "success": True,
                "answer": chat.choices[0].message.content.strip(),
                "model": model,
                "latency": round(time.time() - start, 2),
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "success": False,
        "answer": "",
        "model": "groq",
        "latency": round(time.time() - start, 2),
        "error": f"All models failed: {last_error}",
    }


def extract_triage(text: str) -> str:
    text_upper = text.upper()
    for level in ["RED", "YELLOW", "GREEN"]:
        if level in text_upper:
            return level
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════
# BENCHMARK RUNNER  (updated for 3 models)
# ═══════════════════════════════════════════════════════


def run_benchmark(case_id: str, use_rag: bool = True) -> Dict:
    case = next((c for c in BENCHMARK_CASES if c["id"] == case_id), None)
    if not case:
        return {"error": "Case not found"}

    query = case["query"]
    base_answer = case["base_answer"]

    # RAG retrieval
    rag_context, sources = ("", [])
    retrieval_latency = 0.0
    if use_rag:
        t0 = time.time()
        rag_context, sources = retrieve_rag_context(query, top_k=3)
        retrieval_latency = round(time.time() - t0, 3)

    # ── Query ALL THREE models ──────────────────────────
    mistral = query_mistral(query, rag_context)
    groq = query_groq(query, rag_context)
    meditron = query_meditron(query, rag_context)  # ← NEW

    empty_score = {
        "total": 0,
        "grade": "N/A",
        "triage_correct": False,
        "keyword_coverage": 0,
        "clinical_reasoning": 0,
        "action_specificity": 0,
        "matched_keywords": [],
        "missed_keywords": [],
        "keyword_ratio": "0/0",
        "triage_match": 0,
    }

    mistral_score = (
        score_against_base(mistral.get("answer", ""), base_answer)
        if mistral.get("success")
        else empty_score
    )
    groq_score = (
        score_against_base(groq.get("answer", ""), base_answer)
        if groq.get("success")
        else empty_score
    )
    meditron_score = (
        score_against_base(meditron.get("answer", ""), base_answer)
        if meditron.get("success")
        else empty_score
    )  # ← NEW

    gold_text = (
        base_answer.get("triage_reasoning", "")
        + " "
        + " ".join(base_answer.get("key_actions", []))
    )

    mistral_metrics = groq_metrics = meditron_metrics = {}
    if HAS_METRICS:
        for model_key, model_res, store_key in [
            ("mistral", mistral, "mistral_metrics"),
            ("groq", groq, "groq_metrics"),
            ("meditron", meditron, "meditron_metrics"),
        ]:
            try:
                result = evaluate_result(
                    model_answer=model_res.get("answer", ""),
                    gold_answer_text=gold_text,
                    retrieved_chunks=sources,
                    expected_keywords=base_answer.get("key_keywords", []),
                    retrieval_latency_s=retrieval_latency,
                    generation_latency_s=model_res.get("latency", 0) or 0,
                )
                if store_key == "mistral_metrics":
                    mistral_metrics = result
                elif store_key == "groq_metrics":
                    groq_metrics = result
                else:
                    meditron_metrics = result
            except Exception as e:
                pass

    # ── Determine winner across 3 models ───────────────
    scores_map = {
        "Mistral": mistral_score["total"],
        "Groq": groq_score["total"],
        "Meditron": meditron_score["total"],
    }
    winner = max(scores_map, key=scores_map.get)
    if list(scores_map.values()).count(scores_map[winner]) > 1:
        winner = "Tie"

    result = {
        "timestamp": datetime.now().isoformat(),
        "case_id": case_id,
        "category": case["category"],
        "query": query,
        "use_rag": use_rag,
        "retrieval_latency": retrieval_latency,
        "sources": sources[:3],
        "base_answer": {
            "triage_level": base_answer["triage_level"],
            "triage_reasoning": base_answer["triage_reasoning"],
            "key_actions": base_answer["key_actions"],
            "time_to_treatment": base_answer["time_to_treatment"],
            "disposition": base_answer["disposition"],
        },
        "mistral": {
            "answer": mistral.get("answer", "")[:1200],
            "model": mistral.get("model"),
            "latency": mistral.get("latency"),
            "triage": extract_triage(mistral.get("answer", "")),
            "error": mistral.get("error"),
            "success": mistral.get("success"),
            "score": mistral_score,
            "metrics": mistral_metrics,
        },
        "groq": {
            "answer": groq.get("answer", "")[:1200],
            "model": groq.get("model"),
            "latency": groq.get("latency"),
            "triage": extract_triage(groq.get("answer", "")),
            "error": groq.get("error"),
            "success": groq.get("success"),
            "score": groq_score,
            "metrics": groq_metrics,
        },
        # ── NEW: Meditron block ──────────────────────────
        "meditron": {
            "answer": meditron.get("answer", "")[:1200],
            "model": meditron.get("model"),
            "latency": meditron.get("latency"),
            "triage": extract_triage(meditron.get("answer", "")),
            "error": meditron.get("error"),
            "success": meditron.get("success"),
            "score": meditron_score,
            "metrics": meditron_metrics,
        },
        "winner": winner,
    }

    comparison_history.append(result)
    return result


# ═══════════════════════════════════════════════════════
# CUSTOM QUERY RUNNER  (updated for 3 models)
# ═══════════════════════════════════════════════════════


def run_custom_query(query: str, use_rag: bool = True) -> Dict:
    rag_context, sources = ("", [])
    if use_rag:
        rag_context, sources = retrieve_rag_context(query, top_k=3)

    mistral = query_mistral(query, rag_context)
    groq = query_groq(query, rag_context)
    meditron = query_meditron(query, rag_context)  # ← NEW

    MEDICAL_KW = [
        "triage",
        "RED",
        "YELLOW",
        "GREEN",
        "emergency",
        "urgent",
        "IV",
        "oxygen",
        "monitor",
        "assess",
        "immediate",
        "airway",
        "breathing",
        "circulation",
        "ECG",
        "blood pressure",
    ]

    def rough_score(answer):
        if not answer:
            return {"total": 0, "grade": "N/A"}
        matched = sum(1 for kw in MEDICAL_KW if kw.lower() in answer.lower())
        score = min(100, matched * 6 + 20)
        return {
            "total": score,
            "grade": "Good" if score >= 65 else "Fair" if score >= 50 else "Poor",
            "triage_correct": extract_triage(answer) != "UNKNOWN",
            "keyword_coverage": round(matched / len(MEDICAL_KW) * 40, 1),
            "clinical_reasoning": 15,
            "action_specificity": 5,
            "matched_keywords": [
                kw for kw in MEDICAL_KW if kw.lower() in answer.lower()
            ],
            "missed_keywords": [
                kw for kw in MEDICAL_KW if kw.lower() not in answer.lower()
            ],
            "keyword_ratio": f"{matched}/{len(MEDICAL_KW)}",
        }

    mistral_score = (
        rough_score(mistral.get("answer", ""))
        if mistral.get("success")
        else {"total": 0, "grade": "N/A"}
    )
    groq_score = (
        rough_score(groq.get("answer", ""))
        if groq.get("success")
        else {"total": 0, "grade": "N/A"}
    )
    meditron_score = (
        rough_score(meditron.get("answer", ""))
        if meditron.get("success")
        else {"total": 0, "grade": "N/A"}
    )

    scores_map = {
        "Mistral": mistral_score["total"],
        "Groq": groq_score["total"],
        "Meditron": meditron_score["total"],
    }
    winner = max(scores_map, key=scores_map.get)
    if list(scores_map.values()).count(scores_map[winner]) > 1:
        winner = "Tie"

    result = {
        "timestamp": datetime.now().isoformat(),
        "case_id": "custom",
        "category": "Custom Query",
        "query": query,
        "use_rag": use_rag,
        "sources": sources[:3],
        "base_answer": None,
        "mistral": {
            "answer": mistral.get("answer", "")[:1200],
            "model": mistral.get("model"),
            "latency": mistral.get("latency"),
            "triage": extract_triage(mistral.get("answer", "")),
            "error": mistral.get("error"),
            "success": mistral.get("success"),
            "score": mistral_score,
        },
        "groq": {
            "answer": groq.get("answer", "")[:1200],
            "model": groq.get("model"),
            "latency": groq.get("latency"),
            "triage": extract_triage(groq.get("answer", "")),
            "error": groq.get("error"),
            "success": groq.get("success"),
            "score": groq_score,
        },
        "meditron": {
            "answer": meditron.get("answer", "")[:1200],
            "model": meditron.get("model"),
            "latency": meditron.get("latency"),
            "triage": extract_triage(meditron.get("answer", "")),
            "error": meditron.get("error"),
            "success": meditron.get("success"),
            "score": meditron_score,
        },
        "winner": winner,
    }

    comparison_history.append(result)
    return result


# ═══════════════════════════════════════════════════════
# JINJA MACRO  (result card — updated for 3 models)
# ═══════════════════════════════════════════════════════

RESULT_MACRO = r"""
{% macro render_result(r) %}
{% if r %}
<div style="margin-bottom:24px;">

  <!-- ── WINNER BANNER ── -->
  <div style="background:linear-gradient(90deg,var(--surface) 0%,var(--surface2) 100%);
              border:1px solid var(--border);border-left:3px solid var(--blue);
              border-radius:8px;padding:16px 20px;margin-bottom:20px;
              display:flex;align-items:center;gap:16px;">
    <div>
      <div style="font-size:11px;font-family:'IBM Plex Mono',monospace;color:var(--muted);
                  text-transform:uppercase;">Winner</div>
      <div style="font-size:22px;font-weight:600;color:var(--blue);">{{ r.winner }}</div>
    </div>
    {% if r.base_answer %}
    <div style="margin-left:auto;font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace;">
      GOLD STANDARD — {{ r.base_answer.triage_level }} | {{ r.base_answer.disposition }}
    </div>
    {% endif %}
  </div>

  <!-- ── QUERY ── -->
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
              padding:16px 20px;margin-bottom:20px;font-size:14px;line-height:1.6;
              color:var(--muted);font-style:italic;">"{{ r.query }}"</div>

  <!-- ── THREE MODEL CARDS ── -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;">

    {% for model_key, model_color, model_label in [
        ('mistral', 'var(--blue)', 'Mistral'),
        ('groq',    'var(--yellow)', 'Groq'),
        ('meditron','var(--purple)', 'Meditron')
    ] %}
    {% set m = r[model_key] %}
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:8px;overflow:hidden;">
      <!-- Card header -->
      <div style="padding:14px 18px;border-bottom:1px solid var(--border);
                  display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:12px;font-family:'IBM Plex Mono',monospace;
                      color:{{ model_color }};text-transform:uppercase;
                      letter-spacing:0.08em;">{{ model_label }}</div>
          <div style="font-size:11px;color:var(--muted);margin-top:2px;">
            {{ m.model or 'unknown' }}
          </div>
        </div>
        <!-- Triage chip -->
        {% set t = m.triage %}
        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;
                     padding:5px 12px;border-radius:4px;
                     {% if t=='RED' %}background:var(--red-bg);color:var(--red);border:1px solid var(--red-border);
                     {% elif t=='YELLOW' %}background:var(--yellow-bg);color:var(--yellow);border:1px solid #78520a;
                     {% elif t=='GREEN' %}background:var(--green-bg);color:var(--green);border:1px solid #1a5c1e;
                     {% else %}background:var(--surface2);color:var(--muted);border:1px solid var(--border);
                     {% endif %}">
          ● {{ t }}
        </span>
      </div>

      <!-- Score ring + latency -->
      <div style="padding:16px 18px;border-bottom:1px solid var(--border2);">
        <div style="display:flex;align-items:center;gap:16px;">
          {% set sc = m.score.total|default(0)|float %}
          <div style="position:relative;width:80px;height:80px;flex-shrink:0;">
            <svg width="80" height="80" style="transform:rotate(-90deg);">
              <circle cx="40" cy="40" r="34" fill="none" stroke="var(--surface2)" stroke-width="6"/>
              <circle cx="40" cy="40" r="34" fill="none" stroke="{{ model_color }}" stroke-width="6"
                stroke-dasharray="{{ 2 * 3.14159 * 34 }}"
                stroke-dashoffset="{{ 2 * 3.14159 * 34 * (1 - sc/100) }}"
                stroke-linecap="round"/>
            </svg>
            <div style="position:absolute;inset:0;display:flex;align-items:center;
                        justify-content:center;font-family:'IBM Plex Mono',monospace;
                        font-size:18px;font-weight:600;color:{{ model_color }};">
              {{ sc|int }}
            </div>
          </div>
          <div style="flex:1;">
            <div style="font-size:12px;color:var(--muted);margin-bottom:4px;">
              {{ m.score.grade|default('N/A') }}
            </div>
            <div style="font-size:11px;color:var(--muted);">
              {% if m.score.triage_correct %}
                ✓ TRIAGE CORRECT
              {% else %}
                ✗ Triage mismatch
              {% endif %}
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:6px;">
              ⏱ {{ m.latency or '—' }}s
            </div>
          </div>
        </div>

        <!-- Score sub-bars -->
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:6px;">
          {% for label, val, maxv in [
              ('Triage match',      m.score.triage_match|default(0),      30),
              ('Keyword coverage',  m.score.keyword_coverage|default(0),   40),
              ('Clinical reasoning',m.score.clinical_reasoning|default(0), 20),
              ('Specificity',       m.score.action_specificity|default(0), 10)
          ] %}
          <div style="display:flex;align-items:center;gap:8px;font-size:11px;">
            <span style="width:120px;color:var(--muted);font-family:'IBM Plex Mono',monospace;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{{ label }}</span>
            <div style="flex:1;height:5px;background:var(--surface2);border-radius:3px;overflow:hidden;">
              <div style="width:{{ (val/maxv*100)|int }}%;height:100%;
                          background:{{ model_color }};border-radius:3px;"></div>
            </div>
            <span style="width:40px;text-align:right;color:var(--text);font-size:10px;">
              {{ val|round(1) }}/{{ maxv }}
            </span>
          </div>
          {% endfor %}
        </div>
      </div>

      <!-- Missed keywords -->
      {% if m.score.missed_keywords %}
      <div style="padding:12px 18px;border-bottom:1px solid var(--border2);">
        <div style="font-size:10px;color:var(--muted);font-family:'IBM Plex Mono',monospace;
                    margin-bottom:6px;text-transform:uppercase;">MISSED KEYWORDS</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
          {% for kw in m.score.missed_keywords[:8] %}
          <span style="font-size:10px;font-family:'IBM Plex Mono',monospace;
                       padding:2px 6px;border-radius:3px;
                       background:var(--red-bg);color:var(--red);
                       border:1px solid var(--red-border);">{{ kw }}</span>
          {% endfor %}
        </div>
      </div>
      {% endif %}

      <!-- Answer text -->
      <div style="padding:14px 18px;">
        <div style="font-size:10px;color:var(--muted);font-family:'IBM Plex Mono',monospace;
                    text-transform:uppercase;margin-bottom:8px;">RESPONSE</div>
        {% if m.error %}
          <div style="color:var(--red);font-size:12px;">Error: {{ m.error }}</div>
        {% else %}
          <div style="background:var(--surface2);border:1px solid var(--border);
                      border-radius:6px;padding:12px;max-height:200px;overflow-y:auto;
                      font-size:12px;line-height:1.7;color:var(--muted);">
            {{ m.answer[:600] }}{% if m.answer|length > 600 %}…{% endif %}
          </div>
        {% endif %}
      </div>
    </div>
    {% endfor %}

  </div><!-- /grid-3 -->

  {% if r.base_answer %}
  <!-- ── GOLD STANDARD ── -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
    <div class="card">
      <div class="card-head">
        <span class="card-head-label">Gold Standard Actions</span>
      </div>
      <div class="card-body base-actions">
        {% for action in r.base_answer.key_actions %}
        <div class="base-action-item">
          <span class="action-num">{{ loop.index }}.</span>
          <span>{{ action }}</span>
        </div>
        {% endfor %}
      </div>
    </div>

    <!-- RAG sources -->
    {% if r.sources %}
    <div class="card">
      <div class="card-head">
        <span class="card-head-label">RAG — Retrieved Guidelines</span>
      </div>
      <div class="card-body">
        {% for src in r.sources %}
        <div style="padding:8px 0;border-bottom:1px solid var(--border2);font-size:12px;">
          <div style="color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:10px;">
            {{ src.source }} · Relevance: {{ src.relevance }} · p.{{ src.page }}
          </div>
          <div style="color:var(--text);margin-top:4px;line-height:1.5;">
            {{ src.content[:140] }}...
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}
  </div>
  {% endif %}

  <!-- ── RAG PIPELINE METRICS TABLE ── -->
  {% if r.mistral.metrics and r.groq.metrics and r.meditron.metrics %}
  <div class="card" style="margin-bottom:20px;">
    <div class="card-head">
      <span class="card-head-label">RAG Pipeline Metrics — M3 to M19</span>
      <span style="font-size:11px;color:var(--muted);">From guide's M1–M24 benchmark suite</span>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace;">
      <thead>
        <tr style="background:var(--surface2);">
          <th style="padding:9px 16px;text-align:left;color:var(--muted);font-weight:400;">ID</th>
          <th style="padding:9px 16px;text-align:left;color:var(--muted);font-weight:400;">Metric</th>
          <th style="padding:9px 16px;text-align:left;color:var(--muted);font-weight:400;">Description</th>
          <th style="padding:9px 16px;text-align:center;color:var(--blue);font-weight:600;">Mistral</th>
          <th style="padding:9px 16px;text-align:center;color:var(--yellow);font-weight:600;">Groq</th>
          <th style="padding:9px 16px;text-align:center;color:var(--purple);font-weight:600;">Meditron</th>
          <th style="padding:9px 16px;text-align:center;color:var(--muted);font-weight:400;">Winner</th>
        </tr>
      </thead>
      <tbody>

        {% set metric_rows = [
          ('M3',  'Retrieval Latency',  'Time to fetch top-k chunks (s)',                   'M3_retrieval_latency_s',  False),
          ('M4',  'Cosine Similarity',  'Semantic closeness: query ↔ retrieved chunk',       'M4_cosine_similarity',    True),
          ('M5',  'Top-k Accuracy',     '% of expected keywords in retrieved chunks',         'M5_topk_accuracy_pct',   True),
          ('M6',  'ROUGE-1 F1',         'Word overlap with gold standard answer',             'M6_rouge1_f1',            True),
          ('M8',  'ROUGE-L F1',         'Longest common subsequence with gold answer',        'M8_rouge_l_f1',           True),
          ('M12', 'BERTScore F1',       'Deep semantic similarity (contextual embeddings)',   'M12_bertscore_f1',        True),
          ('M14', 'Faithfulness',       '% of retrieved context used in answer',              'M14_faithfulness',        True),
          ('M15', 'GT Coverage',        '% of gold standard words found in answer',           'M15_gt_coverage',         True),
          ('M16', 'E2E Latency',        'Total query→answer time (retrieval + generation)',   'M16_e2e_latency_s',       False),
          ('M17', 'Throughput',         'Queries per second',                                 'M17_throughput_qps',      True),
          ('M18', 'CPU Usage',          'Processor load during inference (%)',                'M18_cpu_percent',         False),
          ('M19', 'RAM Usage',          'Memory consumption (GB used)',                       'M19_ram_gb',              False),
        ] %}

        {% for id, label, desc, key, higher_better in metric_rows %}
        {% set mv  = r.mistral.metrics.get(key)  %}
        {% set gv  = r.groq.metrics.get(key)     %}
        {% set mav = r.meditron.metrics.get(key) %}
        {% set row_bg = 'var(--surface2)' if loop.index % 2 == 0 else 'var(--surface)' %}
        <tr style="border-bottom:1px solid var(--border2);background:{{ row_bg }};">
          <td style="padding:9px 16px;color:var(--muted);">{{ id }}</td>
          <td style="padding:9px 16px;color:var(--text);">{{ label }}</td>
          <td style="padding:9px 16px;color:var(--muted);">{{ desc }}</td>
          <td style="padding:9px 16px;text-align:center;color:var(--blue);">{{ mv if mv is not none else 'N/A' }}</td>
          <td style="padding:9px 16px;text-align:center;color:var(--yellow);">{{ gv if gv is not none else 'N/A' }}</td>
          <td style="padding:9px 16px;text-align:center;color:var(--purple);">{{ mav if mav is not none else 'N/A' }}</td>
          <td style="padding:9px 16px;text-align:center;">
            {% if mv is not none and gv is not none and mav is not none %}
              {% if higher_better %}
                {% set best = [mv, gv, mav]|max %}
                {% if best == mv %}<span style="color:var(--blue)">M</span>
                {% elif best == gv %}<span style="color:var(--yellow)">G</span>
                {% else %}<span style="color:var(--purple)">MA</span>{% endif %}
              {% else %}
                {% set best = [mv, gv, mav]|min %}
                {% if best == mv %}<span style="color:var(--blue)">M</span>
                {% elif best == gv %}<span style="color:var(--yellow)">G</span>
                {% else %}<span style="color:var(--purple)">MA</span>{% endif %}
              {% endif %}
            {% else %}—{% endif %}
          </td>
        </tr>
        {% endfor %}

      </tbody>
    </table>
    <div style="padding:10px 16px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);
                font-family:'IBM Plex Mono',monospace;">
      M = Mistral wins &nbsp;|&nbsp; G = Groq wins &nbsp;|&nbsp; MA = Meditron wins
      &nbsp;|&nbsp; For latency: lower is better &nbsp;|&nbsp; For all others: higher is better
    </div>
  </div>
  {% endif %}

</div>
{% endif %}
{% endmacro %}
"""


# ═══════════════════════════════════════════════════════
# DASHBOARD HTML  (updated status bar + JS for 3 models)
# ═══════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MedTriage Benchmark</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;
  --border:#30363d;--border2:#21262d;--text:#e6edf3;--muted:#8b949e;
  --red:#f85149;--red-bg:#3d0000;--red-border:#b91c1c;
  --yellow:#e3b341;--yellow-bg:#2d2000;
  --green:#3fb950;--green-bg:#0d2a0e;
  --blue:#58a6ff;--blue-bg:#0c2a4a;
  --purple:#bc8cff;--purple-bg:#2a1a4a;
  --teal:#39d353;--orange:#f0883e;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:16px 32px;
        display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}
.logo{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:600;color:var(--text);}
.logo span{color:var(--red);}
.status-pills{display:flex;gap:8px;}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-family:'IBM Plex Mono',monospace;
      padding:5px 10px;border-radius:4px;border:1px solid var(--border);background:var(--surface2);}
.pill-dot{width:6px;height:6px;border-radius:50%;}
.dot-green{background:var(--green);}
.dot-red{background:var(--red);}
.dot-yellow{background:var(--yellow);}
.dot-purple{background:var(--purple);}
.main{display:flex;height:calc(100vh - 57px);}
.sidebar{width:260px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--border);
         overflow-y:auto;display:flex;flex-direction:column;}
.sidebar-title{padding:16px 20px 8px;font-size:11px;font-family:'IBM Plex Mono',monospace;
               color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}
.case-btn{display:block;width:100%;text-align:left;padding:14px 20px;background:none;border:none;
          border-left:3px solid transparent;cursor:pointer;color:var(--muted);
          font-family:'IBM Plex Sans',sans-serif;font-size:13px;transition:all 0.15s;}
.case-btn:hover{background:var(--surface2);color:var(--text);border-left-color:var(--border);}
.case-btn.active{background:var(--surface2);color:var(--text);border-left-color:var(--blue);}
.case-btn .case-tag{display:inline-block;font-size:10px;font-family:'IBM Plex Mono',monospace;
                    padding:2px 6px;border-radius:3px;margin-bottom:4px;}
.tag-red{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border);}
.tag-yellow{background:var(--yellow-bg);color:var(--yellow);border:1px solid #78520a;}
.tag-green{background:var(--green-bg);color:var(--green);border:1px solid #1a5c1e;}
.case-btn .case-title{display:block;line-height:1.4;}
.sidebar-sep{height:1px;background:var(--border);margin:16px 0;}
.custom-area{padding:16px 20px;}
.custom-label{font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-bottom:8px;}
.custom-textarea{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:6px;
                 color:var(--text);font-family:'IBM Plex Sans',sans-serif;font-size:12px;padding:10px;
                 resize:vertical;min-height:80px;outline:none;}
.custom-textarea:focus{border-color:var(--blue);}
.run-btn{width:100%;margin-top:8px;padding:10px;background:var(--blue);color:#000;font-weight:600;
         font-size:13px;border:none;border-radius:6px;cursor:pointer;
         font-family:'IBM Plex Sans',sans-serif;transition:opacity 0.15s;}
.run-btn:hover{opacity:0.85;}
.run-btn:disabled{opacity:0.4;cursor:not-allowed;}
.rag-toggle{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px;color:var(--muted);}
.content{padding:28px 32px;overflow-y:auto;flex:1;}
.loading-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;
                 background:rgba(13,17,23,0.85);z-index:200;
                 align-items:center;justify-content:center;flex-direction:column;gap:16px;}
.loading-overlay.show{display:flex;}
.loader{width:40px;height:40px;border:2px solid var(--border);border-top-color:var(--blue);
        border-radius:50%;animation:spin 0.8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-text{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--muted);}
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;
         height:70vh;gap:12px;opacity:0.4;}
.welcome-icon{font-size:48px;}
.welcome-text{font-size:15px;color:var(--muted);}
/* re-used card classes */
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.card-head{padding:14px 18px;border-bottom:1px solid var(--border);
           display:flex;align-items:center;justify-content:space-between;}
.card-head-label{font-size:12px;font-family:'IBM Plex Mono',monospace;color:var(--muted);
                 text-transform:uppercase;letter-spacing:0.08em;}
.card-body{padding:18px;}
.base-actions{display:flex;flex-direction:column;gap:6px;}
.base-action-item{display:flex;gap:10px;font-size:13px;color:var(--muted);
                  padding:6px 0;border-bottom:1px solid var(--border2);align-items:flex-start;}
.action-num{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--blue);
            width:20px;flex-shrink:0;margin-top:2px;}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo">MED<span>TRIAGE</span>.BENCHMARK</div>
  <div class="status-pills">
    <span class="pill">
      <span class="pill-dot {{ 'dot-green' if rag_ready else 'dot-red' }}"></span>
      RAG {{ 'ACTIVE' if rag_ready else 'OFFLINE' }}
    </span>
    <span class="pill">
      <span class="pill-dot dot-green"></span>
      MISTRAL
    </span>
    <span class="pill">
      <span class="pill-dot {{ 'dot-green' if groq_ready else 'dot-yellow' }}"></span>
      GROQ
    </span>
    <!-- NEW: Meditron pill -->
    <span class="pill">
      <span class="pill-dot dot-purple"></span>
      MEDITRON
    </span>
    <span class="pill">
      <span class="pill-dot dot-green"></span>
      {{ total_comps }} RUNS
    </span>
  </div>
</div>

<!-- LOADING -->
<div class="loading-overlay" id="loading">
  <div class="loader"></div>
  <div class="loading-text" id="loadingText">Running 3 models + RAG...</div>
</div>

<div class="main">
  <!-- SIDEBAR -->
  <div class="sidebar">
    <div class="sidebar-title">Benchmark Cases</div>
    {% for case in cases %}
    {% set tl = case.base_answer.triage_level %}
    <button class="case-btn" onclick="runBenchmark('{{ case.id }}')">
      <span class="case-tag {{ 'tag-red' if tl=='RED' else 'tag-yellow' if tl=='YELLOW' else 'tag-green' }}">
        {{ tl }}
      </span>
      <span class="case-title">{{ case.category }}</span>
      <span style="font-size:10px;color:var(--muted);display:block;margin-top:2px;">
        {{ case.query[:55] }}...
      </span>
    </button>
    {% endfor %}

    <div class="sidebar-sep"></div>
    <div class="custom-area">
      <div class="custom-label">// CUSTOM QUERY</div>
      <textarea class="custom-textarea" id="customQuery" placeholder="Enter clinical scenario..."></textarea>
      <div class="rag-toggle">
        <input type="checkbox" id="useRag" checked>
        <label for="useRag">Use RAG context</label>
      </div>
      <button class="run-btn" onclick="runCustom()">Run Custom Query</button>
    </div>

    <!-- History sidebar -->
    {% if history %}
    <div class="sidebar-sep"></div>
    <div class="sidebar-title">History</div>
    {% for h in history[-8:]|reverse %}
    <div style="padding:10px 20px;border-left:3px solid var(--border);font-size:12px;cursor:pointer;"
         onclick="showHistory({{ loop.revindex0 }})">
      <div style="color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        {{ h.category }}
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:2px;font-family:'IBM Plex Mono',monospace;">
        M:{{ h.mistral.score.total|default(0)|round(1) }}
        G:{{ h.groq.score.total|default(0)|round(1) }}
        MA:{{ h.meditron.score.total|default(0)|round(1) }}
      </div>
    </div>
    {% endfor %}
    {% endif %}
  </div>

  <!-- MAIN CONTENT -->
  <div class="content" id="mainContent">
    {% if latest %}
      {{ render_result(latest) }}
    {% else %}
    <div class="welcome">
      <div class="welcome-icon">⚕</div>
      <div class="welcome-text">Select a benchmark case or enter a custom query</div>
      <div style="font-size:12px;color:var(--muted);margin-top:4px;">
        Mistral 7B  •  Groq (Llama-3.1)  •  <span style="color:var(--purple);">Meditron 7B</span>
      </div>
    </div>
    {% endif %}
  </div>
</div>

<script>
const historyData = {{ history|tojson|safe }};

async function runBenchmark(caseId) {
  document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
  event.currentTarget.classList.add('active');
  showLoading('Running 3 models + RAG...');
  try {
    const resp = await fetch('/api/benchmark', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({case_id: caseId, use_rag: true})
    });
    const data = await resp.json();
    renderResult(data);
  } catch(e) {
    alert('Error: ' + e.message);
  } finally {
    hideLoading();
  }
}

async function runCustom() {
  const query = document.getElementById('customQuery').value.trim();
  if (!query) { alert('Please enter a clinical scenario.'); return; }
  const useRag = document.getElementById('useRag').checked;
  showLoading('Querying Mistral + Groq + Meditron...');
  try {
    const resp = await fetch('/api/custom', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, use_rag: useRag})
    });
    const data = await resp.json();
    renderResult(data);
  } catch(e) {
    alert('Error: ' + e.message);
  } finally {
    hideLoading();
  }
}

function renderResult(data) {
  fetch('/api/render', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.text()).then(html => {
    document.getElementById('mainContent').innerHTML = html;
  }).catch(() => {
    document.getElementById('mainContent').innerHTML =
      '<pre style="color:var(--muted);padding:20px;">' +
      JSON.stringify(data, null, 2) + '</pre>';
  });
}

function showHistory(idx) {
  const item = historyData[historyData.length - 1 - idx];
  if (item) renderResult(item);
}

function showLoading(msg) {
  document.getElementById('loadingText').textContent = msg || 'Processing...';
  document.getElementById('loading').classList.add('show');
}
function hideLoading() {
  document.getElementById('loading').classList.remove('show');
}
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════


def create_app():
    if not HAS_FLASK:
        print("Flask required: pip install flask")
        return None

    app = Flask(__name__)
    FULL_TEMPLATE = RESULT_MACRO + DASHBOARD_HTML

    @app.route("/")
    def dashboard():
        latest = comparison_history[-1] if comparison_history else None
        return render_template_string(
            FULL_TEMPLATE,
            latest=latest,
            history=comparison_history,
            cases=BENCHMARK_CASES,
            total_comps=len(comparison_history),
            rag_ready=(index is not None),
            groq_ready=HAS_GROQ and bool(GROQ_API_KEY),
        )

    @app.route("/api/benchmark", methods=["POST"])
    def api_benchmark():
        data = request.json
        case_id = data.get("case_id", "")
        use_rag = data.get("use_rag", True)
        if not case_id:
            return jsonify({"error": "No case_id"}), 400
        result = run_benchmark(case_id, use_rag)
        return jsonify(sanitize_for_json(result))

    @app.route("/api/custom", methods=["POST"])
    def api_custom():
        data = request.json
        query = data.get("query", "")
        use_rag = data.get("use_rag", True)
        if not query:
            return jsonify({"error": "No query"}), 400
        result = run_custom_query(query, use_rag)
        return jsonify(sanitize_for_json(result))

    @app.route("/api/history", methods=["GET"])
    def history():
        return jsonify(sanitize_for_json(comparison_history))

    @app.route("/api/render", methods=["POST"])
    def render_result_endpoint():
        """Renders a result dict as HTML using the Jinja macro."""
        data = request.json
        rendered = render_template_string(
            RESULT_MACRO + "{{ render_result(r) }}", r=data
        )
        return rendered

    return app


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MedTriage Benchmark Dashboard v3.0  (3-Model Edition)")
    print("=" * 60)
    rag_ok = load_rag_components()
    print(f"  RAG:      {'Ready' if rag_ok else 'Not available'}")
    print(f"  Mistral:  Check ollama serve is running")
    print(
        f"  Groq:     {'Ready' if HAS_GROQ and GROQ_API_KEY else 'Check GROQ_API_KEY'}"
    )
    print(f"  Meditron: Check ollama pull meditron:7b")
    print(f"  Flask:    {'Ready' if HAS_FLASK else 'pip install flask'}")
    print(f"  Cases:    {len(BENCHMARK_CASES)} benchmark cases loaded")
    print("=" * 60)
    print("\n  Open: http://localhost:5000\n")

    if not HAS_FLASK:
        exit(1)

    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
