"""
RAG Metrics Evaluator — Medical Triage Project
Selected metrics from M1-M24 that are relevant to a medical RAG system.

SELECTED METRICS (from guide's full list):
  Retrieval: M3 (Retrieval Latency), M4 (Cosine Similarity), M5 (Top-k Accuracy)
  Answer Quality: M6 (ROUGE-1), M8 (ROUGE-L), M12 (BERTScore), M14 (Faithfulness), M15 (GT Coverage)
  System Efficiency: M16 (E2E Latency), M17 (Throughput), M18 (CPU Usage), M19 (RAM Usage)

SKIPPED (not applicable):
  M1/M2 — one-time indexing metrics, not per-query
  M7 (ROUGE-2) — redundant with ROUGE-1 and ROUGE-L
  M9 (Context Length) — informational only
  M10 (BLEU) — designed for translation, penalises clinical paraphrasing
  M11 (METEOR) — requires NLTK WordNet, heavy dependency
  M13/M23 (FCD duplicate) — same metric listed twice
  M20-M22, M24 — Legal-specific, not applicable to medical triage
"""

import time
import re
import psutil
import os
from typing import Optional

# BERTScore — optional heavy dependency
try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

# Sentence transformers for cosine similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SIM_MODEL = True
except ImportError:
    HAS_SIM_MODEL = False


# ═══════════════════════════════════════════════════════
# M3 — RETRIEVAL LATENCY
# ═══════════════════════════════════════════════════════

def measure_retrieval_latency(retrieval_fn, query: str) -> dict:
    """
    M3: Time taken to fetch top-k chunks from the FAISS index.
    Usage: wrap your retrieve_rag_context() call with this.
    """
    start = time.perf_counter()
    result = retrieval_fn(query)
    latency = round(time.perf_counter() - start, 4)
    return {
        "metric": "M3_retrieval_latency",
        "value_seconds": latency,
        "result": result
    }


# ═══════════════════════════════════════════════════════
# M4 — COSINE SIMILARITY (Query ↔ Retrieved Chunk)
# ═══════════════════════════════════════════════════════

def cosine_similarity_score(query: str, retrieved_text: str) -> dict:
    """
    M4: Semantic closeness of retrieved chunk to query (-1 to 1).
    Higher = more relevant retrieval.
    """
    if not HAS_SIM_MODEL or not retrieved_text:
        return {"metric": "M4_cosine_similarity", "value": None, "note": "Model not available"}

    try:
        embeddings = _sim_model.encode([query, retrieved_text])
        q, r = embeddings[0], embeddings[1]
        cos = float(np.dot(q, r) / (np.linalg.norm(q) * np.linalg.norm(r)))
        return {
            "metric": "M4_cosine_similarity",
            "value": round(cos, 4),
            "interpretation": (
                "High relevance" if cos > 0.7 else
                "Moderate relevance" if cos > 0.4 else
                "Low relevance"
            )
        }
    except Exception as e:
        return {"metric": "M4_cosine_similarity", "value": None, "error": str(e)}


# ═══════════════════════════════════════════════════════
# M5 — TOP-K ACCURACY
# ═══════════════════════════════════════════════════════

def topk_accuracy(retrieved_sources: list, expected_keywords: list) -> dict:
    """
    M5: Whether the retrieved chunks contain expected clinical keywords.
    Proxy for ground truth passage retrieval accuracy.
    """
    if not retrieved_sources or not expected_keywords:
        return {"metric": "M5_topk_accuracy", "value": 0.0}

    combined_text = " ".join(
        src.get("content", "") for src in retrieved_sources
    ).lower()

    matched = [kw for kw in expected_keywords if kw.lower() in combined_text]
    accuracy = round(len(matched) / len(expected_keywords) * 100, 1)

    return {
        "metric": "M5_topk_accuracy",
        "value_percent": accuracy,
        "matched_keywords": matched,
        "total_keywords": len(expected_keywords),
        "interpretation": (
            "Good retrieval" if accuracy >= 60 else
            "Partial retrieval" if accuracy >= 30 else
            "Poor retrieval — queries may not match indexed content"
        )
    }


# ═══════════════════════════════════════════════════════
# M6 — ROUGE-1 (Word overlap with gold answer)
# ═══════════════════════════════════════════════════════

def rouge1_score(model_answer: str, gold_answer: str) -> dict:
    """
    M6: Unigram word overlap between model answer and gold standard.
    Formula: |gt ∩ ans| / |gt|
    """
    if not model_answer or not gold_answer:
        return {"metric": "M6_rouge1", "value": 0.0}

    def tokenize(text):
        return set(re.sub(r'[^\w\s]', '', text.lower()).split())

    gt_tokens = tokenize(gold_answer)
    ans_tokens = tokenize(model_answer)
    intersection = gt_tokens & ans_tokens

    recall = len(intersection) / len(gt_tokens) if gt_tokens else 0
    precision = len(intersection) / len(ans_tokens) if ans_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "metric": "M6_rouge1",
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "value": round(f1, 4)
    }


# ═══════════════════════════════════════════════════════
# M8 — ROUGE-L (Longest Common Subsequence)
# ═══════════════════════════════════════════════════════

def rouge_l_score(model_answer: str, gold_answer: str) -> dict:
    """
    M8: Longest Common Subsequence between model answer and gold standard.
    Captures structural/sentence-level similarity.
    """
    if not model_answer or not gold_answer:
        return {"metric": "M8_rouge_l", "value": 0.0}

    def lcs_length(a, b):
        # Efficient LCS using DP on word lists
        words_a = a.lower().split()
        words_b = b.lower().split()
        m, n = len(words_a), len(words_b)
        # Space-efficient: only keep two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words_a[i-1] == words_b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    gt_len = len(gold_answer.split())
    ans_len = len(model_answer.split())
    lcs = lcs_length(model_answer, gold_answer)

    recall = lcs / gt_len if gt_len else 0
    precision = lcs / ans_len if ans_len else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "metric": "M8_rouge_l",
        "lcs_length": lcs,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "value": round(f1, 4)
    }


# ═══════════════════════════════════════════════════════
# M12 — BERTScore (Deep Semantic Similarity)
# ═══════════════════════════════════════════════════════

def bertscore(model_answer: str, gold_answer: str) -> dict:
    """
    M12: Deep semantic similarity using contextual embeddings.
    Falls back to cosine similarity if bert_score not installed.
    Install: pip install bert-score
    """
    if HAS_BERTSCORE and model_answer and gold_answer:
        try:
            P, R, F1 = bert_score_fn(
                [model_answer], [gold_answer],
                lang="en", verbose=False
            )
            return {
                "metric": "M12_bertscore",
                "precision": round(float(P[0]), 4),
                "recall": round(float(R[0]), 4),
                "f1": round(float(F1[0]), 4),
                "value": round(float(F1[0]), 4),
                "method": "bert_score"
            }
        except Exception as e:
            pass  # Fall through to cosine fallback

    # Fallback: cosine similarity between sentence embeddings
    if HAS_SIM_MODEL and model_answer and gold_answer:
        try:
            embeddings = _sim_model.encode([model_answer, gold_answer])
            a, b = embeddings[0], embeddings[1]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            return {
                "metric": "M12_bertscore",
                "value": round(cos, 4),
                "method": "cosine_fallback",
                "note": "pip install bert-score for true BERTScore"
            }
        except Exception as e:
            return {"metric": "M12_bertscore", "value": None, "error": str(e)}

    return {"metric": "M12_bertscore", "value": None, "note": "No embedding model available"}


# ═══════════════════════════════════════════════════════
# M14 — FAITHFULNESS
# ═══════════════════════════════════════════════════════

def faithfulness_score(model_answer: str, retrieved_chunks: list) -> dict:
    """
    M14: How much of the retrieved evidence is reflected in the answer.
    Formula: chunks_used / total_chunks * 100
    Proxy: count how many retrieved chunk keywords appear in the answer.
    """
    if not retrieved_chunks or not model_answer:
        return {"metric": "M14_faithfulness", "value": 0.0}

    answer_lower = model_answer.lower()
    chunk_scores = []

    for chunk in retrieved_chunks:
        content = chunk.get("content", "")
        if not content:
            continue
        # Extract meaningful words (>4 chars) from chunk
        chunk_words = set(
            w for w in re.sub(r'[^\w\s]', '', content.lower()).split()
            if len(w) > 4
        )
        if not chunk_words:
            continue
        matched = sum(1 for w in chunk_words if w in answer_lower)
        chunk_scores.append(matched / len(chunk_words))

    if not chunk_scores:
        return {"metric": "M14_faithfulness", "value": 0.0}

    faithfulness = round(sum(chunk_scores) / len(chunk_scores) * 100, 1)
    return {
        "metric": "M14_faithfulness",
        "value_percent": faithfulness,
        "chunks_evaluated": len(chunk_scores),
        "interpretation": (
            "High — answer well grounded in retrieved context" if faithfulness >= 60 else
            "Medium — partial grounding" if faithfulness >= 30 else
            "Low — answer may not reflect retrieved guidelines"
        )
    }


# ═══════════════════════════════════════════════════════
# M15 — GT COVERAGE
# ═══════════════════════════════════════════════════════

def gt_coverage(model_answer: str, gold_answer_text: str) -> dict:
    """
    M15: % of gold standard words found in the model answer.
    Formula: |gt ∩ ans| / |gt| * 100
    """
    if not model_answer or not gold_answer_text:
        return {"metric": "M15_gt_coverage", "value": 0.0}

    gt_words = set(re.sub(r'[^\w\s]', '', gold_answer_text.lower()).split())
    ans_words = set(re.sub(r'[^\w\s]', '', model_answer.lower()).split())

    # Remove stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
        "was", "for", "with", "at", "by", "from", "as", "be", "if",
        "it", "this", "that", "not", "but", "on", "has", "have", "may"
    }
    gt_words -= stopwords
    ans_words -= stopwords

    matched = gt_words & ans_words
    coverage = round(len(matched) / len(gt_words) * 100, 1) if gt_words else 0

    return {
        "metric": "M15_gt_coverage",
        "value_percent": coverage,
        "matched_count": len(matched),
        "gt_word_count": len(gt_words),
        "interpretation": (
            "High coverage" if coverage >= 60 else
            "Moderate coverage" if coverage >= 35 else
            "Low coverage"
        )
    }


# ═══════════════════════════════════════════════════════
# M16 — E2E LATENCY
# ═══════════════════════════════════════════════════════

def measure_e2e_latency(retrieval_latency_s: float, generation_latency_s: float) -> dict:
    """
    M16: Total query-to-answer time.
    Pass in the latency values already measured from retrieval and LLM steps.
    """
    total = round(retrieval_latency_s + generation_latency_s, 3)
    return {
        "metric": "M16_e2e_latency",
        "retrieval_s": round(retrieval_latency_s, 3),
        "generation_s": round(generation_latency_s, 3),
        "total_s": total,
        "interpretation": (
            "Fast" if total < 5 else
            "Acceptable" if total < 15 else
            "Slow — may impact clinical usability"
        )
    }


# ═══════════════════════════════════════════════════════
# M17 — THROUGHPUT
# ═══════════════════════════════════════════════════════

def throughput(e2e_latency_s: float) -> dict:
    """
    M17: Queries per second the system can handle.
    """
    qps = round(1.0 / e2e_latency_s, 4) if e2e_latency_s > 0 else 0
    return {
        "metric": "M17_throughput",
        "queries_per_second": qps,
        "interpretation": (
            "High throughput" if qps > 0.5 else
            "Low throughput — system is slow"
        )
    }


# ═══════════════════════════════════════════════════════
# M18 — CPU USAGE
# ═══════════════════════════════════════════════════════

def cpu_usage() -> dict:
    """
    M18: Current CPU utilisation percentage.
    Call before and after a query to see load.
    """
    try:
        cpu = psutil.cpu_percent(interval=1)
        return {
            "metric": "M18_cpu_usage",
            "value_percent": round(cpu, 1),
            "interpretation": (
                "Normal" if cpu < 70 else
                "High load" if cpu < 90 else
                "Critical — may cause latency"
            )
        }
    except Exception as e:
        return {"metric": "M18_cpu_usage", "value_percent": None, "error": str(e)}


# ═══════════════════════════════════════════════════════
# M19 — RAM USAGE
# ═══════════════════════════════════════════════════════

def ram_usage() -> dict:
    """
    M19: Current RAM consumption in GB.
    """
    try:
        mem = psutil.virtual_memory()
        used_gb = round(mem.used / (1024 ** 3), 2)
        total_gb = round(mem.total / (1024 ** 3), 2)
        percent = round(mem.percent, 1)
        return {
            "metric": "M19_ram_usage",
            "used_gb": used_gb,
            "total_gb": total_gb,
            "percent": percent,
            "interpretation": (
                "Normal" if percent < 70 else
                "High" if percent < 85 else
                "Critical"
            )
        }
    except Exception as e:
        return {"metric": "M19_ram_usage", "used_gb": None, "error": str(e)}


# ═══════════════════════════════════════════════════════
# MASTER EVALUATOR — run all metrics for one benchmark result
# ═══════════════════════════════════════════════════════

def evaluate_result(
    model_answer: str,
    gold_answer_text: str,
    retrieved_chunks: list,
    expected_keywords: list,
    retrieval_latency_s: float = 0.0,
    generation_latency_s: float = 0.0,
) -> dict:
    """
    Run all selected metrics for one model response.
    Call this for both Mistral and Groq results.

    Args:
        model_answer: The model's text response
        gold_answer_text: The full gold standard reasoning + actions as one string
        retrieved_chunks: List of dicts from retrieve_rag_context()
        expected_keywords: List of keywords from base_answer["key_keywords"]
        retrieval_latency_s: Time taken by RAG retrieval
        generation_latency_s: Time taken by LLM (model latency)
    """
    # Build gold text from structured base answer if needed
    query_text = retrieved_chunks[0].get("content", "") if retrieved_chunks else ""

    # Cosine sim: query vs best retrieved chunk
    best_chunk_text = retrieved_chunks[0].get("content", "") if retrieved_chunks else ""

    m16 = measure_e2e_latency(retrieval_latency_s, generation_latency_s)

    metrics = {
        "M4_cosine_similarity": cosine_similarity_score(
            " ".join(expected_keywords), best_chunk_text
        )["value"],
        "M5_topk_accuracy": topk_accuracy(retrieved_chunks, expected_keywords)["value_percent"],
        "M6_rouge1_f1": rouge1_score(model_answer, gold_answer_text)["value"],
        "M8_rouge_l_f1": rouge_l_score(model_answer, gold_answer_text)["value"],
        "M12_bertscore_f1": bertscore(model_answer, gold_answer_text)["value"],
        "M14_faithfulness": faithfulness_score(model_answer, retrieved_chunks)["value_percent"],
        "M15_gt_coverage": gt_coverage(model_answer, gold_answer_text)["value_percent"],
        "M16_e2e_latency_s": m16["total_s"],
        "M17_throughput_qps": throughput(m16["total_s"])["queries_per_second"],
        "M18_cpu_percent": cpu_usage()["value_percent"],
        "M19_ram_gb": ram_usage()["used_gb"],
    }

    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in metrics.items()}
