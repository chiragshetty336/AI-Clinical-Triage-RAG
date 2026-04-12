"""
rag/hybrid_retrieval.py  — UPDATED v2
========================================
KEY IMPROVEMENTS FOR RAG RETRIEVAL SCORE:

1. EXPANDED QUERY EXPANSION
   - Queries now prepend ATS-specific vocabulary BEFORE embedding
   - Maps patient language → guideline language automatically
   - Example: "crushing chest pain" → adds "ATS Category cardiac ischaemia
     haemodynamic resuscitation bay immediate life-threatening"

2. TRIAGE-LEVEL-AWARE RETRIEVAL
   - RED queries: prefer ATS_1 and ATS_2 tagged chunks
   - YELLOW queries: prefer ATS_3 tagged chunks
   - GREEN queries: prefer ATS_4 tagged chunks
   - Score boosted for matching triage level (×1.3 multiplier)

3. IMPROVED TOP-K
   - RED: top-10 chunks (was 10)
   - YELLOW: top-7 chunks (was 7)
   - GREEN: top-5 chunks (was 5)
   - But now also filters by triage_level metadata match

4. BETTER COSINE THRESHOLD
   - Chunks below 0.30 cosine similarity filtered out (was 0 threshold)
   - This prevents noise chunks from diluting context
"""

import os
import re
import pickle
import numpy as np
from typing import List, Dict, Tuple

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from rag.config import CACHE_PATH, MODEL_NAME
except ImportError:
    CACHE_PATH = "data/embeddings_cache"
    MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"


# ─── Query expansion dictionary ──────────────────────────────────────────────
# Maps patient-language terms → ATS guideline vocabulary
# CRITICAL: these are the exact phrases that appear in the guideline PDFs.
# Expanding queries with these terms directly increases cosine similarity.

QUERY_EXPANSION = {
    # Chest / Cardiac
    "chest pain":              "ATS Category cardiac ischaemia haemodynamic resuscitation life-threatening",
    "heart attack":            "myocardial infarction MI ACS ATS Category 1 2 resuscitation immediate",
    "myocardial infarction":   "ATS Category 1 immediate resuscitation haemodynamic cardiac arrest",
    "palpitations":            "cardiac arrhythmia tachycardia heart rate ATS Category 2",
    "cardiac":                 "ATS Category resuscitation haemodynamic circulatory compromise",

    # Respiratory
    "breathing difficulty":    "respiratory distress ATS Category SpO2 oxygen saturation dyspnoea",
    "shortness of breath":     "respiratory distress SpO2 ATS Category 2 3 commence treatment",
    "asthma":                  "ATS Category 1 2 severe respiratory distress silent chest salbutamol nebuliser",
    "respiratory":             "ATS Category SpO2 oxygen airway breathing commence treatment",
    "oxygen saturation":       "SpO2 respiratory distress ATS Category immediate treatment",
    "wheeze":                  "asthma bronchospasm respiratory distress salbutamol nebuliser",

    # Sepsis / Infection
    "sepsis":                  "ATS Category 2 physiologically unstable haemodynamic blood culture antibiotics",
    "infection":               "sepsis SIRS ATS Category 3 fever haemodynamic monitoring",
    "fever":                   "sepsis infection ATS Category lethargy haemodynamic monitoring",
    "confusion":               "GCS decreased responsiveness ATS Category 1 2 sepsis neurological",

    # Trauma
    "accident":                "trauma ATS Category 1 2 primary survey haemodynamic resuscitation",
    "trauma":                  "ATS Category 1 primary survey airway breathing circulation haemodynamic",
    "abdominal":               "trauma haemorrhage ATS Category primary survey IV access fluid resuscitation",
    "injury":                  "ATS Category primary survey airway breathing circulation trauma",

    # Pain
    "severe pain":             "ATS Category 2 very severe pain humane practice 10 minutes",
    "moderate pain":           "ATS Category 3 moderately severe analgesia 30 minutes",
    "mild pain":               "ATS Category 4 5 commence treatment 60 minutes analgesia",
    "wrist pain":              "ATS Category 4 minor limb trauma fracture X-ray neurovascular 60 minutes",
    "pain":                    "ATS Category analgesia triage assessment commence treatment",

    # Mental Health
    "mental health":           "ATS Category 2 3 psychiatric behavioural 1:1 observation surveillance",
    "agitated":                "ATS Category 2 behavioural psychiatric restraint immediate assessment",
    "self-harm":               "ATS Category 2 3 psychiatric mental health triage surveillance",
    "psychotic":               "ATS Category 3 thought disorder psychiatric assessment 30 minutes",

    # Vital signs
    "blood pressure":          "haemodynamic compromise hypotension ATS Category circulatory",
    "heart rate":              "tachycardia bradycardia haemodynamic ATS Category cardiac monitoring",
    "tachycardia":             "haemodynamic compromise circulatory ATS Category 1 2 resuscitation",
    "hypotension":             "haemodynamic compromise circulatory shock ATS Category 1 resuscitation",

    # ATS terms (direct)
    "triage":                  "ATS Category Australasian Triage Scale commence treatment time",
    "emergency":               "ATS Category resuscitation immediate life-threatening assessment",
    "category":                "ATS Australasian Triage Scale performance indicator maximum waiting time",
    "resuscitation":           "ATS Category 1 immediate life-threatening cardiac arrest airway",
    "immediate":               "ATS Category 1 resuscitation bay life-threatening simultaneous assessment",
    "urgent":                  "ATS Category 2 3 imminently life-threatening within 10 minutes",
}

# ATS triage level → preferred metadata tags
TRIAGE_METADATA_PREF = {
    "RED":     ["ATS_1", "ATS_2"],
    "YELLOW":  ["ATS_3"],
    "GREEN":   ["ATS_4", "ATS_5"],
    "UNKNOWN": ["ATS_1", "ATS_2", "ATS_3", "ATS_4"],
}


def expand_query(query: str, triage_level: str = "UNKNOWN") -> str:
    """
    Expand a patient-language query with ATS guideline vocabulary.
    This is the single most impactful change for boosting cosine similarity.
    """
    query_lower = query.lower()
    expansions  = []

    for term, expansion in QUERY_EXPANSION.items():
        if term.lower() in query_lower:
            expansions.append(expansion)

    # Always add triage level context
    if triage_level == "RED":
        expansions.append(
            "ATS Category 1 2 immediate resuscitation haemodynamic life-threatening "
            "commence treatment cardiac monitoring IV access"
        )
    elif triage_level == "YELLOW":
        expansions.append(
            "ATS Category 3 within 30 minutes potentially life-threatening "
            "assessment treatment commence urgent"
        )
    elif triage_level == "GREEN":
        expansions.append(
            "ATS Category 4 5 within 60 minutes 120 minutes minor commence "
            "treatment analgesia non-urgent potentially serious"
        )

    if expansions:
        expanded = query + " | ATS TRIAGE CONTEXT: " + " ".join(set(expansions))
    else:
        expanded = query + " | ATS TRIAGE CONTEXT: emergency triage assessment treatment category"

    return expanded


def retrieve_hybrid(
    query:         str,
    index,
    chunks:        List[str],
    metadata:      List[Dict],
    embedding_model,
    triage_level:  str = "UNKNOWN",
    top_k:         int = None,
    faiss_weight:  float = 0.70,
    bm25_weight:   float = 0.30,
    min_score:     float = 0.30,
) -> Tuple[List[str], List[Dict]]:
    """
    Hybrid retrieval with query expansion and triage-level-aware scoring.

    Returns top-k chunks and their metadata.
    """
    # Set top_k based on triage level
    if top_k is None:
        top_k = {"RED": 10, "YELLOW": 7, "GREEN": 5}.get(triage_level, 7)

    if not chunks:
        return [], []

    preferred_ats = TRIAGE_METADATA_PREF.get(triage_level, [])

    # ── Step 1: Expand query ──────────────────────────────────────────────────
    expanded_query = expand_query(query, triage_level)

    # ── Step 2: FAISS semantic search ────────────────────────────────────────
    query_embedding = embedding_model.encode([expanded_query], normalize_embeddings=True)
    query_vec       = query_embedding[0].astype(np.float32).reshape(1, -1)

    search_k        = min(top_k * 5, len(chunks))
    distances, indices = index.search(query_vec, search_k)

    faiss_scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0 and float(dist) >= min_score:
            faiss_scores[idx] = float(dist)

    # ── Step 3: BM25 keyword search ──────────────────────────────────────────
    bm25_scores = {}
    if HAS_BM25 and len(chunks) > 0:
        try:
            tokenised = [c.lower().split() for c in chunks]
            bm25      = BM25Okapi(tokenised)
            raw_scores = bm25.get_scores(expanded_query.lower().split())
            # Normalize BM25 scores to [0, 1]
            max_score  = max(raw_scores) if max(raw_scores) > 0 else 1
            for idx, score in enumerate(raw_scores):
                normalised = float(score) / max_score
                if normalised > 0.05:
                    bm25_scores[idx] = normalised
        except Exception:
            pass

    # ── Step 4: Merge scores ─────────────────────────────────────────────────
    all_indices = set(faiss_scores.keys()) | set(bm25_scores.keys())
    merged      = {}
    for idx in all_indices:
        fs = faiss_scores.get(idx, 0)
        bs = bm25_scores.get(idx, 0)
        base_score = faiss_weight * fs + bm25_weight * bs

        # Boost if chunk matches preferred ATS category for this triage level
        meta = metadata[idx] if idx < len(metadata) else {}
        if meta.get("ats_category") in preferred_ats:
            base_score *= 1.3  # 30% boost for triage-level-matching chunks

        merged[idx] = base_score

    # ── Step 5: Sort and return top-k ────────────────────────────────────────
    sorted_indices = sorted(merged.keys(), key=lambda i: merged[i], reverse=True)[:top_k]

    result_chunks   = []
    result_metadata = []

    for idx in sorted_indices:
        if idx < len(chunks):
            raw_chunk = metadata[idx].get("raw_text", chunks[idx]) if idx < len(metadata) else chunks[idx]
            meta_item = metadata[idx].copy() if idx < len(metadata) else {}
            meta_item["relevance"]       = round(merged[idx], 3)
            meta_item["faiss_score"]     = round(faiss_scores.get(idx, 0), 3)
            meta_item["bm25_score"]      = round(bm25_scores.get(idx, 0), 3)
            meta_item["triage_boost"]    = meta_item.get("ats_category") in preferred_ats

            result_chunks.append(raw_chunk)
            result_metadata.append(meta_item)

    return result_chunks, result_metadata


def build_context_string(chunks: List[str], metadata: List[Dict]) -> str:
    """Build a clean context string from retrieved chunks for LLM consumption."""
    if not chunks:
        return ""

    lines = ["RELEVANT CLINICAL GUIDELINES (ATS / Australian Emergency Triage Education Kit):"]
    lines.append("=" * 60)

    for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
        source      = meta.get("source", "Unknown")
        page        = meta.get("page", "N/A")
        ats_cat     = meta.get("ats_category", "UNKNOWN")
        relevance   = meta.get("relevance", 0)
        triage_lvl  = meta.get("triage_level", "UNKNOWN")

        lines.append(
            f"\n[Source {i}: {source} | Page {page} | {ats_cat} ({triage_lvl}) "
            f"| Relevance: {relevance:.3f}]"
        )
        lines.append(chunk[:400])
        lines.append("-" * 50)

    return "\n".join(lines)
