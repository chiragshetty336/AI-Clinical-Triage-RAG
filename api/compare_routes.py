"""
api/compare_routes.py
FastAPI router — LLM comparison + base answer similarity scoring.
"""

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from llm_compare import compare_llms, compute_similarity

router = APIRouter(prefix="/compare", tags=["LLM Comparison"])


# ── Base Answers (UNCHANGED) ────────────────────────────────────────────────

BASE_ANSWERS = [
    # (same content — no change)
    "This patient presents with crushing chest pain radiating to the left arm accompanied by diaphoresis, tachycardia at 130 beats per minute and hypotension with systolic BP of 90 mmHg. These findings are consistent with an acute coronary syndrome complicated by cardiogenic shock. Immediate resuscitation is required. The clinical team should ensure airway patency, administer high-flow oxygen, establish two large-bore intravenous lines, obtain a 12-lead ECG urgently, and administer aspirin. Continuous cardiac monitoring is essential. Cardiology must be alerted immediately for possible percutaneous coronary intervention. Do not delay transfer to a cardiac catheterization lab.",
    "A sudden onset severe headache described as the worst of the patient life combined with nuchal rigidity and photophobia raises strong clinical suspicion for subarachnoid hemorrhage or bacterial meningitis. Both conditions are life threatening and require immediate investigation. A non-contrast CT scan of the head should be performed urgently. If CT is negative and meningitis is suspected a lumbar puncture should follow. Blood cultures should be drawn before initiating broad-spectrum intravenous antibiotics. The patient must be monitored continuously and neurosurgical consultation is warranted.",
    "The patient has known asthma and presents with moderate dyspnoea, oxygen saturation of 92 percent and is speaking in short sentences indicating significant respiratory distress. This is an acute moderate to severe asthma exacerbation. Supplemental oxygen should be administered immediately to maintain saturation above 94 percent. Salbutamol via nebulizer should be given back to back and ipratropium bromide added. Systemic corticosteroids should be started without delay. Continuous pulse oximetry and peak flow monitoring are essential. If there is no improvement prepare for escalation to high dependency care.",
    "An 8 year old child presenting with a temperature of 38.5 degrees Celsius and sore throat with mild odynophagia without stridor requires assessment of the upper airway. The absence of stridor is reassuring against epiglottitis. The presentation is consistent with acute tonsillitis or pharyngitis. A throat swab should be obtained. If Group A Streptococcus is suspected a course of amoxicillin is appropriate. Adequate oral hydration should be ensured and paracetamol given for fever relief. Parents should be advised on warning signs including drooling or breathing difficulty.",
    "The patient has sustained a mild ankle sprain with no difficulty bearing weight and no visible swelling. Ottawa ankle rules do not indicate the need for radiographic imaging. Management follows the PRICE protocol: Protection, Rest, Ice application for 15 to 20 minutes every two hours, Compression with an elastic bandage, and Elevation of the limb. Oral anti-inflammatory medication may be taken if not contraindicated. The patient should mobilise as tolerated and follow up if symptoms persist beyond one week.",
    "The patient presents with a 3 day history of cough, rhinorrhoea and a temperature of 37.2 degrees which is within normal limits. There is no evidence of respiratory distress. This presentation is consistent with an uncomplicated upper respiratory tract infection of viral aetiology. Antibiotics are not indicated. Management is supportive including adequate fluid intake, rest, and paracetamol for discomfort. The patient should seek review if symptoms persist beyond two weeks or worsen.",
    "The patient is an elderly woman found unresponsive with no pulse and no spontaneous breathing with bystander CPR in progress. The resuscitation team must take over with high-quality CPR ensuring adequate depth and rate. A defibrillator should be attached immediately to assess cardiac rhythm. If a shockable rhythm is identified defibrillation should be delivered without delay. Adrenaline 1mg intravenously should be administered every 3 to 5 minutes. Airway management should be established. The team should follow the Advanced Life Support algorithm and identify reversible causes.",
    "The patient has type 2 diabetes with blood glucose of 380 mg per dL, mild nausea but remains alert and oriented. Assessment should exclude diabetic ketoacidosis or hyperosmolar hyperglycaemic state. Urinalysis for ketones and blood ketone measurement should be performed along with a metabolic panel. If ketones are absent and the patient is clinically stable cautious oral rehydration alongside insulin adjustment is appropriate. Medications and dietary compliance should be reviewed. If ketones are elevated intravenous fluid resuscitation and insulin infusion per DKA protocol should be initiated.",
]


# ── Models ────────────────────────────────────────────────────────────────


class CompareRequest(BaseModel):
    query: str
    context: Optional[str] = ""


class BaseScoreRequest(BaseModel):
    query_index: int
    mistral_answer: str
    groq_answer: str  # ✅ CHANGED


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/")
async def compare_endpoint(request: CompareRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return compare_llms(request.query, request.context or "")


@router.post("/score-against-base")
async def score_against_base(request: BaseScoreRequest):
    idx = request.query_index
    if idx < 0 or idx >= len(BASE_ANSWERS):
        raise HTTPException(
            status_code=400, detail=f"query_index must be 0 to {len(BASE_ANSWERS)-1}"
        )

    base = BASE_ANSWERS[idx]

    m_scores = compute_similarity(base, request.mistral_answer)
    g_scores = compute_similarity(base, request.groq_answer)  # ✅ CHANGED

    return {
        "query_index": idx,
        "mistral_vs_base": {
            "composite_score": m_scores.get("composite_score"),
            "semantic_similarity": m_scores.get("semantic_similarity"),
            "rouge1_f1": m_scores.get("rouge1_f1"),
            "jaccard_overlap": m_scores.get("jaccard_overlap"),
        },
        "groq_vs_base": {  # ✅ CHANGED
            "composite_score": g_scores.get("composite_score"),
            "semantic_similarity": g_scores.get("semantic_similarity"),
            "rouge1_f1": g_scores.get("rouge1_f1"),
            "jaccard_overlap": g_scores.get("jaccard_overlap"),
        },
    }


@router.get("/health")
async def health_check():
    import requests as req

    status = {
        "mistral_ollama": False,
        "groq_key_set": False,  # ✅ CHANGED
    }

    # Check Ollama
    try:
        r = req.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), timeout=3)
        status["mistral_ollama"] = r.status_code == 200
    except Exception:
        pass

    # Check Groq key
    key = os.getenv("GROQ_API_KEY", "")
    status["groq_key_set"] = bool(key) and key.startswith("gsk_")  # ✅ CHANGED

    return status
