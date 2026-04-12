"""
Medical Triage Benchmark Dataset v2
Queries written to match language in Australian Triage Education Kit,
AIIMS Triage Guidelines, Emergency Severity Index, and field triage PDFs.
Keywords extracted from the actual text patterns found in these documents.
"""

BENCHMARK_CASES = [
    {
        "id": "case_001",
        "category": "Chest Pain Triage",
        # Language mirrors the ATS triage education kit case studies (MI + cardiac stents case)
        "query": (
            "A 52-year-old male presents to emergency with acute chest pain "
            "that is severe, central, and crushing. He is diaphoretic and pale. "
            "Heart rate is 110 bpm, blood pressure 90/60 mmHg, oxygen saturation 92%. "
            "He had a myocardial infarction and cardiac stents 10 days ago. "
            "What is the triage category and immediate nursing actions?"
        ),
        "base_answer": {
            "triage_level": "RED",
            "triage_reasoning": (
                "Immediate life-threatening presentation. Haemodynamic instability "
                "(hypotension + tachycardia), low SpO2, and prior cardiac history indicate "
                "acute coronary syndrome with cardiogenic compromise. ATS Category 1 — "
                "must be seen immediately."
            ),
            "key_actions": [
                "Immediate triage to resuscitation bay",
                "Continuous cardiac monitoring and ECG",
                "IV access, oxygen therapy",
                "Aspirin 300mg if no contraindication",
                "Notify treating team immediately",
                "Monitor blood pressure and SpO2 continuously",
                "Prepare for urgent intervention"
            ],
            "key_keywords": [
                "RED", "triage", "immediate", "resuscitation", "cardiac monitoring",
                "ECG", "oxygen", "aspirin", "IV access", "haemodynamic",
                "tachycardia", "hypotension", "ATS", "Category 1"
            ],
            "time_to_treatment": "Immediate",
            "disposition": "Resuscitation bay"
        }
    },
    {
        "id": "case_002",
        "category": "Blunt Abdominal Trauma",
        # Matches the blunt trauma/tachypnoeic case from emergency_triage_education_kit p.226
        "query": (
            "A 35-year-old female arrives via ambulance following a motor vehicle "
            "accident. Airway is clear. She is tachypnoeic with respiratory rate 28 "
            "and tachycardic at HR 120 bpm. She has significant blunt trauma to the "
            "abdomen with possible liver injury. GCS is 14. BP 100/70 mmHg. "
            "Assign triage category and list priority nursing interventions."
        ),
        "base_answer": {
            "triage_level": "RED",
            "triage_reasoning": (
                "Significant blunt abdominal trauma with tachypnoea, tachycardia, "
                "and haemodynamic compromise. High risk of internal haemorrhage "
                "from liver injury. ATS Category 1-2. Airway currently intact "
                "but patient may deteriorate rapidly."
            ),
            "key_actions": [
                "Immediate trauma assessment — primary survey ABCDE",
                "Two large-bore IV access, fluid resuscitation",
                "Oxygen therapy, continuous monitoring",
                "Urgent surgical review for abdominal injury",
                "Analgesia as per protocol",
                "Cervical spine precautions",
                "FAST ultrasound for intra-abdominal haemorrhage"
            ],
            "key_keywords": [
                "RED", "triage", "trauma", "tachypnoeic", "tachycardic", "abdominal",
                "liver", "IV access", "oxygen", "monitoring", "primary survey",
                "haemorrhage", "immediate", "resuscitation", "ABCDE"
            ],
            "time_to_treatment": "Immediate",
            "disposition": "Trauma bay"
        }
    },
    {
        "id": "case_003",
        "category": "Mild Pain — Lower Urgency",
        # Matches the mild pain / circulation intact case from education kit p.227
        "query": (
            "A 45-year-old female presents with mild pain in her right wrist after "
            "a fall at home. Airway, breathing, and circulation are intact. "
            "She rates her pain 4 out of 10. Vital signs are normal. "
            "There is mild swelling over the wrist but no neurovascular deficit. "
            "What triage category should be assigned and when should treatment commence?"
        ),
        "base_answer": {
            "triage_level": "GREEN",
            "triage_reasoning": (
                "Airway, breathing and circulation intact. Mild pain is the main problem. "
                "No haemodynamic compromise. ATS Category 4 — patient should commence "
                "treatment within 60 minutes. Non-urgent presentation suitable for "
                "fast track or general waiting."
            ),
            "key_actions": [
                "Triage Category 4 — commence treatment within 60 minutes",
                "Analgesia for pain management",
                "X-ray to exclude fracture",
                "Neurovascular observations of the limb",
                "Reassess if pain worsens or neurovascular status changes",
                "Splinting if required"
            ],
            "key_keywords": [
                "GREEN", "triage", "mild pain", "circulation", "intact", "Category 4",
                "60 minutes", "analgesia", "X-ray", "non-urgent", "commence treatment",
                "neurovascular", "reassess", "ATS"
            ],
            "time_to_treatment": "Within 60 minutes",
            "disposition": "Fast track / general waiting"
        }
    },
    {
        "id": "case_004",
        "category": "Mental Health Emergency",
        # Matches mental health emergencies reference in triage education kit
        "query": (
            "A 28-year-old male is brought to emergency by police. He is agitated, "
            "verbally aggressive, and making threats of self-harm. He is pacing the "
            "waiting room and refuses to engage with staff. No obvious physical injury. "
            "Vital signs cannot be obtained due to non-compliance. "
            "What is the triage category and what immediate actions should be taken?"
        ),
        "base_answer": {
            "triage_level": "YELLOW",
            "triage_reasoning": (
                "Acute mental health emergency with agitation and self-harm threats. "
                "Patient is potentially dangerous and requires urgent psychiatric assessment. "
                "ATS Category 2-3 for mental health presentations with immediate risk. "
                "Safety of patient and staff is the first priority."
            ),
            "key_actions": [
                "Ensure safety of patient, staff, and other patients immediately",
                "Alert security and senior staff",
                "De-escalation techniques — calm non-threatening approach",
                "Mental health team notification",
                "Attempt to obtain vital signs when safe to do so",
                "Document behaviour and history from police",
                "Prepare for involuntary assessment if required"
            ],
            "key_keywords": [
                "YELLOW", "triage", "mental health", "agitated", "self-harm",
                "safety", "de-escalation", "urgent", "assessment",
                "Category 2", "aggressive", "security", "violent"
            ],
            "time_to_treatment": "Within 10-30 minutes",
            "disposition": "Mental health assessment room"
        }
    },
    {
        "id": "case_005",
        "category": "Sepsis Recognition",
        # Matches the sepsis case study language in emergency_triage_education_kit
        "query": (
            "A 67-year-old female with known diabetes presents to triage confused "
            "and febrile with temperature 38.9 degrees. Heart rate 118 bpm, respiratory "
            "rate 24 per minute, blood pressure 95/65 mmHg, SpO2 94% on room air. "
            "Her family says she has been unwell for 2 days with a urinary tract infection. "
            "She is difficult to rouse. Assign triage level and immediate management."
        ),
        "base_answer": {
            "triage_level": "RED",
            "triage_reasoning": (
                "Sepsis with end-organ compromise — altered level of consciousness, "
                "haemodynamic instability, high fever, tachycardia, and tachypnoea "
                "in a diabetic patient with source of infection. High risk of septic "
                "shock. ATS Category 1-2. Sepsis bundle must be initiated immediately."
            ),
            "key_actions": [
                "Immediate medical review — ATS Category 1",
                "IV access, blood cultures before antibiotics",
                "IV antibiotics within 1 hour of presentation",
                "IV fluid resuscitation",
                "Oxygen to maintain SpO2 above 94%",
                "Urinary catheter for fluid balance monitoring",
                "Lactate measurement and blood glucose",
                "ICU or HDU notification"
            ],
            "key_keywords": [
                "RED", "triage", "sepsis", "confused", "febrile", "haemodynamic",
                "antibiotics", "IV access", "blood cultures", "fluid resuscitation",
                "oxygen", "immediate", "Category 1", "tachycardia", "tachypnoea"
            ],
            "time_to_treatment": "Immediate",
            "disposition": "Resuscitation bay / ICU"
        }
    },
    {
        "id": "case_006",
        "category": "Respiratory Distress",
        "query": (
            "A 19-year-old male with known asthma is brought in by his mother. "
            "He cannot complete a full sentence due to breathlessness. "
            "Respiratory rate is 30 per minute, SpO2 88% on room air. "
            "He is using accessory muscles. On auscultation there is a silent chest. "
            "He appears very distressed. Assign triage category and emergency management."
        ),
        "base_answer": {
            "triage_level": "RED",
            "triage_reasoning": (
                "Life-threatening asthma attack. Silent chest, inability to speak in "
                "full sentences, SpO2 below 90%, and use of accessory muscles indicate "
                "severe respiratory failure. ATS Category 1 — immediate intervention "
                "required to prevent respiratory arrest."
            ),
            "key_actions": [
                "Immediate placement in resuscitation bay — ATS Category 1",
                "High-flow oxygen via non-rebreather mask",
                "Continuous nebulised salbutamol",
                "IV or IM corticosteroids",
                "IV access and continuous monitoring",
                "Senior physician review immediately",
                "Prepare for intubation if patient deteriorates"
            ],
            "key_keywords": [
                "RED", "triage", "asthma", "respiratory", "oxygen", "salbutamol",
                "nebulised", "accessory muscles", "silent chest", "immediate",
                "Category 1", "corticosteroids", "IV access", "monitoring"
            ],
            "time_to_treatment": "Immediate",
            "disposition": "Resuscitation bay"
        }
    },
]


def get_benchmark_case(case_id: str) -> dict:
    for case in BENCHMARK_CASES:
        if case["id"] == case_id:
            return case
    return None


def get_all_cases() -> list:
    return BENCHMARK_CASES


def score_against_base(model_response: str, base_answer: dict) -> dict:
    """
    Score a model response against the gold-standard base answer.
    4 dimensions totalling 100 points.
    """
    import re
    response_upper = model_response.upper()
    response_lower = model_response.lower()

    scores = {}

    # 1. Triage Level Match (30 points)
    correct_triage = base_answer["triage_level"]
    if correct_triage in response_upper:
        scores["triage_match"] = 30
        scores["triage_correct"] = True
    else:
        scores["triage_match"] = 0
        scores["triage_correct"] = False

    # 2. Keyword Coverage from gold answer (40 points)
    keywords = base_answer["key_keywords"]
    matched_keywords = []
    missed_keywords = []
    for kw in keywords:
        if kw.lower() in response_lower:
            matched_keywords.append(kw)
        else:
            missed_keywords.append(kw)

    keyword_score = (len(matched_keywords) / len(keywords)) * 40 if keywords else 0
    scores["keyword_coverage"] = round(keyword_score, 1)
    scores["matched_keywords"] = matched_keywords
    scores["missed_keywords"] = missed_keywords
    scores["keyword_ratio"] = f"{len(matched_keywords)}/{len(keywords)}"

    # 3. Clinical Reasoning Quality (20 points)
    reasoning_terms = [
        "assess", "monitor", "immediate", "urgent", "risk", "due to",
        "because", "indicates", "suggests", "consistent with", "consider",
        "recommend", "administer", "initiate", "establish", "triage",
        "category", "priority", "intervention", "management"
    ]
    reasoning_count = sum(1 for term in reasoning_terms if term in response_lower)
    reasoning_score = min(20, reasoning_count * 2)
    scores["clinical_reasoning"] = reasoning_score

    # 4. Action Specificity (10 points)
    has_numbers = bool(re.search(r'\d+\s*(mg|ml|L|min|hours?|bpm|mmHg|%|\/min)', model_response))
    has_drug_or_action = any(term in response_lower for term in [
        "oxygen", "iv", "aspirin", "salbutamol", "morphine", "saline",
        "monitoring", "ecg", "antibiotics", "fluid", "nebul", "catheter",
        "intubat", "resuscit", "x-ray", "ultrasound"
    ])
    specificity_score = 0
    if has_numbers:
        specificity_score += 5
    if has_drug_or_action:
        specificity_score += 5
    scores["action_specificity"] = specificity_score

    # Total
    total = (
        scores["triage_match"] +
        scores["keyword_coverage"] +
        scores["clinical_reasoning"] +
        scores["action_specificity"]
    )
    scores["total"] = round(total, 1)
    scores["grade"] = (
        "Excellent" if total >= 80 else
        "Good" if total >= 65 else
        "Fair" if total >= 50 else
        "Poor"
    )

    return scores
