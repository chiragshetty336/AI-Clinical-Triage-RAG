import requests


def classify_triage(query):

    query_lower = query.lower()

    # 🔥 STRONG RULE-BASED TRIAGE

    # RED (life-threatening)
    if any(
        x in query_lower
        for x in [
            "unconscious",
            "not breathing",
            "severe chest pain",
            "shortness of breath",
            "cannot breathe",
            "severe bleeding",
        ]
    ):
        return "RED"

    # YELLOW (urgent)
    if any(
        x in query_lower
        for x in ["moderate pain", "persistent fever", "vomiting", "infection"]
    ):
        return "YELLOW"

    # GREEN (minor)
    if any(
        x in query_lower for x in ["mild fever", "mild pain", "small cut", "headache"]
    ):
        return "GREEN"

    # ------------------------------
    # LLM fallback (RARE USE)
    # ------------------------------
    prompt = f"""
Classify into ONE: RED, YELLOW, GREEN.

Patient:
{query}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:7b", "prompt": prompt, "stream": False},
            timeout=60,
        )

        result = response.json().get("response", "").strip().upper()

        if "RED" in result:
            return "RED"
        elif "YELLOW" in result:
            return "YELLOW"
        else:
            return "GREEN"

    except Exception as e:
        print("⚠ Triage fallback error:", e)
        return "YELLOW"
