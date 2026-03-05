import requests


def classify_triage(query):

    prompt = f"""
You are an emergency department triage assistant.

Classify the patient condition into ONE category:

RED - Life threatening (cardiac arrest, severe breathing distress, shock, unconscious)

YELLOW - Urgent but stable (infection, moderate pain, abnormal vitals)

GREEN - Minor / informational

Return ONLY one word: RED, YELLOW, or GREEN.

Patient description:
{query}
"""

    try:
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={"model": "phi3:mini", "prompt": prompt, "stream": False},
        )

        result = response.json()["response"].strip().upper()

        if "RED" in result:
            return "RED"
        elif "YELLOW" in result:
            return "YELLOW"
        else:
            return "GREEN"

    except:
        return "YELLOW"
