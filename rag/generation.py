import requests


def generate_answer(context, question, triage_level):

    # 🔥 LIMIT CONTEXT SIZE
    context = context[:1200]

    prompt = f"""
You are a medical triage assistant.

CRITICAL RULES:
- Answer must be VERY SHORT
- Maximum 3-4 lines total
- DO NOT add extra steps
- DO NOT mention advanced medical procedures (ACLS, oxygen, etc.)
- Focus ONLY on what a normal person should do
- If triage is RED, ALWAYS mention "medical emergency" or "life-threatening"
If triage is GREEN, ALWAYS include:
- rest advice
- AND "seek medical help if symptoms worsen"

-----------------------
PATIENT QUERY:
{question}

TRIAGE LEVEL: {triage_level}

-----------------------

FORMAT:

Triage Level: {triage_level}

Reason:
(1 short sentence explaining the problem AND clearly mention if it is serious, e.g., "life-threatening" or "medical emergency" if applicable)

What to do:
- Give simple advice
- If triage is RED → immediate emergency action
- If triage is GREEN → include:
  "monitor symptoms and seek medical help if condition worsens"

-----------------------

GOOD EXAMPLE:

Triage Level: RED

Reason:
The patient is not breathing, which is life-threatening.

What to do:
Call emergency services immediately.

-----------------------
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,  # 🔥 more stable + less overthinking
            },
            timeout=300,
        )

        if response.status_code != 200:
            return f"⚠ API Error: {response.text}"

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            return "⚠ Empty response from model"

        return answer.strip()

    except Exception as e:
        print("❌ Generation error:", e)
        return "⚠ Error generating medical response."
