import requests


def generate_answer(context, question):

    # 🔥 LIMIT CONTEXT SIZE (keep only useful part)
    context = context[:1500]

    prompt = f"""
You are a medical triage assistant helping a normal person.

Your job is to give:
- SIMPLE
- CLEAR
- EASY TO UNDERSTAND answers

IMPORTANT RULES:
- Do NOT use complex medical terms
- Do NOT copy raw text from context
- Explain in plain English
- Be calm and direct

-----------------------
CONTEXT:
{context}
-----------------------

PATIENT QUERY:
{question}

-----------------------

OUTPUT FORMAT (STRICTLY FOLLOW):

Triage Level: RED or YELLOW or GREEN

Reason:
Explain in 1-2 simple sentences what might be happening.

What to do:
Give clear and practical next steps.

-----------------------

EXAMPLE:

Triage Level: RED

Reason:
The patient is not breathing, which is a life-threatening emergency.

What to do:
Call emergency services immediately and start CPR if trained.

-----------------------

Now generate the answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,  # 🔥 makes output stable & less random
            },
            timeout=300,
        )

        if response.status_code != 200:
            return f"⚠ API Error: {response.text}"

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            return "⚠ Empty response from model"

        # 🔥 EXTRA CLEANING (very important)
        answer = answer.replace("Clinical Summary:", "")
        answer = answer.replace("Recommended Actions:", "")
        answer = answer.replace("Evidence:", "")
        answer = answer.replace("Red Flags:", "")

        return answer.strip()

    except Exception as e:
        print("❌ Generation error:", e)
        return "⚠ Error generating medical response. Please try again."
