import requests


def reflect_and_improve(answer, context, query):

    prompt = f"""
You are a medical safety assistant.

Your job is to REVIEW and IMPROVE the answer.

-----------------------
ORIGINAL ANSWER:
{answer}
-----------------------

CONTEXT:
{context}
-----------------------

PATIENT QUERY:
{query}

-----------------------

TASK:

1. Check if the answer is:
   - clear
   - correct
   - easy to understand

2. Fix problems if any:
   - unclear explanation
   - missing important warning
   - wrong tone

3. Keep it SIMPLE for normal people

-----------------------

FINAL OUTPUT FORMAT:

Triage Level: RED or YELLOW or GREEN

Reason:
(simple explanation)

What to do:
(clear steps)

-----------------------

Return ONLY the improved answer.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            },
            timeout=300,
        )

        if response.status_code != 200:
            return answer  # fallback

        data = response.json()
        improved = data.get("response", "").strip()

        if not improved:
            return answer

        return improved

    except:
        return answer
