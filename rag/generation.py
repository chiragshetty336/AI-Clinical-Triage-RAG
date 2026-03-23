import requests


def generate_answer(context, question):

    # 🔥 LIMIT CONTEXT SIZE
    context = context[:2000]

    prompt = f"""
You are a medical triage assistant.

RULES:
- Use the provided context as PRIMARY source.
- You may use basic medical reasoning ONLY if it is directly related to the situation.
- DO NOT introduce unrelated conditions.
- Do NOT contradict the given triage level.
- DO NOT cite external guidelines unless present in context.
- If context is missing details, give safe general emergency advice.

-----------------------
CONTEXT:
{context}
-----------------------

PATIENT:
{question}

-----------------------
OUTPUT FORMAT:

Clinical Summary:
- Describe the situation.

Recommended Actions:
- Use context-supported actions.
- Add basic emergency care steps if obvious (e.g., trauma → check airway, bleeding).

Evidence:
- Quote from context if available.

Red Flags:
- Mention serious warning signs relevant to situation.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:7b", "prompt": prompt, "stream": False},
            timeout=300,
        )

        if response.status_code != 200:
            return f"⚠ API Error: {response.text}"

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            return "⚠ Empty response from model"

        return answer

    except Exception as e:
        print("❌ Generation error:", e)
        return "⚠ Error generating medical response. Please try again."
