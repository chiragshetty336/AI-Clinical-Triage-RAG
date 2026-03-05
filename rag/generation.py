import requests


def generate_answer(context, question):
    prompt = f"""
You are a clinical assistant.

Use ONLY the information in context.

Context:
{context}

Question:
{question}

Provide:
1. Immediate Actions
2. Possible Causes
3. Recommended Investigations
4. Monitoring Plan
5. Red Flag Signs
"""

    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "phi3:mini", "prompt": prompt, "stream": False},
    )

    return response.json()["response"]
