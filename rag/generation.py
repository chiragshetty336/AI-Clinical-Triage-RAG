import requests


def generate_answer(context, question):
    prompt = f"""
You are an experienced clinical doctor.

Explain the patient's condition clearly in simple language.

Speak as if you are explaining the case to a junior doctor or nurse.

Avoid unnecessary technical jargon.

Context:
{context}

Patient description:
{question}

Provide:

1. What is likely happening with the patient
2. Immediate actions needed
3. Possible causes
4. Recommended tests
5. Warning signs to monitor
"""

    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "phi3:mini", "prompt": prompt, "stream": False},
    )

    return response.json()["response"]
