import requests


def generate_answer(context, question):

    prompt = f"""
    You are a medical emergency triage assistant.

    STRICT RULES:
    - Use ONLY the medical information from the provided context.
    - DO NOT invent patient details.
    - DO NOT assume age, gender, or medical history.
    - DO NOT create fictional scenarios.
    - If the context does not contain enough information, say:
    "The guidelines do not provide enough information."

    CONTEXT:
    {context}

    PATIENT CASE:
    {question}

    Respond in this format:

    Clinical Summary:
    <Explain the situation based only on the query>

    Recommended Actions:
    <Immediate medical steps if applicable>

    Evidence from Guidelines:
    <Quote relevant text from context>

    Red Flag Signs:
    <List dangerous symptoms to monitor>
    """

    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "phi3:mini", "prompt": prompt, "stream": False},
        timeout=60,
    )

    return response.json()["response"]
