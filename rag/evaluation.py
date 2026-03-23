from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_faithfulness(answer, context):

    if not answer or not context:
        return 0

    answer = answer.lower()
    context = context.lower()

    score = 0
    words = answer.split()

    for w in words:
        if w in context:
            score += 1

    return min(100, (score / len(words)) * 100)
