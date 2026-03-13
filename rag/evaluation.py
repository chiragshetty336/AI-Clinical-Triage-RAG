from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_faithfulness(answer, context):

    if not answer or not context:
        return 0

    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(answer_embedding, context_embedding)

    score = float(similarity[0][0])

    return round(score * 100, 2)
