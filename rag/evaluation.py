def calculate_faithfulness(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if len(answer_words) == 0:
        return 0

    overlap = answer_words.intersection(context_words)
    faithfulness = len(overlap) / len(answer_words)

    return round(faithfulness * 100, 2)
