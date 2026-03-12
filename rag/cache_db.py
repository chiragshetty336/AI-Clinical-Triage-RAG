import numpy as np
import json
from rag.db import get_connection

SIMILARITY_THRESHOLD = 0.85


def search_cache(query_embedding):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT query, embedding, answer, triage_level, sources
        FROM faq_cache
        """
    )

    rows = cur.fetchall()

    best_match = None
    best_score = 0

    for row in rows:

        stored_embedding = np.array(row[1])

        similarity = np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
        )

        if similarity > best_score:
            best_score = similarity
            best_match = row

    cur.close()
    conn.close()

    if best_match and best_score > SIMILARITY_THRESHOLD:

        return {
            "query": best_match[0],
            "answer": best_match[2],
            "triage_level": best_match[3],
            "sources": best_match[4],
        }

    return None


def store_cache(query, embedding, answer, triage_level, sources):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO faq_cache (query, embedding, answer, triage_level, sources)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (
            query,
            embedding.tolist(),
            answer,
            triage_level,
            json.dumps(sources),
        ),
    )

    conn.commit()
