import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer, util

# ==============================
# SAFE IMPORT (CRITICAL FIX)
# ==============================

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==============================
# LOAD MODELS
# ==============================

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ==============================
# GEMINI ANSWER (SAFE)
# ==============================


def get_gemini_answer(query):

    if not GEMINI_AVAILABLE:
        print("⚠ Gemini not installed")
        return "Gemini unavailable"

    if not GEMINI_API_KEY:
        print("⚠ GEMINI_API_KEY missing")
        return "Gemini API key not configured"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        response = model.generate_content(
            f"You are a medical triage expert. Answer briefly.\n\nQuestion: {query}"
        )

        return response.text if response.text else "No response"

    except Exception as e:
        print("❌ Gemini error:", str(e))
        return "Gemini error"


# ==============================
# SIMILARITY FUNCTION
# ==============================


def compute_similarity(answer1, answer2):

    if not answer1 or not answer2:
        return 0.0

    try:
        embeddings = similarity_model.encode([answer1, answer2], convert_to_tensor=True)

        similarity = util.cos_sim(embeddings[0], embeddings[1])

        return float(similarity)

    except Exception as e:
        print("⚠ Similarity error:", str(e))
        return 0.0


# ==============================
# MAIN EVALUATION FUNCTION
# ==============================


def evaluate_answers(query, rag_answer):

    # 🔹 If Gemini not available → skip safely
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return {
            "gpt_answer": "Evaluation skipped (Gemini unavailable)",
            "similarity_score": 0.0,
        }

    # 🔹 Get Gemini answer
    gemini_answer = get_gemini_answer(query)

    # 🔹 Compute similarity
    similarity = compute_similarity(rag_answer, gemini_answer)

    return {"gpt_answer": gemini_answer, "similarity_score": similarity}
