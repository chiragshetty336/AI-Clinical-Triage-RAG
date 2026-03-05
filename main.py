import os
from rag.ingestion import load_pdfs_with_cache
from rag.indexing import build_index_from_embeddings, load_index
from rag.agent import medical_agent
from rag.config import INDEX_PATH


def main():

    if not os.path.exists(INDEX_PATH):
        chunks, embeddings, metadata = load_pdfs_with_cache()
        index = build_index_from_embeddings(embeddings)
    else:
        chunks, embeddings, metadata = load_pdfs_with_cache()
        index = load_index()

    while True:
        query = input("\nAsk medical question (or type exit): ")

        if query.lower() == "exit":
            break

        result = medical_agent(query, index, chunks, metadata)

        print("\n================ CLINICAL DECISION =================\n")
        print(result["answer"])
        print("\n===================================================\n")

        print(f"\nConfidence Score: {round(result['confidence'],3)}")
        print(f"Faithfulness Score: {round(result['faithfulness'],2)}%")

        if result["safety_flag"]:
            print("⚠ WARNING: Low grounding detected.")

        if result["emergency"]:
            print("🚨 Emergency scenario detected.")

        print("\nSources Used:")
        for i, src in enumerate(result["sources"][:3]):
            print(f"[{i+1}] {src['source']} – Page {src['page']}")


if __name__ == "__main__":
    main()
