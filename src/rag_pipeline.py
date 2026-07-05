# src/rag_pipeline.py

from src.embeddings import load_vector_store


def retrieve_context(question, k=3):

    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    return context