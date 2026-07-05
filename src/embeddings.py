# src/embeddings.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def save_vector_store(chunks):
    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    vector_store.save_local("vector_store")


def load_vector_store():
    embeddings = get_embeddings()

    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )