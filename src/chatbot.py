# src/chatbot.py

import ollama
from src.rag_pipeline import retrieve_context


def ask_medical_assistant(question):

    context = retrieve_context(question)

    prompt = f"""
You are a healthcare assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer in simple language.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]