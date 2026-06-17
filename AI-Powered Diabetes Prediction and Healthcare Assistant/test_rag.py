from src.rag_pipeline import retrieve_context

question = "What foods should diabetic patients avoid?"

print(retrieve_context(question))