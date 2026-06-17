from pathlib import Path

from src.embeddings import save_vector_store

all_text = ""

for file in Path("medical_docs").glob("*.txt"):
    all_text += file.read_text(encoding="utf-8")
    all_text += "\n"

chunks = [
    all_text[i:i+500]
    for i in range(0, len(all_text), 500)
]

save_vector_store(chunks)

print("Vector store created successfully!")