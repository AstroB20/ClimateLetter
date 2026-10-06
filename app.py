import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# --------------------------------
# Configuration
# --------------------------------

load_dotenv()

PERSIST_DIR = "storage"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set"
    )


# --------------------------------
# Check that index exists
# --------------------------------

if not Path(PERSIST_DIR).exists():
    raise RuntimeError(
        "No vector index found.\n"
        "Run index.py first to create the index."
    )


# --------------------------------
# Gemini LLM
# --------------------------------

print("1. Loading Gemini...")

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)


# --------------------------------
# Embedding model
# --------------------------------

print("2. Loading embedding model...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------
# Load existing index
# --------------------------------

print("3. Loading vector index...")

storage_context = StorageContext.from_defaults(
    persist_dir=PERSIST_DIR
)

index = load_index_from_storage(
    storage_context
)

print("4. Vector index loaded!")


# --------------------------------
# Create query engine
# --------------------------------

print("5. Creating query engine...")

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("6. Climate RAG is ready!")
print("Type 'exit' to quit.\n")


# --------------------------------
# Interactive RAG
# --------------------------------

while True:

    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nSearching documents...\n")

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response)

  