import os
import shutil
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# --------------------------------
# Configuration
# --------------------------------

load_dotenv()

DATA_DIR = Path("data")
PERSIST_DIR = Path("storage")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set"
    )


# --------------------------------
# Gemini
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
# Remove old index
# --------------------------------

if PERSIST_DIR.exists():
    print("3. Removing old index...")

    shutil.rmtree(PERSIST_DIR)


# --------------------------------
# Load PDFs using PyMuPDF
# --------------------------------

print("4. Loading PDFs...")

documents = []

pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise RuntimeError(
        f"No PDF files found in {DATA_DIR}"
    )


for pdf_path in pdf_files:

    print(f"\nProcessing: {pdf_path.name}")

    pdf = pymupdf.open(pdf_path)

    print(f"Pages: {len(pdf)}")

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text").strip()

        if not text:
            continue

        document = Document(
            text=text,
            metadata={
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "page_number": page_number,
            },
        )

        documents.append(document)

    pdf.close()


print(
    f"\n5. Extracted {len(documents)} pages "
    f"from {len(pdf_files)} PDF(s)"
)


# --------------------------------
# Create vector index
# --------------------------------

print("\n6. Creating embeddings and vector index...")

index = VectorStoreIndex.from_documents(
    documents
)


# --------------------------------
# Persist
# --------------------------------

print("7. Persisting index...")

index.storage_context.persist(
    persist_dir=str(PERSIST_DIR)
)

print("\n8. Index successfully created!")
print(f"Stored in: {PERSIST_DIR}")
