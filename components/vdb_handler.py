# vdb_handler.py  ── Python ≥3.9, pinecone ≥6, openai ≥1
from __future__ import annotations

import os, uuid
from typing import List

import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# BASIC STUFF & CONFIGS
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY first")

openai.api_key = OPENAI_API_KEY


INDEX_NAME = "memoria-embeddings"
DIMENSION  = 1536
NAMESPACE  = "ns1"

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# Function that turns text into vector
def embed(text: str) -> List[float]:
    return openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    ).data[0].embedding

# Embed + upload to vdb
def upsert_text(text: str, namespace: str = NAMESPACE) -> str:
    vector = embed(text)
    vector_id = str(uuid.uuid4())

    index.upsert(
        vectors=[{
            "id":     vector_id,
            "values": vector,
            "metadata": {"text": text},   # add anything else you like
        }],
        namespace=namespace,
    )
    return vector_id

if __name__ == "__main__":
    vid = upsert_text("hello, memory system!")
    print(f"✅  Upserted vector {vid}")



