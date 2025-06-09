import os
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, List, Any

import openai
from pinecone import Pinecone


EMBED_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")


def embed(text: str) -> List[float]:
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return response.data[0].embedding


PC_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "__default__")

if not (PC_API_KEY and INDEX_HOST):
    raise RuntimeError(
        "Set PINECONE_API_KEY and PINECONE_INDEX_HOST in the environment"
    )

pc = Pinecone(api_key=PC_API_KEY)                   #
index = pc.Index(host=INDEX_HOST)

def upsert_message(text: str, role: str = "user") -> str:
    vector = embed(text)
    vector_id = str(uuid4())

    metadata: Dict[str, Any] = {
        "role": role,
        "message": text,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": EMBED_MODEL,

    }

    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": vector,
            "metadata": metadata,
        }],
        namespace=NAMESPACE,
    )
    return vector_id

if __name__ == "__main__":
    print(upsert_message("hello"))


