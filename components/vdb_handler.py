import os
import uuid
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


class VDBHandler:
    def __init__(
            self,
            openai_api_key: str,
            pinecone_api_key: str,
            # The name of the Pinecone index.
            index_name: str = "memoria-embeddings",
            # Dimension of vectors
            dimension: int = 1536,
            # The namespace to use within the Pinecone index.
            namespace: str = "ns1"
    ):

        #  API Key validations
        if not (openai_api_key and pinecone_api_key):
            raise ValueError("OpenAI and Pinecone API keys must be provided.")

        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        openai.api_key = self.openai_api_key

        self.index_name = index_name
        self.dimension = dimension
        self.namespace = namespace

        #  Initialize Pinecone
        pc = Pinecone(api_key=self.pinecone_api_key)

        #  Create index if it doesn't exist
        if self.index_name not in pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("Index created successfully.")

        self.index = pc.Index(self.index_name)
        print("Pinecone index is ready.")

    # calls openai and turns text to a vector
    def embed(self, text: str) -> List[float]:
        try:
            return openai.embeddings.create(
                model="text-embedding-3-small",
                input=[text],
            ).data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            raise


    # calls embed function then uploads returned vector to index.
    # Function returns The unique ID of the upserted vector.
    def upsert(self, text: str, metadata: Dict[str, Any] = None) -> str:
        if metadata is None:
            metadata = {}

        # Ensure the original text is always part of the metadata
        metadata["text"] = text

        vector = self.embed(text)
        vector_id = str(uuid.uuid4())

        self.index.upsert(
            vectors=[{
                "id": vector_id,
                "values": vector,
                "metadata": metadata,
            }],
            namespace=self.namespace,
        )
        return vector_id

    # Saves a conversational turn (both user and agent utterances) to the database.
    def save_turn(self, user_text: str, agent_text: str, turn_id: str | None = None) -> Dict[str, str]:
        if turn_id is None:
            turn_id = f"turn_{uuid.uuid4()}"

        print(f"Saving conversational turn {turn_id}...")

        user_vector_id = self.upsert(user_text, metadata={"role": "user", "turn_id": turn_id})
        agent_vector_id = self.upsert(agent_text, metadata={"role": "agent", "turn_id": turn_id})

        return {
            "turn_id": turn_id,
            "user_vector_id": user_vector_id,
            "agent_vector_id": agent_vector_id
        }

#  Example Usage
if __name__ == "__main__":
    # The user of the class is now responsible for loading environment variables
    load_dotenv()

    # Retrieve keys from the environment
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not (openai_key and pinecone_key):
        raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY in your .env file")

    # Pass the keys during initialization
    handler = VDBHandler(
        openai_api_key=openai_key,
        pinecone_api_key=pinecone_key,
        namespace="dialogue-test"
    )

    vec_id = handler.upsert("This is a standalone memory.")
    print(f"✅  Uploaded single vector: {vec_id}")
    turn_data = handler.save_turn(
        user_text="What is the capital of France?",
        agent_text="The capital of France is Paris."
    )
    print(f"✅  Saved turn '{turn_data['turn_id']}' with vectors: "
          f"user({turn_data['user_vector_id']}), agent({turn_data['agent_vector_id']})")
