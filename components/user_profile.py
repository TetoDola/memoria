# memoria/components/user_profile_generator.py
import os
import json
from typing import Dict, Any

import openai
from dotenv import load_dotenv


class UserProfileGenerator:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        openai_api_key: str = None,
    ) -> None:
        self.model = model
        self.temperature = temperature

        #  API Key validations
        if not (openai_api_key):
            raise ValueError("OpenAI API keys must be provided.")

        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    def _summarize_and_update_from_conversation(self, conversation_history: str, user_profile: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the user's statements in the following conversation and extract key details for a user profile.
        Focus on:
        - Preferences: What the user likes or dislikes.
        - Life Events: Significant current or past events.
        - Goals: What the user aims to achieve.
        - Key Attributes: Name, location, profession, etc.
        - Anything else that is important

        Conversation:
        {conversation_history}
        
        Old User Profile:
        {user_profile}

        Return a JSON object with keys 'preferences', 'life_events', 'goals', 'key_attributes'.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that builds user profiles from conversations."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during profile summarization: {e}")
            return {}


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    conversation = "Hi, I'm Alex. I live in Berlin and work as a game developer. I love anime and recently started rock climbing. I’m planning a trip to Japan next year."
    old_profile = "{}"

    generator = UserProfileGenerator(openai_api_key=api_key)
    updated_profile = generator._summarize_and_update_from_conversation(conversation, old_profile)

    print(json.dumps(updated_profile, indent=2))