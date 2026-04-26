"""
V2 Generator
- Takes retrieved context + user query
- Constructs prompt with multi-department context
- Calls LLM for answer (streaming supported)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, SYSTEM_PROMPT


class SimpleGenerator:
    """LLM generator with context stuffing and streaming."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, query: str, context: str) -> str:
        system_message = SYSTEM_PROMPT.format(context=context)
        response = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content

    def generate_stream(self, query: str, context: str):
        system_message = SYSTEM_PROMPT.format(context=context)
        stream = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
