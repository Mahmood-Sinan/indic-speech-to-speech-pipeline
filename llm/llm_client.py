"""Groq chat shared by the voice pipelines and the standalone LLM check."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqChat:
    """Keep conversation history, committing each turn only after success."""

    def __init__(self, system_prompt, model=DEFAULT_MODEL):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set GROQ_API_KEY in the environment or the project's .env file.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
            max_retries=1,
        )
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    def send_message(self, text):
        messages = self.messages + [{"role": "user", "content": text}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise RuntimeError("Groq returned an empty response.")
        self.messages = messages + [{"role": "assistant", "content": content}]
        return content
