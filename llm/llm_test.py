"""Check Groq credentials without loading the speech models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.llm_client import GroqChat


if __name__ == "__main__":
    chat = GroqChat("You are a helpful assistant. Keep your answers concise.")
    print(chat.send_message("Explain how AI works in two sentences."))
