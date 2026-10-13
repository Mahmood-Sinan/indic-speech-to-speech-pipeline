"""Contract checks for Groq requests and conversation history (no network)."""

import json
import os
import unittest
from unittest.mock import patch

import httpx2 as httpx
from openai import AuthenticationError, OpenAI

from llm.llm_client import DEFAULT_MODEL, GroqChat


class GroqChatTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.chat = GroqChat("Speak English concisely.")
        self.addCleanup(lambda: self.chat.client.close())

    def use_transport(self, handler):
        self.chat.client.close()
        self.chat.client = OpenAI(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_groq_request_and_followup_history(self):
        requests = []

        def handler(request):
            self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Hello."}}],
            })

        self.use_transport(handler)
        self.assertEqual(self.chat.send_message("Hi"), "Hello.")
        self.chat.send_message("Continue")
        self.assertEqual(requests[0]["model"], DEFAULT_MODEL)
        self.assertEqual(requests[1]["messages"], [
            {"role": "system", "content": "Speak English concisely."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Continue"},
        ])

    def test_failed_request_does_not_commit_turn(self):
        self.use_transport(lambda request: httpx.Response(401, json={
            "error": {"message": "Invalid key", "type": "invalid_request_error"},
        }))
        history = self.chat.messages.copy()
        with self.assertRaises(AuthenticationError) as error:
            self.chat.send_message("Hi")
        self.assertEqual(error.exception.status_code, 401)
        self.assertEqual(self.chat.messages, history)

    def test_empty_response_does_not_commit_turn(self):
        for choices in ([], [{"message": {"role": "assistant", "content": " "}}]):
            with self.subTest(choices=choices):
                self.use_transport(lambda request: httpx.Response(200, json={"choices": choices}))
                history = self.chat.messages.copy()
                with self.assertRaisesRegex(RuntimeError, "empty response"):
                    self.chat.send_message("Hi")
                self.assertEqual(self.chat.messages, history)

    def test_missing_key_fails_with_setup_hint(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "Set GROQ_API_KEY"):
                GroqChat("Speak English.")


if __name__ == "__main__":
    unittest.main()
