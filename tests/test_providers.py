from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from llm_runner.providers import call_anthropic, call_google, call_openai

REQUEST = {
    "system_prompt": "",
    "max_output_tokens": 500,
    "temperature": None,
    "timeout_seconds": 10,
    "anthropic_web_search_max_uses": 2,
}


class ProviderParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            name: os.environ.get(name)
            for name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        }
        os.environ["OPENAI_API_KEY"] = "openai-secret"
        os.environ["ANTHROPIC_API_KEY"] = "anthropic-secret"
        os.environ["GOOGLE_API_KEY"] = "google-secret"

    def tearDown(self) -> None:
        for name, value in self.original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @patch("llm_runner.providers.request_json")
    def test_openai_text_queries_sources_and_usage(self, request_json) -> None:
        request_json.return_value = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"query": "best demo recording tools"},
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Consider Northstar.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/review",
                                    "title": "Review",
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
        response = call_openai(
            {
                "provider": "openai",
                "model": "test",
                "api_key_env": "OPENAI_API_KEY",
            },
            "web",
            "Question?",
            REQUEST,
        )
        self.assertEqual(response.text, "Consider Northstar.")
        self.assertEqual(response.search_queries, ["best demo recording tools"])
        self.assertEqual(response.sources[0]["url"], "https://example.com/review")
        self.assertEqual(response.usage["total_tokens"], 30)
        self.assertEqual(response.search_count, 1)
        payload = request_json.call_args.kwargs["payload"]
        self.assertEqual(payload["tools"], [{"type": "web_search"}])

    @patch("llm_runner.providers._anthropic_request")
    def test_anthropic_continues_pause_turn(self, anthropic_request) -> None:
        anthropic_request.side_effect = [
            {
                "id": "msg_1",
                "stop_reason": "pause_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "input": {"query": "video hosting options"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com/list",
                                "title": "A list",
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "server_tool_use": {"web_search_requests": 1},
                },
            },
            {
                "id": "msg_2",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Here are three options."}],
                "usage": {"input_tokens": 8, "output_tokens": 9},
            },
        ]
        response = call_anthropic(
            {
                "provider": "anthropic",
                "model": "test",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "web",
            "Question?",
            REQUEST,
        )
        self.assertEqual(anthropic_request.call_count, 2)
        self.assertEqual(response.text, "Here are three options.")
        self.assertEqual(response.usage["input_tokens"], 20)
        self.assertTrue(response.searched)
        self.assertEqual(response.search_count, 1)
        self.assertEqual(response.search_queries, ["video hosting options"])

    @patch("llm_runner.providers.request_json")
    def test_google_grounding_metadata(self, request_json) -> None:
        request_json.return_value = {
            "responseId": "google_1",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Try Northstar."}]},
                    "groundingMetadata": {
                        "webSearchQueries": ["best customer video host"],
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/guide",
                                    "title": "Guide",
                                }
                            }
                        ],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 18,
            },
        }
        response = call_google(
            {
                "provider": "google",
                "model": "test model",
                "api_key_env": "GOOGLE_API_KEY",
            },
            "web",
            "Question?",
            REQUEST,
        )
        self.assertEqual(response.text, "Try Northstar.")
        self.assertTrue(response.searched)
        self.assertEqual(response.search_count, 1)
        self.assertEqual(response.sources[0]["title"], "Guide")
        self.assertIn("test%20model", request_json.call_args.args[1])
        self.assertNotIn("google-secret", request_json.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
