from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_runner.config import load_inputs
from llm_runner.engine import (
    _parse_time,
    prepare_run,
    read_attempts,
    run_experiment,
    verify_run,
)
from llm_runner.providers import ProviderError, ProviderResponse


def response(text: str = "Synthetic answer") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        finish_reason="stop",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        searched=False,
    )


class EngineTests(unittest.TestCase):
    def test_utc_heartbeat_parses_on_supported_python_versions(self) -> None:
        parsed = _parse_time("2026-07-23T12:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(parsed.tzinfo)

    def make_inputs(self, root: Path):
        prompts = {
            "schema_version": "1.0",
            "prompts": [{"id": "one", "group": "need", "text": "What should I use?"}],
        }
        config = {
            "schema_version": "1.0",
            "experiment": {
                "name": "integration-test",
                "output_dir": "runs",
                "repetitions": 1,
            },
            "prompts_file": "prompts.json",
            "conditions": ["plain"],
            "models": [
                {
                    "id": "model-a",
                    "provider": "openai",
                    "model": "synthetic-model",
                    "api_key_env": "SYNTHETIC_API_KEY",
                    "conditions": ["plain"],
                    "pricing": {
                        "input_usd_per_million_tokens": 1,
                        "output_usd_per_million_tokens": 2,
                    },
                }
            ],
            "request": {
                "system_prompt": "",
                "max_output_tokens": 100,
                "temperature": None,
                "timeout_seconds": 5,
                "max_attempts": 1,
                "retry_backoff_seconds": 0,
                "concurrency": 1,
                "abort_after_failures": 1,
                "anthropic_web_search_max_uses": 1,
            },
            "estimate": {
                "input_tokens_per_call": 10,
                "output_tokens_per_call": 20,
                "web_searches_per_web_call": 1,
            },
            "budget": {"max_estimated_usd": 1},
        }
        (root / "prompts.json").write_text(json.dumps(prompts))
        (root / "config.json").write_text(json.dumps(config))
        return load_inputs(root / "config.json")

    def test_complete_run_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(Path(directory))
            run_dir, manifest = prepare_run(inputs, None)
            exit_code, status = run_experiment(
                inputs,
                run_dir,
                manifest,
                provider_call=lambda *_: response(),
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(status["state"], "complete")
            report = verify_run(run_dir)
            self.assertTrue(report["valid"])
            self.assertEqual(report["successful_jobs"], 1)

    def test_failed_job_can_resume_without_replacing_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(Path(directory))
            run_dir, manifest = prepare_run(inputs, None)

            def fail(*_):
                raise ProviderError("synthetic failure", retryable=False)

            first_code, first_status = run_experiment(
                inputs,
                run_dir,
                manifest,
                provider_call=fail,
            )
            self.assertEqual(first_code, 2)
            self.assertEqual(first_status["failed"], 1)
            self.assertFalse(verify_run(run_dir)["valid"])

            resumed_dir, resumed_manifest = prepare_run(inputs, run_dir)
            second_code, second_status = run_experiment(
                inputs,
                resumed_dir,
                resumed_manifest,
                provider_call=lambda *_: response("Recovered"),
            )
            self.assertEqual(second_code, 0)
            self.assertEqual(second_status["successful"], 1)
            attempts, errors = read_attempts(run_dir)
            self.assertEqual(errors, [])
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[-1]["attempt_sequence"], 2)
            report = verify_run(run_dir)
            self.assertTrue(report["valid"])
            self.assertEqual(report["reattempt_records"], 1)


if __name__ == "__main__":
    unittest.main()
