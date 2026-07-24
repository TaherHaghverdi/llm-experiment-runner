from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_runner.config import (
    ConfigError,
    build_jobs,
    estimate_cost,
    load_dotenv,
    load_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_example_builds_expected_job_plan(self) -> None:
        inputs = load_inputs(ROOT / "config.example.json")
        jobs = build_jobs(inputs)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({job["job_id"] for job in jobs}), 36)
        self.assertEqual({job["condition"] for job in jobs}, {"plain", "web"})

    def test_estimate_counts_calls_by_model(self) -> None:
        inputs = load_inputs(ROOT / "config.example.json")
        estimate = estimate_cost(inputs, build_jobs(inputs))
        self.assertEqual(
            set(estimate["by_model"]),
            {
                "openai-example",
                "anthropic-example",
                "google-example",
            },
        )
        self.assertTrue(
            all(row["calls"] == 12 for row in estimate["by_model"].values())
        )

    def test_estimate_includes_web_search_pricing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "config.example.json").read_text())
            for model in config["models"]:
                model["enabled"] = model["id"] == "openai-example"
            config["models"][0]["pricing"] = {
                "input_usd_per_million_tokens": 1,
                "output_usd_per_million_tokens": 2,
                "web_search_usd_per_1000_requests": 10,
                "as_of": "2026-07-23",
                "source_url": "https://example.com/pricing",
            }
            (root / "config.json").write_text(json.dumps(config))
            (root / "prompts.example.json").write_text(
                (ROOT / "prompts.example.json").read_text()
            )
            inputs = load_inputs(root / "config.json")
            estimate = estimate_cost(inputs, build_jobs(inputs))
            self.assertTrue(estimate["complete"])
            self.assertEqual(estimate["by_model"]["openai-example"]["web_calls"], 6)
            self.assertAlmostEqual(estimate["known_total_usd"], 0.0786)

    def test_duplicate_prompt_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "config.example.json").read_text())
            prompts = json.loads((ROOT / "prompts.example.json").read_text())
            prompts["prompts"].append(dict(prompts["prompts"][0]))
            (root / "config.json").write_text(json.dumps(config))
            (root / "prompts.example.json").write_text(json.dumps(prompts))
            with self.assertRaisesRegex(ConfigError, "Duplicate prompt id"):
                load_inputs(root / "config.json")

    def test_dotenv_does_not_override_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("EXISTING_KEY=from-file\nNEW_TEST_KEY='new value'\n")
            original = os.environ.get("EXISTING_KEY")
            original_new = os.environ.get("NEW_TEST_KEY")
            try:
                os.environ["EXISTING_KEY"] = "from-environment"
                os.environ.pop("NEW_TEST_KEY", None)
                loaded = load_dotenv(path)
                self.assertEqual(os.environ["EXISTING_KEY"], "from-environment")
                self.assertEqual(os.environ["NEW_TEST_KEY"], "new value")
                self.assertEqual(loaded, ["NEW_TEST_KEY"])
            finally:
                if original is None:
                    os.environ.pop("EXISTING_KEY", None)
                else:
                    os.environ["EXISTING_KEY"] = original
                if original_new is None:
                    os.environ.pop("NEW_TEST_KEY", None)
                else:
                    os.environ["NEW_TEST_KEY"] = original_new


if __name__ == "__main__":
    unittest.main()
