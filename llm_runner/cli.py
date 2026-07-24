from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_KEY_ENVS,
    ConfigError,
    build_jobs,
    enabled_models,
    estimate_cost,
    load_dotenv,
    load_inputs,
    required_key_envs,
)
from .engine import inspect_status, prepare_run, run_experiment, verify_run
from .providers import ProviderError, list_models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-positioning",
        description="Run and verify multi-provider LLM positioning benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="Validate inputs and preview calls and cost"
    )
    plan.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run", help="Start or resume an experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--env", type=Path)
    run.add_argument("--resume", type=Path)

    verify = subparsers.add_parser("verify", help="Check a run against its manifest")
    verify.add_argument("run_dir", type=Path)
    verify.add_argument("--json", action="store_true", dest="as_json")

    status = subparsers.add_parser(
        "status", help="Show progress and stale-run detection"
    )
    status.add_argument("run_dir", type=Path)
    status.add_argument("--json", action="store_true", dest="as_json")

    models = subparsers.add_parser(
        "models", help="List models exposed by a provider API"
    )
    models.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        required=True,
    )
    models.add_argument("--env", type=Path)
    models.add_argument("--api-key-env")
    models.add_argument("--base-url")
    models.add_argument("--timeout", type=float, default=30)
    models.add_argument("--contains", default="")
    models.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def _print_plan(inputs: Any) -> dict[str, Any]:
    jobs = build_jobs(inputs)
    estimate = estimate_cost(inputs, jobs)
    by_provider: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    for job in jobs:
        by_provider[job["provider"]] = by_provider.get(job["provider"], 0) + 1
        by_condition[job["condition"]] = by_condition.get(job["condition"], 0) + 1
    print(f"Experiment: {inputs.config['experiment']['name']}")
    print(f"Prompts: {len(inputs.prompts)}")
    print(f"Enabled models: {len(enabled_models(inputs.config))}")
    print(f"Repetitions: {inputs.config['experiment']['repetitions']}")
    print(f"Planned calls: {len(jobs)}")
    print(
        "Calls by provider: "
        + ", ".join(f"{key}={value}" for key, value in by_provider.items())
    )
    print(
        "Calls by condition: "
        + ", ".join(f"{key}={value}" for key, value in by_condition.items())
    )
    for model_id, row in estimate["by_model"].items():
        print(
            f"  {model_id}: {row['calls']} calls, "
            f"estimated {_money(row['estimated_usd'])}"
        )
    if estimate["complete"]:
        print(f"Estimated total: {_money(estimate['known_total_usd'])}")
    else:
        missing = ", ".join(estimate["models_without_pricing"])
        print(f"Estimated total: incomplete (missing pricing for: {missing})")
    maximum = inputs.config["budget"].get("max_estimated_usd")
    within_budget: bool | None = None
    if maximum is not None:
        within_budget = estimate["complete"] and estimate["known_total_usd"] <= float(
            maximum
        )
        print(f"Configured estimate ceiling: {_money(float(maximum))}")
        print(f"Within ceiling: {'yes' if within_budget else 'no'}")
    placeholders = [
        model["id"]
        for model in enabled_models(inputs.config)
        if model["model"].startswith("replace-with-")
    ]
    if placeholders:
        print("Model IDs still need replacement: " + ", ".join(placeholders))
    return {
        "jobs": jobs,
        "estimate": estimate,
        "within_budget": within_budget,
        "placeholder_models": placeholders,
    }


def _load_env_if_requested(path: Path | None) -> None:
    if path is not None:
        loaded = load_dotenv(path.resolve())
        print(
            f"Loaded {len(loaded)} value(s) from {path}; existing environment values won"
        )


def _print_report(value: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return
    for key, item in value.items():
        if isinstance(item, list):
            if item:
                print(f"{key}:")
                for child in item:
                    print(f"  - {child}")
            else:
                print(f"{key}: []")
        else:
            print(f"{key}: {item}")


def _run(args: argparse.Namespace) -> int:
    inputs = load_inputs(args.config)
    plan = _print_plan(inputs)
    if plan["placeholder_models"]:
        raise ConfigError(
            "Replace every placeholder model ID before starting a paid run"
        )
    maximum = inputs.config["budget"].get("max_estimated_usd")
    if maximum is not None:
        if not plan["estimate"]["complete"]:
            raise ConfigError(
                "A budget ceiling is configured, but one or more models have no pricing"
            )
        if plan["estimate"]["known_total_usd"] > float(maximum):
            raise ConfigError("Estimated cost exceeds config.budget.max_estimated_usd")
    _load_env_if_requested(args.env)
    missing = [
        name for name in required_key_envs(inputs.config) if not os.environ.get(name)
    ]
    if missing:
        raise ConfigError(
            "Missing API key environment variable(s): " + ", ".join(missing)
        )
    run_dir, manifest = prepare_run(inputs, args.resume)
    print(f"Run directory: {run_dir}")
    exit_code, status = run_experiment(inputs, run_dir, manifest)
    print(f"State: {status['state']}")
    print(
        f"Progress: {status['successful']}/{status['planned']} succeeded, "
        f"{status['failed']} failed, {status['unfinished']} unfinished"
    )
    if status.get("message"):
        print(status["message"])
    return exit_code


def _models(args: argparse.Namespace) -> int:
    _load_env_if_requested(args.env)
    key_env = args.api_key_env or DEFAULT_KEY_ENVS[args.provider]
    if not os.environ.get(key_env):
        raise ConfigError(f"Missing API key environment variable: {key_env}")
    values = list_models(
        args.provider,
        api_key_env=key_env,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    needle = args.contains.casefold()
    if needle:
        values = [
            value for value in values if needle in str(value.get("id", "")).casefold()
        ]
    values.sort(key=lambda value: str(value.get("id", "")))
    if args.as_json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
    else:
        for value in values:
            details = ", ".join(
                f"{key}={item}"
                for key, item in value.items()
                if key != "id" and item not in (None, "", [])
            )
            print(f"{value['id']}{' · ' + details if details else ''}")
        print(f"{len(values)} model(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            inputs = load_inputs(args.config)
            _print_plan(inputs)
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "verify":
            report = verify_run(args.run_dir)
            _print_report(report, args.as_json)
            return 0 if report["valid"] else 2
        if args.command == "status":
            report = inspect_status(args.run_dir)
            _print_report(report, args.as_json)
            return 0
        if args.command == "models":
            return _models(args)
        raise ConfigError(f"Unknown command: {args.command}")
    except (ConfigError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
