from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import CONTRACT_VERSION

SUPPORTED_PROVIDERS = {"openai", "anthropic", "google"}
SUPPORTED_CONDITIONS = {"plain", "web"}
DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Inputs:
    config_path: Path
    prompts_path: Path
    config: dict[str, Any]
    prompts: list[dict[str, str]]
    config_sha256: str
    prompts_sha256: str


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _need(mapping: dict[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        raise ConfigError(f"{where}.{key} must be {kind.__name__}")
    return value


def _positive_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _positive_number(value: Any, where: str, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where} must be a number")
    number = float(value)
    if number < 0 if allow_zero else number <= 0:
        qualifier = "zero or greater" if allow_zero else "greater than zero"
        raise ConfigError(f"{where} must be {qualifier}")
    return number


def load_dotenv(path: Path) -> list[str]:
    """Load a small, conventional .env file without adding a dependency."""
    if not path.exists():
        raise ConfigError(f"Environment file not found: {path}")
    loaded: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigError(f"Invalid .env line {line_number}: expected NAME=value")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ConfigError(
                f"Invalid environment variable name on line {line_number}"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_inputs(config_path: Path) -> Inputs:
    config_path = config_path.resolve()
    config, config_hash = _read_json(config_path)
    if config.get("schema_version") != CONTRACT_VERSION:
        raise ConfigError(
            f"config schema_version must be {CONTRACT_VERSION!r}; "
            f"got {config.get('schema_version')!r}"
        )

    experiment = _need(config, "experiment", dict, "config")
    name = _need(experiment, "name", str, "config.experiment").strip()
    if not name:
        raise ConfigError("config.experiment.name cannot be empty")
    _need(experiment, "output_dir", str, "config.experiment")
    _positive_int(experiment.get("repetitions"), "config.experiment.repetitions")

    prompts_file = _need(config, "prompts_file", str, "config")
    prompts_path = (config_path.parent / prompts_file).resolve()
    prompt_doc, prompts_hash = _read_json(prompts_path)
    if prompt_doc.get("schema_version") != CONTRACT_VERSION:
        raise ConfigError(
            f"prompt schema_version must be {CONTRACT_VERSION!r}; "
            f"got {prompt_doc.get('schema_version')!r}"
        )
    prompts_raw = _need(prompt_doc, "prompts", list, "prompt document")
    if not prompts_raw:
        raise ConfigError("prompt document must contain at least one prompt")
    prompts: list[dict[str, str]] = []
    prompt_ids: set[str] = set()
    for index, prompt in enumerate(prompts_raw):
        where = f"prompts[{index}]"
        if not isinstance(prompt, dict):
            raise ConfigError(f"{where} must be an object")
        prompt_id = _need(prompt, "id", str, where).strip()
        group = _need(prompt, "group", str, where).strip()
        text = _need(prompt, "text", str, where).strip()
        if not prompt_id or not group or not text:
            raise ConfigError(f"{where} id, group, and text cannot be empty")
        if prompt_id in prompt_ids:
            raise ConfigError(f"Duplicate prompt id: {prompt_id}")
        prompt_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "group": group, "text": text})

    conditions = _need(config, "conditions", list, "config")
    if not conditions:
        raise ConfigError("config.conditions cannot be empty")
    if len(set(conditions)) != len(conditions):
        raise ConfigError("config.conditions cannot contain duplicates")
    unknown_conditions = set(conditions) - SUPPORTED_CONDITIONS
    if unknown_conditions:
        raise ConfigError(f"Unsupported conditions: {sorted(unknown_conditions)}")

    models = _need(config, "models", list, "config")
    if not models:
        raise ConfigError("config.models cannot be empty")
    model_ids: set[str] = set()
    enabled_count = 0
    for index, model in enumerate(models):
        where = f"config.models[{index}]"
        if not isinstance(model, dict):
            raise ConfigError(f"{where} must be an object")
        model_id = _need(model, "id", str, where).strip()
        provider = _need(model, "provider", str, where).strip().lower()
        model_name = _need(model, "model", str, where).strip()
        if not model_id or not model_name:
            raise ConfigError(f"{where}.id and .model cannot be empty")
        if model_id in model_ids:
            raise ConfigError(f"Duplicate model id: {model_id}")
        model_ids.add(model_id)
        if provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(f"{where}.provider is unsupported: {provider}")
        model_conditions = _need(model, "conditions", list, where)
        if not model_conditions:
            raise ConfigError(f"{where}.conditions cannot be empty")
        unsupported = set(model_conditions) - set(conditions)
        if unsupported:
            raise ConfigError(
                f"{where}.conditions contains values not enabled globally: {sorted(unsupported)}"
            )
        if model.get("enabled", True):
            enabled_count += 1
        key_env = model.get("api_key_env", DEFAULT_KEY_ENVS[provider])
        if not isinstance(key_env, str) or not key_env.strip():
            raise ConfigError(f"{where}.api_key_env must be a non-empty string")
        pricing = model.get("pricing")
        if pricing is not None:
            if not isinstance(pricing, dict):
                raise ConfigError(f"{where}.pricing must be an object")
            _positive_number(
                pricing.get("input_usd_per_million_tokens"),
                f"{where}.pricing.input_usd_per_million_tokens",
                allow_zero=True,
            )
            _positive_number(
                pricing.get("output_usd_per_million_tokens"),
                f"{where}.pricing.output_usd_per_million_tokens",
                allow_zero=True,
            )
            web_price = pricing.get("web_search_usd_per_1000_requests")
            if web_price is not None:
                _positive_number(
                    web_price,
                    f"{where}.pricing.web_search_usd_per_1000_requests",
                    allow_zero=True,
                )
            for metadata_key in ("as_of", "source_url"):
                metadata_value = pricing.get(metadata_key)
                if metadata_value is not None and (
                    not isinstance(metadata_value, str) or not metadata_value.strip()
                ):
                    raise ConfigError(
                        f"{where}.pricing.{metadata_key} must be a non-empty string"
                    )
    if not enabled_count:
        raise ConfigError("At least one model must be enabled")

    request = _need(config, "request", dict, "config")
    system_prompt = request.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ConfigError("config.request.system_prompt must be a string")
    _positive_int(request.get("max_output_tokens"), "config.request.max_output_tokens")
    temperature = request.get("temperature")
    if temperature is not None:
        _positive_number(temperature, "config.request.temperature", allow_zero=True)
    _positive_number(request.get("timeout_seconds"), "config.request.timeout_seconds")
    _positive_int(request.get("max_attempts"), "config.request.max_attempts")
    _positive_number(
        request.get("retry_backoff_seconds"),
        "config.request.retry_backoff_seconds",
        allow_zero=True,
    )
    _positive_int(request.get("concurrency"), "config.request.concurrency")
    _positive_int(
        request.get("abort_after_failures"), "config.request.abort_after_failures"
    )
    _positive_int(
        request.get("anthropic_web_search_max_uses"),
        "config.request.anthropic_web_search_max_uses",
    )

    estimate = _need(config, "estimate", dict, "config")
    _positive_int(
        estimate.get("input_tokens_per_call"), "config.estimate.input_tokens_per_call"
    )
    _positive_int(
        estimate.get("output_tokens_per_call"), "config.estimate.output_tokens_per_call"
    )
    _positive_number(
        estimate.get("web_searches_per_web_call"),
        "config.estimate.web_searches_per_web_call",
        allow_zero=True,
    )

    budget = _need(config, "budget", dict, "config")
    maximum = budget.get("max_estimated_usd")
    if maximum is not None:
        _positive_number(maximum, "config.budget.max_estimated_usd", allow_zero=True)

    return Inputs(
        config_path=config_path,
        prompts_path=prompts_path,
        config=config,
        prompts=prompts,
        config_sha256=config_hash,
        prompts_sha256=prompts_hash,
    )


def enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [model for model in config["models"] if model.get("enabled", True)]


def build_jobs(inputs: Inputs) -> list[dict[str, Any]]:
    config = inputs.config
    repetitions = config["experiment"]["repetitions"]
    request_fingerprint = {
        "system_prompt": config["request"].get("system_prompt", ""),
        "max_output_tokens": config["request"]["max_output_tokens"],
        "temperature": config["request"].get("temperature"),
    }
    jobs: list[dict[str, Any]] = []
    for prompt in inputs.prompts:
        for model in enabled_models(config):
            for condition in config["conditions"]:
                if condition not in model["conditions"]:
                    continue
                for repetition in range(1, repetitions + 1):
                    identity = {
                        "contract": CONTRACT_VERSION,
                        "experiment": config["experiment"]["name"],
                        "prompt": prompt,
                        "model_id": model["id"],
                        "provider": model["provider"],
                        "model": model["model"],
                        "condition": condition,
                        "repetition": repetition,
                        "request": request_fingerprint,
                    }
                    canonical = json.dumps(
                        identity,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    job_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
                    jobs.append(
                        {
                            "job_id": job_id,
                            "prompt": prompt,
                            "model_id": model["id"],
                            "provider": model["provider"],
                            "model": model["model"],
                            "condition": condition,
                            "repetition": repetition,
                        }
                    )
    return jobs


def estimate_cost(inputs: Inputs, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    models_by_id = {model["id"]: model for model in enabled_models(inputs.config)}
    estimate = inputs.config["estimate"]
    input_tokens = estimate["input_tokens_per_call"]
    output_tokens = estimate["output_tokens_per_call"]
    searches_per_web_call = float(estimate["web_searches_per_web_call"])
    by_model: dict[str, dict[str, Any]] = {}
    known_total = 0.0
    unknown: list[str] = []
    for job in jobs:
        model_id = job["model_id"]
        if model_id in by_model:
            by_model[model_id]["calls"] += 1
            if job["condition"] == "web":
                by_model[model_id]["web_calls"] += 1
            continue
        model = models_by_id[model_id]
        pricing = model.get("pricing")
        by_model[model_id] = {
            "provider": model["provider"],
            "model": model["model"],
            "calls": 1,
            "web_calls": 1 if job["condition"] == "web" else 0,
            "pricing": pricing,
            "estimated_usd": None,
        }
    for model_id, row in by_model.items():
        calls = row["calls"]
        pricing = row["pricing"]
        if pricing is None:
            unknown.append(model_id)
            continue
        amount = (
            calls
            * (
                input_tokens * float(pricing["input_usd_per_million_tokens"])
                + output_tokens * float(pricing["output_usd_per_million_tokens"])
            )
            / 1_000_000
        )
        if row["web_calls"]:
            web_price = pricing.get("web_search_usd_per_1000_requests")
            if web_price is None:
                unknown.append(model_id)
                continue
            amount += (
                row["web_calls"] * searches_per_web_call * float(web_price) / 1_000
            )
        row["estimated_usd"] = round(amount, 6)
        known_total += amount
    return {
        "assumptions": {
            "input_tokens_per_call": input_tokens,
            "output_tokens_per_call": output_tokens,
            "web_searches_per_web_call": searches_per_web_call,
        },
        "by_model": by_model,
        "known_total_usd": round(known_total, 6),
        "models_without_pricing": unknown,
        "complete": not unknown,
    }


def required_key_envs(config: dict[str, Any]) -> list[str]:
    values = []
    for model in enabled_models(config):
        values.append(model.get("api_key_env", DEFAULT_KEY_ENVS[model["provider"]]))
    return sorted(set(values))
