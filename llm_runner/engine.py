from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import CONTRACT_VERSION, __version__
from .config import DEFAULT_KEY_ENVS, Inputs, build_jobs, enabled_models, estimate_cost
from .providers import ProviderError, ProviderResponse, call_provider


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Python 3.9 and 3.10 do not accept a trailing Z here.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return clean[:60] or "experiment"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")  # noqa: TRY004
    return value


def _manifest_for(
    inputs: Inputs, jobs: list[dict[str, Any]], run_id: str
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "runner_version": __version__,
        "run_id": run_id,
        "created_at": utc_now(),
        "experiment_name": inputs.config["experiment"]["name"],
        "config_sha256": inputs.config_sha256,
        "prompts_sha256": inputs.prompts_sha256,
        "config_snapshot": inputs.config,
        "prompts_snapshot": {
            "schema_version": CONTRACT_VERSION,
            "prompts": inputs.prompts,
        },
        "estimate": estimate_cost(inputs, jobs),
        "planned_jobs": jobs,
    }


def prepare_run(inputs: Inputs, resume_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    jobs = build_jobs(inputs)
    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("schema_version") != CONTRACT_VERSION:
            raise ValueError(
                f"Run contract {manifest.get('schema_version')!r} is not supported; "
                f"expected {CONTRACT_VERSION!r}"
            )
        if manifest.get("config_snapshot") != inputs.config:
            raise ValueError(
                "The current config does not match this run's immutable manifest. "
                "Restore the original config or start a new run."
            )
        expected_prompts = {
            "schema_version": CONTRACT_VERSION,
            "prompts": inputs.prompts,
        }
        if manifest.get("prompts_snapshot") != expected_prompts:
            raise ValueError(
                "The current prompts do not match this run's immutable manifest. "
                "Restore the original prompts or start a new run."
            )
        if manifest.get("planned_jobs") != jobs:
            raise ValueError("The generated job plan does not match the run manifest")
        return run_dir, manifest

    output_dir = Path(inputs.config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = inputs.config_path.parent / output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    stem = f"{stamp}-{_slug(inputs.config['experiment']['name'])}"
    run_dir = output_dir / stem
    suffix = 2
    while run_dir.exists():
        run_dir = output_dir / f"{stem}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    run_id = run_dir.name
    manifest = _manifest_for(inputs, jobs, run_id)
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _write_json_atomic(
        run_dir / "status.json",
        {
            "schema_version": CONTRACT_VERSION,
            "run_id": run_id,
            "state": "ready",
            "updated_at": utc_now(),
            "heartbeat_at": utc_now(),
            "planned": len(jobs),
            "successful": 0,
            "failed": 0,
            "unfinished": len(jobs),
        },
    )
    return run_dir, manifest


def read_attempts(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        return [], []
    attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"results.jsonl line {line_number}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"results.jsonl line {line_number}: expected object")
                continue
            attempts.append(value)
    return attempts, errors


def latest_by_job(attempts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        job_id = attempt.get("job_id")
        if isinstance(job_id, str):
            output[job_id] = attempt
    return output


def _counts(manifest: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, int]:
    planned_ids = {job["job_id"] for job in manifest["planned_jobs"]}
    latest = latest_by_job(attempts)
    successful = sum(
        1
        for job_id, attempt in latest.items()
        if job_id in planned_ids and attempt.get("status") == "success"
    )
    failed = sum(
        1
        for job_id, attempt in latest.items()
        if job_id in planned_ids and attempt.get("status") == "error"
    )
    return {
        "planned": len(planned_ids),
        "successful": successful,
        "failed": failed,
        "unfinished": len(planned_ids) - successful - failed,
    }


def _status(
    run_dir: Path,
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
    state: str,
    message: str | None = None,
) -> dict[str, Any]:
    status = {
        "schema_version": CONTRACT_VERSION,
        "run_id": manifest["run_id"],
        "state": state,
        "updated_at": utc_now(),
        "heartbeat_at": utc_now(),
        **_counts(manifest, attempts),
    }
    if message:
        status["message"] = message
    _write_json_atomic(run_dir / "status.json", status)
    return status


def _append_attempt(run_dir: Path, attempt: dict[str, Any]) -> None:
    path = run_dir / "results.jsonl"
    serialized = json.dumps(attempt, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _redact_unexpected(message: str, inputs: Inputs) -> str:
    output = message
    for model in enabled_models(inputs.config):
        env_name = model.get("api_key_env", DEFAULT_KEY_ENVS[model["provider"]])
        secret = os.environ.get(env_name)
        if secret:
            output = output.replace(secret, "[REDACTED]")
    return output[:4000]


def _actual_cost(model: dict[str, Any], response: ProviderResponse) -> float | None:
    pricing = model.get("pricing")
    input_tokens = response.usage.get("input_tokens")
    output_tokens = response.usage.get("output_tokens")
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    value = (
        input_tokens * float(pricing["input_usd_per_million_tokens"])
        + output_tokens * float(pricing["output_usd_per_million_tokens"])
    ) / 1_000_000
    if response.search_count:
        web_price = pricing.get("web_search_usd_per_1000_requests")
        if web_price is None:
            return None
        value += response.search_count * float(web_price) / 1_000
    return round(value, 8)


def execute_job(
    inputs: Inputs,
    manifest: dict[str, Any],
    job: dict[str, Any],
    sequence: int,
    *,
    provider_call: Callable[
        [dict[str, Any], str, str, dict[str, Any]], ProviderResponse
    ] = call_provider,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    model_by_id = {model["id"]: model for model in enabled_models(inputs.config)}
    model = model_by_id[job["model_id"]]
    request_config = inputs.config["request"]
    started_at = utc_now()
    start = time.monotonic()
    max_attempts = request_config["max_attempts"]
    last_error: Exception | None = None
    request_attempts = 0
    response: ProviderResponse | None = None
    for attempt_number in range(1, max_attempts + 1):
        request_attempts = attempt_number
        try:
            response = provider_call(
                model,
                job["condition"],
                job["prompt"]["text"],
                request_config,
            )
            break
        except ProviderError as exc:
            last_error = exc
            if not exc.retryable or attempt_number == max_attempts:
                break
        # Provider adapters are an extension boundary. Convert unexpected adapter
        # failures into a stored job error instead of crashing the whole run.
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break
        delay = float(request_config["retry_backoff_seconds"]) * (
            2 ** (attempt_number - 1)
        )
        if delay:
            sleep(delay)

    completed_at = utc_now()
    latency_ms = round((time.monotonic() - start) * 1000)
    base = {
        "schema_version": CONTRACT_VERSION,
        "attempt_id": str(uuid.uuid4()),
        "attempt_sequence": sequence,
        "job_id": job["job_id"],
        "run_id": manifest["run_id"],
        "experiment_name": manifest["experiment_name"],
        "provider": job["provider"],
        "model_id": job["model_id"],
        "model": job["model"],
        "condition": job["condition"],
        "prompt": job["prompt"],
        "repetition": job["repetition"],
        "started_at": started_at,
        "completed_at": completed_at,
        "latency_ms": latency_ms,
        "request_attempts": request_attempts,
    }
    if response is not None:
        return {
            **base,
            "status": "success",
            "response": {
                "id": response.response_id,
                "text": response.text,
                "finish_reason": response.finish_reason,
            },
            "usage": response.usage,
            "estimated_cost_usd": _actual_cost(model, response),
            "web": {
                "searched": response.searched,
                "search_count": response.search_count,
                "queries": response.search_queries,
                "sources": response.sources,
            },
            "warnings": response.warnings,
            "error": None,
        }

    error = last_error or RuntimeError("Provider call ended without a response")
    status_code = error.status_code if isinstance(error, ProviderError) else None
    retryable = error.retryable if isinstance(error, ProviderError) else False
    return {
        **base,
        "status": "error",
        "response": None,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "estimated_cost_usd": None,
        "web": {"searched": False, "search_count": 0, "queries": [], "sources": []},
        "warnings": [],
        "error": {
            "type": type(error).__name__,
            "message": _redact_unexpected(str(error), inputs),
            "status_code": status_code,
            "retryable": retryable,
        },
    }


def run_experiment(
    inputs: Inputs,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    provider_call: Callable[
        [dict[str, Any], str, str, dict[str, Any]], ProviderResponse
    ] = call_provider,
) -> tuple[int, dict[str, Any]]:
    attempts, parse_errors = read_attempts(run_dir)
    if parse_errors:
        raise ValueError(
            "Cannot resume because results.jsonl is malformed:\n- "
            + "\n- ".join(parse_errors)
        )
    latest = latest_by_job(attempts)
    pending = [
        job
        for job in manifest["planned_jobs"]
        if latest.get(job["job_id"], {}).get("status") != "success"
    ]
    if not pending:
        status = _status(
            run_dir, manifest, attempts, "complete", "All jobs already succeeded"
        )
        return 0, status

    previous_attempt_counts: dict[str, int] = {}
    for attempt in attempts:
        job_id = attempt.get("job_id")
        if isinstance(job_id, str):
            previous_attempt_counts[job_id] = previous_attempt_counts.get(job_id, 0) + 1

    _status(run_dir, manifest, attempts, "running")
    workers = min(inputs.config["request"]["concurrency"], len(pending))
    abort_after = inputs.config["request"]["abort_after_failures"]
    failures_this_invocation = 0
    next_index = 0
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-runner")

    def submit(job: dict[str, Any]) -> None:
        sequence = previous_attempt_counts.get(job["job_id"], 0) + 1
        future = executor.submit(
            execute_job,
            inputs,
            manifest,
            job,
            sequence,
            provider_call=provider_call,
        )
        futures[future] = job

    try:
        while next_index < len(pending) and len(futures) < workers:
            submit(pending[next_index])
            next_index += 1

        processed_this_invocation = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    attempt = future.result()
                # A worker must not be able to take down unrelated completed jobs.
                except Exception as exc:  # noqa: BLE001
                    attempt = {
                        "schema_version": CONTRACT_VERSION,
                        "attempt_id": str(uuid.uuid4()),
                        "attempt_sequence": previous_attempt_counts.get(
                            job["job_id"], 0
                        )
                        + 1,
                        "job_id": job["job_id"],
                        "run_id": manifest["run_id"],
                        "experiment_name": manifest["experiment_name"],
                        "provider": job["provider"],
                        "model_id": job["model_id"],
                        "model": job["model"],
                        "condition": job["condition"],
                        "prompt": job["prompt"],
                        "repetition": job["repetition"],
                        "started_at": utc_now(),
                        "completed_at": utc_now(),
                        "latency_ms": 0,
                        "request_attempts": 0,
                        "status": "error",
                        "response": None,
                        "usage": {
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                        },
                        "estimated_cost_usd": None,
                        "web": {
                            "searched": False,
                            "search_count": 0,
                            "queries": [],
                            "sources": [],
                        },
                        "warnings": [],
                        "error": {
                            "type": type(exc).__name__,
                            "message": _redact_unexpected(str(exc), inputs),
                            "status_code": None,
                            "retryable": False,
                        },
                    }
                _append_attempt(run_dir, attempt)
                attempts.append(attempt)
                processed_this_invocation += 1
                if attempt["status"] == "error":
                    failures_this_invocation += 1
                counts = _counts(manifest, attempts)
                print(
                    f"[{counts['successful']}/{counts['planned']}] "
                    f"{attempt['status']} · {attempt['model_id']} · "
                    f"{attempt['condition']} · {attempt['prompt']['id']} · "
                    f"repeat {attempt['repetition']}",
                    flush=True,
                )
                _status(run_dir, manifest, attempts, "running")

            if failures_this_invocation < abort_after:
                while next_index < len(pending) and len(futures) < workers:
                    submit(pending[next_index])
                    next_index += 1
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        status = _status(
            run_dir,
            manifest,
            attempts,
            "interrupted",
            "Interrupted by user; resume with the same config and run directory",
        )
        return 130, status
    else:
        executor.shutdown(wait=True)

    counts = _counts(manifest, attempts)
    latest = latest_by_job(attempts)
    all_success = all(
        latest.get(job["job_id"], {}).get("status") == "success"
        for job in manifest["planned_jobs"]
    )
    if all_success:
        state = "complete"
        message = "Every planned job succeeded"
        exit_code = 0
    elif failures_this_invocation >= abort_after and next_index < len(pending):
        state = "stopped_failure_threshold"
        message = (
            f"Stopped scheduling new jobs after {failures_this_invocation} failures in this "
            "invocation. Fix the cause, then resume."
        )
        exit_code = 2
    else:
        state = "incomplete"
        message = "Every pending job was attempted, but one or more jobs still failed"
        exit_code = 2
    status = _status(run_dir, manifest, attempts, state, message)
    return exit_code, status


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    attempts, parse_errors = read_attempts(run_dir)
    jobs = manifest.get("planned_jobs")
    if not isinstance(jobs, list):
        raise ValueError("manifest.json has no valid planned_jobs array")  # noqa: TRY004
    planned_ids = {job.get("job_id") for job in jobs if isinstance(job, dict)}
    latest = latest_by_job(attempts)
    unknown_job_ids = sorted(job_id for job_id in latest if job_id not in planned_ids)
    missing = sorted(job_id for job_id in planned_ids if job_id not in latest)
    failed = sorted(
        job_id
        for job_id in planned_ids
        if latest.get(job_id, {}).get("status") == "error"
    )
    successful = sorted(
        job_id
        for job_id in planned_ids
        if latest.get(job_id, {}).get("status") == "success"
    )
    invalid_contract_attempts = [
        attempt.get("attempt_id", "unknown")
        for attempt in attempts
        if attempt.get("schema_version") != CONTRACT_VERSION
    ]
    duplicate_attempts = len(attempts) - len(latest)
    valid = not (
        parse_errors
        or unknown_job_ids
        or missing
        or failed
        or invalid_contract_attempts
    )
    return {
        "schema_version": CONTRACT_VERSION,
        "run_id": manifest.get("run_id"),
        "valid": valid,
        "planned_jobs": len(planned_ids),
        "successful_jobs": len(successful),
        "failed_jobs": len(failed),
        "missing_jobs": len(missing),
        "attempt_records": len(attempts),
        "reattempt_records": duplicate_attempts,
        "parse_errors": parse_errors,
        "unknown_job_ids": unknown_job_ids,
        "failed_job_ids": failed,
        "missing_job_ids": missing,
        "invalid_contract_attempt_ids": invalid_contract_attempts,
    }


def inspect_status(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    attempts, parse_errors = read_attempts(run_dir)
    status_path = run_dir / "status.json"
    status = _read_json(status_path) if status_path.exists() else {}
    counts = _counts(manifest, attempts)
    output = {**status, **counts, "parse_errors": parse_errors}
    heartbeat = _parse_time(status.get("heartbeat_at"))
    request = manifest.get("config_snapshot", {}).get("request", {})
    timeout = request.get("timeout_seconds", 90)
    max_attempts = request.get("max_attempts", 3)
    try:
        stale_after = max(180.0, float(timeout) * int(max_attempts) * 2)
    except (TypeError, ValueError):
        stale_after = 540.0
    if status.get("state") == "running" and heartbeat is not None:
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        output["heartbeat_age_seconds"] = round(age)
        output["stale_after_seconds"] = round(stale_after)
        output["appears_stale"] = age > stale_after
    else:
        output["appears_stale"] = False
    return output
