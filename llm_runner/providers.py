from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_KEY_ENVS

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}


class ProviderError(RuntimeError):
    def __init__(
        self, message: str, *, retryable: bool = False, status_code: int | None = None
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass
class ProviderResponse:
    text: str
    finish_reason: str | None
    usage: dict[str, int | None]
    searched: bool
    search_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    response_id: str | None = None


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output


def _dedupe_sources(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output = []
    for value in values:
        url = str(value.get("url", "")).strip()
        title = str(value.get("title", "")).strip()
        if not url:
            continue
        key = (url, title)
        if key not in seen:
            seen.add(key)
            output.append({"url": url, "title": title})
    return output


def _secret_for(model: dict[str, Any]) -> tuple[str, str]:
    provider = model["provider"]
    env_name = model.get("api_key_env", DEFAULT_KEY_ENVS[provider])
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ProviderError(f"Missing API key environment variable: {env_name}")
    return env_name, value


def _redact(text: str, secrets: list[str]) -> str:
    output = text
    for secret in secrets:
        if secret:
            output = output.replace(secret, "[REDACTED]")
    return output[:4000]


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
    secrets: list[str] | None = None,
) -> dict[str, Any]:
    body = None
    request_headers = {"Accept": "application/json", **headers}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(exc)
        detail = _redact(detail, secrets or [])
        retryable = (
            exc.code == 408 or exc.code == 409 or exc.code == 429 or exc.code >= 500
        )
        raise ProviderError(
            f"HTTP {exc.code}: {detail}",
            retryable=retryable,
            status_code=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        detail = _redact(str(exc), secrets or [])
        raise ProviderError(f"Network error: {detail}", retryable=True) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider returned invalid JSON", retryable=False) from exc
    if not isinstance(value, dict):
        raise ProviderError(
            "Provider returned a non-object JSON response", retryable=False
        )
    return value


def _base_url(model: dict[str, Any]) -> str:
    return str(model.get("base_url", DEFAULT_BASE_URLS[model["provider"]])).rstrip("/")


def _usage(
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
) -> dict[str, int | None]:
    def integer(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    input_value = integer(input_tokens)
    output_value = integer(output_tokens)
    total_value = integer(total_tokens)
    if total_value is None and input_value is not None and output_value is not None:
        total_value = input_value + output_value
    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": total_value,
    }


def call_openai(
    model: dict[str, Any],
    condition: str,
    prompt: str,
    request_config: dict[str, Any],
) -> ProviderResponse:
    _, secret = _secret_for(model)
    payload: dict[str, Any] = {
        "model": model["model"],
        "input": prompt,
        "max_output_tokens": request_config["max_output_tokens"],
    }
    system_prompt = request_config.get("system_prompt", "").strip()
    if system_prompt:
        payload["instructions"] = system_prompt
    temperature = request_config.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature
    if condition == "web":
        payload["tools"] = [{"type": "web_search"}]

    response = request_json(
        "POST",
        f"{_base_url(model)}/responses",
        headers={"Authorization": f"Bearer {secret}"},
        payload=payload,
        timeout=float(request_config["timeout_seconds"]),
        secrets=[secret],
    )
    texts: list[str] = []
    queries: list[str] = []
    sources: list[dict[str, str]] = []
    searched = False
    search_count = 0
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            searched = True
            search_count += 1
            action = item.get("action")
            if isinstance(action, dict):
                query = action.get("query")
                if isinstance(query, str):
                    queries.append(query)
                query_list = action.get("queries")
                if isinstance(query_list, list):
                    queries.extend(
                        value for value in query_list if isinstance(value, str)
                    )
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
            for annotation in content.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                citation = annotation.get("url_citation")
                if isinstance(citation, dict):
                    url = citation.get("url")
                    title = citation.get("title", "")
                else:
                    url = annotation.get("url")
                    title = annotation.get("title", "")
                if isinstance(url, str):
                    sources.append({"url": url, "title": str(title or "")})
    output_text = response.get("output_text")
    if not texts and isinstance(output_text, str):
        texts.append(output_text)
    text = "\n\n".join(value.strip() for value in texts if value.strip()).strip()
    if not text:
        raise ProviderError(
            "OpenAI response did not contain answer text", retryable=False
        )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return ProviderResponse(
        text=text,
        finish_reason=str(response.get("status"))
        if response.get("status") is not None
        else None,
        usage=_usage(
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        ),
        searched=searched,
        search_count=search_count,
        search_queries=_dedupe_strings(queries),
        sources=_dedupe_sources(sources),
        response_id=str(response.get("id")) if response.get("id") else None,
    )


def _anthropic_request(
    model: dict[str, Any],
    secret: str,
    payload: dict[str, Any],
    request_config: dict[str, Any],
) -> dict[str, Any]:
    return request_json(
        "POST",
        f"{_base_url(model)}/messages",
        headers={
            "x-api-key": secret,
            "anthropic-version": "2023-06-01",
        },
        payload=payload,
        timeout=float(request_config["timeout_seconds"]),
        secrets=[secret],
    )


def call_anthropic(
    model: dict[str, Any],
    condition: str,
    prompt: str,
    request_config: dict[str, Any],
) -> ProviderResponse:
    _, secret = _secret_for(model)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    payload: dict[str, Any] = {
        "model": model["model"],
        "max_tokens": request_config["max_output_tokens"],
        "messages": messages,
    }
    system_prompt = request_config.get("system_prompt", "").strip()
    if system_prompt:
        payload["system"] = system_prompt
    temperature = request_config.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature
    if condition == "web":
        payload["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": request_config["anthropic_web_search_max_uses"],
            }
        ]

    responses: list[dict[str, Any]] = []
    for _ in range(4):
        response = _anthropic_request(model, secret, payload, request_config)
        responses.append(response)
        if response.get("stop_reason") != "pause_turn":
            break
        content = response.get("content")
        if not isinstance(content, list):
            raise ProviderError("Anthropic pause_turn response had no reusable content")
        messages.append({"role": "assistant", "content": content})
        payload["messages"] = messages
    else:
        raise ProviderError(
            "Anthropic response remained paused after four continuations"
        )

    texts: list[str] = []
    queries: list[str] = []
    sources: list[dict[str, str]] = []
    warnings: list[str] = []
    searched = False
    input_tokens = 0
    output_tokens = 0
    saw_input_usage = False
    saw_output_usage = False
    usage_search_count = 0
    observed_search_count = 0
    for response in responses:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        if isinstance(usage.get("input_tokens"), int):
            input_tokens += usage["input_tokens"]
            saw_input_usage = True
        if isinstance(usage.get("output_tokens"), int):
            output_tokens += usage["output_tokens"]
            saw_output_usage = True
        server_tool_usage = usage.get("server_tool_use")
        if isinstance(server_tool_usage, dict) and isinstance(
            server_tool_usage.get("web_search_requests"), int
        ):
            usage_search_count += server_tool_usage["web_search_requests"]
        for block in response.get("content", []):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
                for citation in block.get("citations", []):
                    if not isinstance(citation, dict):
                        continue
                    url = citation.get("url")
                    if isinstance(url, str):
                        sources.append(
                            {"url": url, "title": str(citation.get("title", "") or "")}
                        )
            elif block_type == "server_tool_use":
                searched = True
                if block.get("name") == "web_search":
                    observed_search_count += 1
                tool_input = block.get("input")
                if isinstance(tool_input, dict) and isinstance(
                    tool_input.get("query"), str
                ):
                    queries.append(tool_input["query"])
            elif block_type == "web_search_tool_result":
                searched = True
                if block.get("is_error"):
                    warnings.append(
                        str(block.get("content", "Anthropic web search tool error"))
                    )
                content = block.get("content")
                if isinstance(content, list):
                    for result in content:
                        if not isinstance(result, dict):
                            continue
                        url = result.get("url")
                        if isinstance(url, str):
                            sources.append(
                                {
                                    "url": url,
                                    "title": str(result.get("title", "") or ""),
                                }
                            )
                elif isinstance(content, dict) and content.get("error_code"):
                    warnings.append(
                        f"Anthropic web search error: {content['error_code']}"
                    )
    text = "\n\n".join(value.strip() for value in texts if value.strip()).strip()
    if not text:
        raise ProviderError(
            "Anthropic response did not contain answer text", retryable=False
        )
    final = responses[-1]
    return ProviderResponse(
        text=text,
        finish_reason=str(final.get("stop_reason"))
        if final.get("stop_reason")
        else None,
        usage=_usage(
            input_tokens if saw_input_usage else None,
            output_tokens if saw_output_usage else None,
        ),
        searched=searched,
        search_count=usage_search_count or observed_search_count,
        search_queries=_dedupe_strings(queries),
        sources=_dedupe_sources(sources),
        warnings=_dedupe_strings(warnings),
        response_id=str(final.get("id")) if final.get("id") else None,
    )


def call_google(
    model: dict[str, Any],
    condition: str,
    prompt: str,
    request_config: dict[str, Any],
) -> ProviderResponse:
    _, secret = _secret_for(model)
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": request_config["max_output_tokens"],
        },
    }
    system_prompt = request_config.get("system_prompt", "").strip()
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    temperature = request_config.get("temperature")
    if temperature is not None:
        payload["generationConfig"]["temperature"] = temperature
    if condition == "web":
        payload["tools"] = [{"google_search": {}}]

    quoted_model = urllib.parse.quote(model["model"], safe="")
    response = request_json(
        "POST",
        f"{_base_url(model)}/models/{quoted_model}:generateContent",
        headers={"x-goog-api-key": secret},
        payload=payload,
        timeout=float(request_config["timeout_seconds"]),
        secrets=[secret],
    )
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = response.get("promptFeedback")
        raise ProviderError(
            f"Google response had no candidates: {feedback}", retryable=False
        )
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ProviderError("Google returned an invalid candidate", retryable=False)
    content = (
        candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
    )
    texts = []
    for part in content.get("parts", []):
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    text = "\n\n".join(value.strip() for value in texts if value.strip()).strip()
    if not text:
        raise ProviderError(
            "Google response did not contain answer text", retryable=False
        )

    metadata = (
        candidate.get("groundingMetadata")
        if isinstance(candidate.get("groundingMetadata"), dict)
        else {}
    )
    queries = [
        value
        for value in metadata.get("webSearchQueries", [])
        if isinstance(value, str)
    ]
    sources: list[dict[str, str]] = []
    for chunk in metadata.get("groundingChunks", []):
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        url = web.get("uri")
        if isinstance(url, str):
            sources.append({"url": url, "title": str(web.get("title", "") or "")})
    usage = (
        response.get("usageMetadata")
        if isinstance(response.get("usageMetadata"), dict)
        else {}
    )
    return ProviderResponse(
        text=text,
        finish_reason=(
            str(candidate.get("finishReason"))
            if candidate.get("finishReason")
            else None
        ),
        usage=_usage(
            usage.get("promptTokenCount"),
            usage.get("candidatesTokenCount"),
            usage.get("totalTokenCount"),
        ),
        searched=bool(queries or sources),
        search_count=len(queries) if queries else (1 if sources else 0),
        search_queries=_dedupe_strings(queries),
        sources=_dedupe_sources(sources),
        response_id=str(response.get("responseId"))
        if response.get("responseId")
        else None,
    )


def call_provider(
    model: dict[str, Any],
    condition: str,
    prompt: str,
    request_config: dict[str, Any],
) -> ProviderResponse:
    provider = model["provider"]
    if provider == "openai":
        return call_openai(model, condition, prompt, request_config)
    if provider == "anthropic":
        return call_anthropic(model, condition, prompt, request_config)
    if provider == "google":
        return call_google(model, condition, prompt, request_config)
    raise ProviderError(f"Unsupported provider: {provider}")


def list_models(
    provider: str,
    *,
    api_key_env: str | None,
    base_url: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    model = {
        "provider": provider,
        "api_key_env": api_key_env or DEFAULT_KEY_ENVS[provider],
    }
    if base_url:
        model["base_url"] = base_url
    _, secret = _secret_for(model)
    url_root = _base_url(model)
    if provider == "openai":
        response = request_json(
            "GET",
            f"{url_root}/models",
            headers={"Authorization": f"Bearer {secret}"},
            payload=None,
            timeout=timeout,
            secrets=[secret],
        )
        values = response.get("data", [])
        return [
            {
                "id": value.get("id"),
                "created": value.get("created"),
                "owned_by": value.get("owned_by"),
            }
            for value in values
            if isinstance(value, dict) and isinstance(value.get("id"), str)
        ]
    if provider == "anthropic":
        response = request_json(
            "GET",
            f"{url_root}/models?limit=1000",
            headers={"x-api-key": secret, "anthropic-version": "2023-06-01"},
            payload=None,
            timeout=timeout,
            secrets=[secret],
        )
        values = response.get("data", [])
        return [
            {
                "id": value.get("id"),
                "display_name": value.get("display_name"),
                "created_at": value.get("created_at"),
            }
            for value in values
            if isinstance(value, dict) and isinstance(value.get("id"), str)
        ]
    if provider == "google":
        response = request_json(
            "GET",
            f"{url_root}/models?pageSize=1000",
            headers={"x-goog-api-key": secret},
            payload=None,
            timeout=timeout,
            secrets=[secret],
        )
        values = response.get("models", [])
        output = []
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                continue
            name = value["name"]
            output.append(
                {
                    "id": name.removeprefix("models/"),
                    "display_name": value.get("displayName"),
                    "supported_methods": value.get("supportedGenerationMethods", []),
                }
            )
        return output
    raise ProviderError(f"Unsupported provider: {provider}")
