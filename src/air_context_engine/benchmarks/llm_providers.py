"""Provider-independent LLM contracts and offline OpenAI-compatible client."""

from __future__ import annotations

import math
import random
import re
import time
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429})
WAIT_SECONDS_PATTERN = re.compile(
    r"(?:please\s+wait|retry\s+after|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*s",
    flags=re.IGNORECASE,
)


class LLMProviderError(RuntimeError):
    """Provider failure with retry diagnostics safe for benchmark artifacts."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        attempt_count: int = 1,
        retry_wait_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.attempt_count = attempt_count
        self.retry_wait_seconds = retry_wait_seconds


class LLMProvider(Protocol):
    """Normalized interface consumed by answer generation and judge stages."""

    @property
    def name(self) -> str: ...

    @property
    def base_url(self) -> str | None: ...

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        *,
        json_mode: bool = False,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


def validate_local_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LOCAL_LLM_BASE_URL geçerli bir http(s) adresi olmalıdır.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "LOCAL_LLM_BASE_URL kimlik bilgisi, sorgu veya fragment içermemelidir."
        )
    if normalized.endswith("/chat/completions"):
        raise ValueError(
            "LOCAL_LLM_BASE_URL /chat/completions içermemeli; API kökünü göstermelidir."
        )
    return normalized


class LocalLLMClient:
    """Ollama, llama.cpp, vLLM and private OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries negatif olamaz.")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay negatif olamaz.")
        self._base_url = validate_local_base_url(base_url)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            base_url=f"{self._base_url}/",
            headers=headers,
            timeout=timeout,
        )
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    @property
    def name(self) -> str:
        return "local"

    @property
    def base_url(self) -> str:
        return self._base_url

    @staticmethod
    def _usage(
        payload: dict[str, Any], prompt: str, answer: str
    ) -> dict[str, int | str | bool]:
        raw = payload.get("usage")
        usage = raw if isinstance(raw, dict) else {}
        raw_input = int(usage.get("prompt_tokens") or 0)
        raw_output = int(usage.get("completion_tokens") or 0)
        input_exact = raw_input > 0
        output_exact = raw_output > 0 or not answer
        input_tokens = raw_input if input_exact else max(1, math.ceil(len(prompt) / 4))
        output_tokens = (
            raw_output if output_exact else max(1, math.ceil(len(answer) / 4))
        )
        if input_exact and output_exact:
            usage_source = "provider"
        elif not input_exact and not output_exact:
            usage_source = "estimated"
        else:
            usage_source = "mixed"
        raw_total = int(usage.get("total_tokens") or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ),
            "thought_tokens": int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                or 0
            ),
            "total_tokens": (
                raw_total
                if raw_total > 0 and input_exact and output_exact
                else input_tokens + output_tokens
            ),
            "usage_source": usage_source,
            "usage_estimated": usage_source != "provider",
        }

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in RETRYABLE_HTTP_STATUS_CODES or status_code >= 500

    def _retry_delay(
        self,
        response: httpx.Response | None,
        message: str,
        retry_number: int,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        match = WAIT_SECONDS_PATTERN.search(message)
        if match:
            return max(0.0, float(match.group(1)))
        exponential = float(getattr(self, "retry_base_delay", 1.0)) * (
            2 ** max(retry_number - 1, 0)
        )
        return exponential + random.uniform(0.0, min(exponential * 0.25, 1.0))

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        *,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        attempt_count = 0
        retry_wait_seconds = 0.0
        payload: Any = None
        response: httpx.Response | None = None
        max_retries = int(getattr(self, "max_retries", 0))
        while attempt_count <= max_retries:
            attempt_count += 1
            try:
                response = self.client.post("chat/completions", json=request)
            except httpx.HTTPError as exc:
                if attempt_count <= max_retries:
                    delay = self._retry_delay(None, str(exc), attempt_count)
                    retry_wait_seconds += delay
                    time.sleep(delay)
                    continue
                raise LLMProviderError(
                    "Yerel LLM endpoint'ine ulaşılamadı. Otomatik cloud fallback "
                    f"yapılmadı: {exc}",
                    retryable=True,
                    attempt_count=attempt_count,
                    retry_wait_seconds=retry_wait_seconds,
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(
                    "Yerel LLM endpoint'i geçerli JSON döndürmedi."
                ) from exc
            if not response.is_error:
                break
            error = payload.get("error") if isinstance(payload, dict) else None
            provider_message = error.get("message") if isinstance(error, dict) else None
            error_message = str(provider_message or response.reason_phrase)
            retryable = self._is_retryable_status(response.status_code)
            if retryable and attempt_count <= max_retries:
                delay = self._retry_delay(response, error_message, attempt_count)
                retry_wait_seconds += delay
                time.sleep(delay)
                continue
            raise LLMProviderError(
                f"Yerel LLM HTTP {response.status_code}: {error_message}",
                status_code=response.status_code,
                retryable=retryable,
                attempt_count=attempt_count,
                retry_wait_seconds=retry_wait_seconds,
            )
        if response is None:
            raise RuntimeError("Yerel LLM endpoint'i yanıt döndürmedi.")
        if not isinstance(payload, dict):
            raise ValueError("Yerel LLM yanıtı JSON nesnesi değil.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Yerel LLM yanıtında choices bulunamadı.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Yerel LLM choice kaydı geçersiz.")
        message_payload = choice.get("message")
        if not isinstance(message_payload, dict) or not isinstance(
            message_payload.get("content"), str
        ):
            raise ValueError("Yerel LLM yanıtında metin içeriği bulunamadı.")
        answer = message_payload["content"]
        return {
            "answer": answer,
            "usage": self._usage(payload, prompt, answer),
            "resolved_model_version": payload.get("model"),
            "response_metadata": {
                "id": payload.get("id"),
                "created": payload.get("created"),
                "model": payload.get("model"),
                "usage": payload.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "reasoning": message_payload.get("reasoning"),
                "backend": "local-openai-compatible",
            },
            "finish_reason": choice.get("finish_reason"),
            "attempt_count": attempt_count,
            "retry_count": max(0, attempt_count - 1),
            "retry_wait_seconds": round(retry_wait_seconds, 4),
        }

    def close(self) -> None:
        self.client.close()
