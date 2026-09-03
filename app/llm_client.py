"""Pluggable LLM client wrapper supporting Anthropic and Gemini APIs.

Usage:
    from app.llm_client import get_llm_client

    client = get_llm_client()                  # uses settings.LLM_PROVIDER
    client = get_llm_client(provider="gemini") # explicit override
    response_text = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Summarize this transcript: ...",
        max_tokens=2000,
    )

Provider-specific code is isolated in private helpers so that summarize.py
and generator.py only ever call client.generate().
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from app.config import settings

T = TypeVar("T")

DEFAULT_BACKOFF_DELAYS = (2.0, 4.0, 8.0)


def _is_transient_error(error: Exception) -> tuple[bool, str]:
    """Determine if an API exception represents a transient failure worth retrying.

    Returns:
        A tuple of (is_transient, reason_label).
    """
    # Check numeric status codes if exposed directly on the exception
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status_code, int):
        if status_code in (400, 401, 403, 404):
            return False, f"HTTP {status_code}"
        if status_code in (429, 500, 502, 503, 504):
            return True, f"{status_code}"

    # Standard python network/connection exceptions
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        if not isinstance(error, (FileNotFoundError, PermissionError)):
            return True, type(error).__name__

    # Check string representations of error message/details
    msg = str(error).lower()

    # Fast-fail for permanent errors
    for non_retryable in (
        "404",
        "not_found",
        "401",
        "unauthorized",
        "403",
        "forbidden",
        "400",
        "bad_request",
        "invalid_argument",
    ):
        if non_retryable in msg:
            return False, non_retryable

    # Transient error patterns
    if "503" in msg or "unavailable" in msg or "high demand" in msg:
        return True, "503"
    if "429" in msg or "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
        return True, "429"
    if "timeout" in msg or "timed out" in msg or "connection" in msg:
        return True, "Connection/Timeout"

    return False, "Non-transient error"


def _retry_api_call(
    provider_name: str,
    call_fn: Callable[[], T],
    backoff_delays: tuple[float, ...] = DEFAULT_BACKOFF_DELAYS,
) -> T:
    """Execute call_fn with exponential backoff on transient errors."""
    max_retries = len(backoff_delays)
    attempt = 0

    while True:
        try:
            return call_fn()
        except Exception as e:
            is_transient, reason = _is_transient_error(e)
            if not is_transient:
                raise RuntimeError(f"{provider_name} API call failed: {e}") from e

            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"{provider_name} API call failed after {max_retries} retries: {e}"
                ) from e

            delay = backoff_delays[attempt - 1]
            print(
                f"{provider_name} API busy ({reason}), retrying in {int(delay)}s... "
                f"(attempt {attempt}/{max_retries})"
            )
            time.sleep(delay)


class LLMConfigError(Exception):
    """Raised for missing API keys or misconfigured LLM provider."""


class LLMTruncationError(RuntimeError):
    """Raised when the LLM response was truncated due to hitting the max token limit.

    Callers can catch this to distinguish a partial response from a
    genuinely malformed or failed one, and potentially retry with a
    higher max_tokens budget.
    """


class LLMClient:
    """Unified LLM client that delegates to Anthropic or Gemini under the hood."""

    def __init__(self, provider: str) -> None:
        self.provider = provider.strip().lower()
        if self.provider not in ("anthropic", "gemini"):
            raise LLMConfigError(
                f"Unsupported LLM_PROVIDER: '{self.provider}'. "
                f"Must be 'anthropic' or 'gemini'."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        response_mime_type: str | None = None,
    ) -> str:
        """Send a prompt to the configured LLM and return the text response.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-facing prompt / content.
            max_tokens: Maximum tokens in the response.
            response_mime_type: Optional MIME type to constrain the response
                format. Use "application/json" to force valid JSON output
                (supported by Gemini; ignored by Anthropic).

        Returns:
            The model's text response as a string.

        Raises:
            LLMConfigError: If the API key is missing or provider is invalid.
            LLMTruncationError: If the response was truncated (hit max token limit).
            RuntimeError: If the API call itself fails.
        """
        if self.provider == "anthropic":
            return _generate_anthropic(system_prompt, user_prompt, max_tokens)
        elif self.provider == "gemini":
            return _generate_gemini(system_prompt, user_prompt, max_tokens, response_mime_type)
        else:
            raise LLMConfigError(f"Unsupported provider: {self.provider}")


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory that creates an LLMClient for the active provider.

    Args:
        provider: Override the provider from settings. If None, reads
                  settings.LLM_PROVIDER.

    Returns:
        A configured LLMClient instance.

    Raises:
        LLMConfigError: If the required API key is missing.
    """
    active_provider = (provider or settings.LLM_PROVIDER).strip().lower()

    # Validate the API key is present before constructing the client
    api_key = settings.get_api_key(active_provider)
    if not api_key or not api_key.strip():
        raise LLMConfigError(
            f"API key for provider '{active_provider}' is not set. "
            f"Please add {'ANTHROPIC_API_KEY' if active_provider == 'anthropic' else 'GEMINI_API_KEY'} "
            f"to your .env file."
        )

    return LLMClient(active_provider)


# ---------------------------------------------------------------------------
# Private provider-specific implementations
# ---------------------------------------------------------------------------

def _generate_anthropic(
    system_prompt: str, user_prompt: str, max_tokens: int
) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError as e:
        raise LLMConfigError(
            "The 'anthropic' package is required. Install with: pip install anthropic"
        ) from e

    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or not api_key.strip():
        raise LLMConfigError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    def _call():
        return client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    message = _retry_api_call("Anthropic", _call)

    # Check for truncation: Anthropic uses stop_reason == "max_tokens"
    if getattr(message, "stop_reason", None) == "max_tokens":
        partial = ""
        for block in message.content:
            if hasattr(block, "text"):
                partial += block.text
        raise LLMTruncationError(
            f"Anthropic response truncated (stop_reason=max_tokens, "
            f"max_tokens={max_tokens}). Partial response length: {len(partial)} chars."
        )

    # Extract text from the response content blocks
    text_parts = []
    for block in message.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts)


def _generate_gemini(
    system_prompt: str, user_prompt: str, max_tokens: int,
    response_mime_type: str | None = None,
) -> str:
    """Call the Google Gemini API via the google-genai SDK.

    Args:
        response_mime_type: If set (e.g. "application/json"), enables Gemini's
            constrained decoding so the model is forced to produce output
            matching the specified MIME type.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise LLMConfigError(
            "The 'google-genai' package is required. Install with: pip install google-genai"
        ) from e

    api_key = settings.GEMINI_API_KEY
    if not api_key or not api_key.strip():
        raise LLMConfigError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)

    def _call():
        return client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                # When response_mime_type is set (e.g. "application/json"),
                # Gemini uses constrained decoding to guarantee valid output format.
                response_mime_type=response_mime_type,
                # Explicitly disable automatic function calling (AFC) since this call
                # only performs text generation without tools; prevents SDK AFC warning.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

    response = _retry_api_call("Gemini", _call)

    # Debug: log finish_reason and token usage (uncomment to diagnose truncation)
    # Thinking-enabled models (e.g. gemini-3.5-flash) consume max_output_tokens
    # budget for both thinking AND output, so thoughts_token_count is critical.
    candidates = getattr(response, "candidates", None)
    _finish = None
    if candidates and len(candidates) > 0:
        _finish = getattr(candidates[0], "finish_reason", None)
    # _usage = getattr(response, "usage_metadata", None)
    # if _usage:
    #     print(
    #         f"  [llm] finish_reason={_finish}, "
    #         f"prompt={getattr(_usage, 'prompt_token_count', '?')}, "
    #         f"output={getattr(_usage, 'candidates_token_count', '?')}, "
    #         f"thoughts={getattr(_usage, 'thoughts_token_count', '?')}, "
    #         f"total={getattr(_usage, 'total_token_count', '?')}, "
    #         f"max_output_tokens={max_tokens}"
    #     )

    # Check for truncation: Gemini exposes candidates[0].finish_reason
    # Note: FinishReason is a string-backed enum; str() returns
    # "FinishReason.MAX_TOKENS" not "MAX_TOKENS", so use direct == comparison.
    if _finish is not None and _finish == "MAX_TOKENS":
        partial_text = response.text or ""
        raise LLMTruncationError(
            f"Gemini response truncated (finish_reason=MAX_TOKENS, "
            f"max_output_tokens={max_tokens}). Partial response length: {len(partial_text)} chars."
        )

    return response.text or ""
