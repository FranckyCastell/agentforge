"""Error classification and retry logic for AgentForge."""

from __future__ import annotations

MAX_RETRIES: int = 2
RETRY_DELAY: float = 1.5
RETRYABLE_PATTERNS: tuple[str, ...] = (
    "upstream",
    "500",
    "502",
    "503",
    "429",
    "timeout",
    "connection",
)


def clean_error_msg(msg: str) -> str:
    """Translate raw provider error strings into user-friendly messages."""
    if "BadRequestError" in msg and "Upstream request failed" in msg:
        return "The model provider is not available. Try again."
    if "RateLimitError" in msg:
        return "Rate limit reached. Wait a few seconds."
    if "AuthenticationError" in msg:
        return "Authentication error. Check your API key in .env"
    if "Timeout" in msg:
        return "Timeout. Try again."
    return msg


def is_retryable(err_str: str) -> bool:
    """Return True if the error message indicates a transient failure worth retrying."""
    err_lower = err_str.lower()
    return any(p in err_lower for p in RETRYABLE_PATTERNS)
