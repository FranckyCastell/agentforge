"""Tests for core.errors — error classification and retry logic."""

import pytest

from core.errors import (
    MAX_RETRIES,
    RETRY_DELAY,
    RETRYABLE_PATTERNS,
    clean_error_msg,
    is_retryable,
)


class TestCleanErrorMsg:
    def test_upstream_error(self):
        msg = "BadRequestError: Upstream request failed"
        assert clean_error_msg(msg) == "The model provider is not available. Try again."

    def test_rate_limit_error(self):
        msg = "RateLimitError: Too many requests"
        assert clean_error_msg(msg) == "Rate limit reached. Wait a few seconds."

    def test_authentication_error(self):
        msg = "AuthenticationError: Invalid API key"
        assert clean_error_msg(msg) == "Authentication error. Check your API key in .env"

    def test_timeout_error(self):
        msg = "TimeoutError: Request timed out"
        assert clean_error_msg(msg) == "Timeout. Try again."

    def test_unknown_error_passthrough(self):
        msg = "SomeRandomError: Something went wrong"
        assert clean_error_msg(msg) == msg

    def test_empty_string(self):
        assert clean_error_msg("") == ""


class TestIsRetryable:
    @pytest.mark.parametrize("pattern", list(RETRYABLE_PATTERNS))
    def test_all_patterns_detected(self, pattern):
        assert is_retryable(f"Error: {pattern} occurred")

    def test_upstream_retryable(self):
        assert is_retryable("upstream error") is True

    def test_500_retryable(self):
        assert is_retryable("HTTP 500 Internal Server Error") is True

    def test_429_retryable(self):
        assert is_retryable("429 Too Many Requests") is True

    def test_timeout_retryable(self):
        assert is_retryable("connection timeout") is True

    def test_auth_not_retryable(self):
        assert is_retryable("AuthenticationError: bad key") is False

    def test_generic_not_retryable(self):
        assert is_retryable("ValueError: something else") is False

    def test_empty_string_not_retryable(self):
        assert is_retryable("") is False

    def test_case_insensitive(self):
        assert is_retryable("UPSTREAM ERROR") is True
        assert is_retryable("TimeOUT") is True


class TestConstants:
    def test_max_retries_is_positive(self):
        assert MAX_RETRIES > 0

    def test_retry_delay_is_positive(self):
        assert RETRY_DELAY > 0

    def test_retryable_patterns_not_empty(self):
        assert len(RETRYABLE_PATTERNS) > 0

    def test_retryable_patterns_are_strings(self):
        for p in RETRYABLE_PATTERNS:
            assert isinstance(p, str)
