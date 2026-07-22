"""Tests for core.providers — API key resolution, model formatting, and conflict detection."""

import os

import pytest

from core.providers import (
    PROVIDER_CONFIGS,
    format_model_for_adk,
    resolve_api_key,
    validate_provider_config,
)


class TestResolveApiKey:
    def test_literal_api_key(self):
        prov = {"api_key": "sk-literal-key"}
        assert resolve_api_key(prov) == "sk-literal-key"

    def test_env_var_lookup(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        prov = {"api_key_env": "TEST_API_KEY"}
        assert resolve_api_key(prov) == "sk-from-env"

    def test_env_var_value_is_itself_a_key(self):
        prov = {"api_key_env": "sk-looks-like-key"}
        assert resolve_api_key(prov) == "sk-looks-like-key"

    def test_groq_key_prefix(self):
        prov = {"api_key_env": "gsk_123456"}
        assert resolve_api_key(prov) == "gsk_123456"

    def test_cerebras_key_prefix(self):
        prov = {"api_key_env": "csk_123456"}
        assert resolve_api_key(prov) == "csk_123456"

    def test_google_key_prefix(self):
        prov = {"api_key_env": "AIzaSyTest123"}
        assert resolve_api_key(prov) == "AIzaSyTest123"

    def test_empty(self):
        prov = {}
        assert resolve_api_key(prov) == ""

    def test_no_matching_env(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        prov = {"api_key_env": "NONEXISTENT_KEY"}
        assert resolve_api_key(prov) == ""


class TestFormatModelForAdk:
    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = format_model_for_adk({"name": "openai", "model": "gpt-4o-mini"})
        assert result == "openai/gpt-4o-mini"
        assert os.environ.get("OPENAI_API_KEY") == "sk-test"

    def test_openrouter_provider(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = format_model_for_adk({"name": "openrouter", "model": "deepseek-v4-flash"})
        assert result == "openai/deepseek-v4-flash"
        assert os.environ.get("OPENAI_API_BASE") == "https://openrouter.ai/api/v1"

    def test_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        result = format_model_for_adk({"name": "anthropic", "model": "claude-3-5-sonnet-20241022"})
        assert result == "anthropic/claude-3-5-sonnet-20241022"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test"

    def test_google_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaTest")
        result = format_model_for_adk({"name": "google", "model": "gemini-2.5-flash"})
        assert result == "gemini-2.5-flash"

    def test_groq_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        result = format_model_for_adk({"name": "groq", "model": "llama-3.1-70b-versatile"})
        assert result == "groq/llama-3.1-70b-versatile"

    def test_local_provider_with_base_url(self, monkeypatch):
        result = format_model_for_adk({
            "name": "local",
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
        })
        assert result == "openai/llama3.2"
        assert os.environ.get("OPENAI_API_BASE") == "http://localhost:11434/v1"
        assert os.environ.get("OPENAI_API_KEY") == "ollama"

    def test_deepseek_provider(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
        result = format_model_for_adk({"name": "deepseek", "model": "deepseek-chat"})
        assert result == "openai/deepseek-chat"
        assert os.environ.get("OPENAI_API_BASE") == "https://api.deepseek.com/v1"

    def test_unknown_provider_falls_back_to_opencode(self):
        result = format_model_for_adk({"name": "unknown_provider", "model": "some-model"})
        assert result == "openai/some-model"


class TestValidateProviderConfig:
    def test_no_conflicts_same_provider(self):
        prov = {"name": "openrouter", "model": "deepseek-v4-flash"}
        warnings = validate_provider_config([
            ("agent1", prov),
            ("agent2", prov),
        ])
        assert warnings == []

    def test_no_warnings_for_non_compatible_providers(self):
        warnings = validate_provider_config([
            ("agent1", {"name": "anthropic"}),
            ("agent2", {"name": "google"}),
        ])
        assert warnings == []

    def test_conflict_detected_different_openai_compatible(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-key-1234")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-key-5678")
        warnings = validate_provider_config([
            ("weather_agent", {"name": "openrouter", "model": "deepseek-v4-flash"}),
            ("news_agent", {"name": "deepseek", "model": "deepseek-chat"}),
        ])
        assert len(warnings) >= 1
        assert "weather_agent" in warnings[0]
        assert "news_agent" in warnings[0]


class TestProviderConfigs:
    def test_all_providers_have_prefix(self):
        for name, conf in PROVIDER_CONFIGS.items():
            assert "prefix" in conf, f"Provider '{name}' missing 'prefix'"

    def test_all_providers_have_base(self):
        for name, conf in PROVIDER_CONFIGS.items():
            assert "base" in conf, f"Provider '{name}' missing 'base'"

    def test_expected_provider_count(self):
        assert len(PROVIDER_CONFIGS) == 11

    def test_provider_names(self):
        expected = {"opencode", "openrouter", "openai", "anthropic", "google",
                     "xai", "deepseek", "mistral", "groq", "cerebras", "local"}
        assert set(PROVIDER_CONFIGS.keys()) == expected
