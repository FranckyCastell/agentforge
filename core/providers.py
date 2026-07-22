"""LLM provider configuration and model-string formatting for Google ADK / LiteLLM."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("orchestrator_adk")

PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "opencode":   {"prefix": "openai",     "base": "https://opencode.ai/zen/v1",         "key_env": "OPENCODE_ZEN_API_KEY"},
    "openrouter": {"prefix": "openai",     "base": "https://openrouter.ai/api/v1",       "key_env": "OPENROUTER_API_KEY"},
    "openai":     {"prefix": "openai",     "base": None,                                  "key_env": "OPENAI_API_KEY"},
    "anthropic":  {"prefix": "anthropic",  "base": None,                                  "key_env": "ANTHROPIC_API_KEY"},
    "google":     {"prefix": "gemini",     "base": None,                                  "key_env": "GEMINI_API_KEY"},
    "xai":        {"prefix": "openai",     "base": "https://api.x.ai/v1",                 "key_env": "XAI_API_KEY"},
    "deepseek":   {"prefix": "openai",     "base": "https://api.deepseek.com/v1",         "key_env": "DEEPSEEK_API_KEY"},
    "mistral":    {"prefix": "mistral",    "base": None,                                  "key_env": "MISTRAL_API_KEY"},
    "groq":       {"prefix": "groq",       "base": None,                                  "key_env": "GROQ_API_KEY"},
    "cerebras":   {"prefix": "cerebras",   "base": None,                                  "key_env": "CEREBRAS_API_KEY"},
    "local":      {"prefix": "openai",     "base": None,                                  "key_env": None},
}

# Prefixes that share the OPENAI_API_KEY / OPENAI_API_BASE env vars.
# Only one of these can be active at a time per process.
_OPENAI_COMPATIBLE: frozenset[str] = frozenset(
    {"opencode", "openrouter", "openai", "xai", "deepseek", "local"}
)


def resolve_api_key(prov: dict[str, Any]) -> str:
    """Resolve an API key from an agent's provider config.

    Checks (in order):
    1. A literal ``api_key`` field in the config.
    2. The environment variable named in ``api_key_env``.
    3. Whether the ``api_key_env`` value itself looks like a key.
    """
    if "api_key" in prov and prov["api_key"]:
        return prov["api_key"]
    val = prov.get("api_key_env", "")
    if val and val in os.environ:
        return os.environ[val]
    if val and (
        val.startswith("sk-")
        or val.startswith("sk_")
        or val.startswith("gsk_")
        or val.startswith("csk_")
        or val.startswith("AIza")
    ):
        return val
    return ""


def format_model_for_adk(prov: dict[str, Any], agent_name: str = "") -> str:
    """Build the LiteLLM-compatible model string and set the necessary env vars.

    For OpenAI-compatible providers (OpenRouter, DeepSeek, xAI, local, …) the
    function sets ``OPENAI_API_KEY`` and ``OPENAI_API_BASE`` in the process
    environment.  Because these env vars are shared, only one OpenAI-compatible
    provider can be active at a time.  Use :func:`validate_provider_config` at
    startup to detect conflicts.

    Parameters
    ----------
    prov:
        The ``provider`` dict from an agent's ``config.yaml``.
    agent_name:
        Agent name used for logging context only.

    Returns
    -------
    str
        A model string in LiteLLM format (e.g. ``"openai/gpt-4o-mini"``).
    """
    provider_name = prov.get("name", "opencode")
    model_name = prov.get("model", "deepseek-v4-flash-free")
    base_url = prov.get("base_url")

    pconf = PROVIDER_CONFIGS.get(provider_name, PROVIDER_CONFIGS["opencode"])
    prefix = pconf["prefix"]

    if provider_name == "local" and base_url:
        os.environ["OPENAI_API_BASE"] = base_url
        api_key = resolve_api_key(prov) or "ollama"
        os.environ["OPENAI_API_KEY"] = api_key
        return f"openai/{model_name}"

    api_key = resolve_api_key(prov)
    if not api_key:
        key_env = pconf.get("key_env")
        if key_env:
            api_key = os.environ.get(key_env, "")

    if prefix == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if pconf.get("base"):
            os.environ["OPENAI_API_BASE"] = pconf["base"]
        elif base_url:
            os.environ["OPENAI_API_BASE"] = base_url
        return f"openai/{model_name}"
    elif prefix == "anthropic":
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        return f"anthropic/{model_name}"
    elif prefix == "gemini":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        return model_name
    else:
        if api_key:
            env_key = f"{prefix.upper()}_API_KEY"
            os.environ[env_key] = api_key
        return f"{prefix}/{model_name}"


def validate_provider_config(agent_provider_pairs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Detect conflicting OpenAI-compatible providers across agents.

    Parameters
    ----------
    agent_provider_pairs:
        List of ``(agent_name, provider_dict)`` tuples for every loaded agent.

    Returns
    -------
    list[str]
        Warning messages for each conflict detected (empty if none).
    """
    warnings: list[str] = []
    seen_compat: dict[str, str] = {}

    for agent_name, prov in agent_provider_pairs:
        provider_name = prov.get("name", "opencode")
        if provider_name not in _OPENAI_COMPATIBLE:
            continue
        base = prov.get("base_url") or PROVIDER_CONFIGS.get(provider_name, {}).get("base")
        key = resolve_api_key(prov) or os.environ.get(
            PROVIDER_CONFIGS.get(provider_name, {}).get("key_env", ""), ""
        )
        identity = f"{provider_name}:{base or 'default'}:{key[:8]}"
        for prev_agent, prev_identity in seen_compat.items():
            if prev_identity != identity:
                warnings.append(
                    f"Agents '{prev_agent}' and '{agent_name}' use different OpenAI-compatible "
                    f"providers — only the last-loaded one will take effect. "
                    f"({prev_identity} vs {identity})"
                )
        seen_compat[agent_name] = identity

    return warnings
