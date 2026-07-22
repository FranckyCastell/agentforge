"""AgentForge core package — modular multi-agent orchestration on Google ADK.

Public API:
    load_orchestrator  — discover and load all agents from YAML configs.
    main               — CLI / JSON-events entry point.
"""

from .agent_loader import load_orchestrator
from .errors import clean_error_msg, is_retryable
from .providers import PROVIDER_CONFIGS, format_model_for_adk, resolve_api_key, validate_provider_config
from .runner import main

__all__ = [
    "PROVIDER_CONFIGS",
    "clean_error_msg",
    "format_model_for_adk",
    "is_retryable",
    "load_orchestrator",
    "main",
    "resolve_api_key",
    "validate_provider_config",
]

__version__ = "0.1.0"
