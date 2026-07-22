"""MCP server registry, loading, and toolset construction."""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger("orchestrator_adk")

_mcp_registry: dict[str, dict[str, Any]] | None = None


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a single YAML document from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_mcp_registry() -> dict[str, dict[str, Any]]:
    """Load the central MCP server registry from ``mcp_servers.yaml``.

    The result is cached for the lifetime of the process.
    """
    global _mcp_registry
    if _mcp_registry is not None:
        return _mcp_registry
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "mcp_servers.yaml")
    if os.path.exists(path):
        _mcp_registry = _load_yaml(path) or {}
    else:
        _mcp_registry = {}
    return _mcp_registry


def build_mcp_toolset(mcp_def: dict[str, Any]) -> object | None:
    """Build a Google ADK ``McpToolset`` from a server definition dict.

    Supports two types:
    - ``remote``: URL-based MCP servers.
    - ``local``: stdio-based MCP servers launched via a command.

    Returns ``None`` if the toolset cannot be created.
    """
    from google.adk.tools import McpToolset
    from mcp.client.stdio import StdioServerParameters

    mcp_type = mcp_def.get("type", "local")
    if mcp_type == "remote":
        url = mcp_def.get("url", "")
        if not url:
            logger.warning("Remote MCP without url: %s", mcp_def)
            return None
        try:
            return McpToolset(url=url)
        except Exception as e:
            logger.warning("Error loading remote MCP %s: %s", url, e)
            return None

    command = mcp_def.get("command", "")
    if isinstance(command, list):
        command = command[0] if command else ""
        args = mcp_def.get("command", [])[1:] if len(mcp_def.get("command", [])) > 1 else mcp_def.get("args", [])
    else:
        args = mcp_def.get("args", [])

    env = mcp_def.get("env") or mcp_def.get("environment") or None
    if env:
        resolved_env: dict[str, str] = {}
        for k, v in env.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                resolved_env[k] = os.environ.get(env_key, v)
            else:
                resolved_env[k] = v
        env = resolved_env

    try:
        params = StdioServerParameters(command=command, args=args, env=env)
        return McpToolset(connection_params=params)
    except Exception as e:
        logger.warning("Error loading local MCP %s: %s", command, e)
        return None


def load_local_mcp_registry(abs_dir: str) -> dict[str, dict[str, Any]]:
    """Load MCP server definitions from an agent's local ``mcps/`` directory.

    Parameters
    ----------
    abs_dir:
        Absolute path to the agent's directory.
    """
    registry: dict[str, dict[str, Any]] = {}
    mcps_dir = os.path.join(abs_dir, "mcps")
    if not os.path.isdir(mcps_dir):
        return registry
    for fname in sorted(os.listdir(mcps_dir)):
        if fname.endswith((".yaml", ".yml")):
            mcp_cfg = _load_yaml(os.path.join(mcps_dir, fname))
            name = mcp_cfg.get("name", fname)
            registry[name] = mcp_cfg
    return registry


def load_mcp_tools(abs_dir: str, config: dict[str, Any]) -> tuple[list[object], list[str]]:
    """Load all MCP toolsets referenced in an agent's config.

    Resolves each MCP reference against the agent's local ``mcps/`` directory
    first, then the central ``mcp_servers.yaml`` registry.

    Returns a tuple of ``(toolsets, tool_names)``.
    """
    tools: list[object] = []
    tool_names: list[str] = []

    local_registry = load_local_mcp_registry(abs_dir)
    central_registry = load_mcp_registry()

    mcp_refs = config.get("mcps", [])
    for ref in mcp_refs:
        mcp_def = local_registry.get(ref) or central_registry.get(ref)
        if mcp_def is None:
            logger.warning("  └─ MCP '%s' not found (local mcps/ nor mcp_servers.yaml)", ref)
            continue
        if mcp_def.get("enabled") is False:
            continue
        toolset = build_mcp_toolset(mcp_def)
        if toolset:
            tools.append(toolset)
            tool_names.append(ref)
            source = "local mcps/" if ref in local_registry else "central registry"
            logger.info("  └─ MCP '%s' loaded from %s", ref, source)

    return tools, tool_names
