"""Agent discovery and loading from YAML configuration files."""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from dotenv import load_dotenv
from google.adk import Agent
from google.genai import types

from .mcp_loader import load_mcp_tools
from .prompt_builder import generate_orchestrator_prompt
from .providers import format_model_for_adk, validate_provider_config
from .skills import load_skills, load_yaml

logger = logging.getLogger("orchestrator_adk")

INTERNAL_TOOLS: frozenset[str] = frozenset({"transfer_to_agent", "transfer_agent"})


def load_agent_from_dir(agent_dir: str) -> tuple[Agent, dict[str, Any]]:
    """Load a single agent from its directory.

    The directory must contain a ``config.yaml``. A ``prompt.yaml`` is optional
    but recommended. Skills and MCP tools are loaded automatically.

    Returns a tuple of ``(Agent, meta_dict)`` where *meta_dict* contains
    display metadata used by the orchestrator prompt and JSON-events output.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_dir = os.path.join(project_root, agent_dir)
    env_file = os.path.join(abs_dir, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    cfg_path = os.path.join(abs_dir, "config.yaml")
    prompt_path = os.path.join(abs_dir, "prompt.yaml")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {abs_dir}")

    config = load_yaml(cfg_path)
    prompt_cfg = load_yaml(prompt_path) if os.path.exists(prompt_path) else {}
    system_prompt = prompt_cfg.get("system_prompt", "")

    skills_info = load_skills(abs_dir, config)

    desc_skills = ""
    if skills_info:
        skills_parts: list[str] = []
        for s in skills_info:
            sid = s.get("id", "")
            desc = s.get("description", "")
            tags = ", ".join(s.get("tags", []))
            examples = ", ".join(s.get("examples", [])[:2])
            skills_parts.append(
                f"[{sid}] {desc}"
                + (f" (Tags: {tags})" if tags else "")
                + (f" (Examples: {examples})" if examples else "")
            )
        desc_skills = " Skills: " + "; ".join(skills_parts)

    tools, tool_names = load_mcp_tools(abs_dir, config)
    model_adk = format_model_for_adk(config.get("provider", {}), agent_dir)
    agent_name = config.get("name", agent_dir).replace(" ", "_").lower()

    agent = Agent(
        name=agent_name,
        description=f"{config.get('name')} - {desc_skills}",
        instruction=system_prompt,
        model=model_adk,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(max_output_tokens=4096),
    )

    meta: dict[str, Any] = {
        "display_name": config.get("name", agent_name),
        "model": config.get("provider", {}).get("model", "?"),
        "provider": config.get("provider", {}).get("name", "?"),
        "skills": skills_info,
        "tools": tool_names,
    }

    return agent, meta


def discover_agents(base_dir: str) -> list[dict[str, str]]:
    """Auto-discover agent directories under ``agents/``.

    A directory is considered an agent if it contains a ``config.yaml`` file.
    Directories ending in ``.disabled`` or with ``enabled: false`` in their
    config are skipped. The ``orchestrator`` directory is excluded (it is the
    root agent, loaded separately).
    """
    agents: list[dict[str, str]] = []
    agents_dir = os.path.join(base_dir, "agents")
    if not os.path.isdir(agents_dir):
        return agents
    for entry in sorted(os.listdir(agents_dir)):
        full = os.path.join(agents_dir, entry)
        cfg = os.path.join(full, "config.yaml")
        if not os.path.isdir(full) or not os.path.isfile(cfg):
            continue
        if entry.endswith(".disabled"):
            logger.info("  └─ Agent '%s' skipped (.disabled suffix)", entry)
            continue
        if entry == "orchestrator":
            continue
        agent_cfg = load_yaml(cfg)
        if agent_cfg.get("enabled") is False:
            logger.info("  └─ Agent '%s' skipped (enabled: false)", entry)
            continue
        agents.append({"dir": f"agents/{entry}"})
    return agents


def load_orchestrator(orch_dir: str = "agents/orchestrator") -> tuple[Agent, list[Agent], list[dict[str, Any]]]:
    """Load the root orchestrator agent and all its sub-agents.

    Returns a tuple of ``(root_agent, sub_agents, all_metas)`` where
    *all_metas* includes the orchestrator's metadata at index 0.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_dir = os.path.join(project_root, orch_dir)
    config = load_yaml(os.path.join(abs_dir, "config.yaml"))
    prompt_cfg = load_yaml(os.path.join(abs_dir, "prompt.yaml"))

    agents_cfg = config.get("agents")
    if not agents_cfg:
        agents_cfg = discover_agents(project_root)
        if agents_cfg:
            logger.info("Agents auto-discovered: %s", [a["dir"] for a in agents_cfg])

    sub_agents: list[Agent] = []
    agent_metas: list[dict[str, Any]] = []
    provider_pairs: list[tuple[str, dict[str, Any]]] = []

    for info in (agents_cfg or []):
        try:
            sub, meta = load_agent_from_dir(info["dir"])
            sub_agents.append(sub)
            agent_metas.append(meta)
            logger.info("Loaded sub-agent ADK: '%s' (%s)", sub.name, sub.model)
            provider_pairs.append((sub.name, {"name": meta["provider"], "model": meta["model"]}))
        except Exception as e:
            logger.warning("Error loading agent from %s: %s", info["dir"], e)

    # Validate provider configuration for conflicts
    orq_provider = config.get("provider", {})
    provider_pairs.insert(0, ("orchestrator", orq_provider))
    warnings = validate_provider_config(provider_pairs)
    for w in warnings:
        logger.warning("Provider conflict: %s", w)

    model_adk = format_model_for_adk(orq_provider, "orchestrator")
    base_prompt = prompt_cfg.get(
        "system_prompt",
        "You are the main orchestrator. Delegate to specialized sub-agents based on the query.",
    )

    dynamic_prompt = generate_orchestrator_prompt(base_prompt, sub_agents, agent_metas)

    root_agent = Agent(
        name="orchestrator",
        description="Generic orchestrator agent",
        instruction=dynamic_prompt,
        model=model_adk,
        sub_agents=sub_agents,
        generate_content_config=types.GenerateContentConfig(max_output_tokens=4096),
    )

    orq_meta: dict[str, Any] = {
        "display_name": config.get("name", "Orchestrator"),
        "model": orq_provider.get("model", "?"),
        "provider": orq_provider.get("name", "?"),
        "skills": load_skills(abs_dir, config),
        "tools": [],
    }

    all_metas = [orq_meta] + agent_metas

    return root_agent, sub_agents, all_metas
