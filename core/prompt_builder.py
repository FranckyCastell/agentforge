"""Dynamic orchestrator system-prompt generation."""

from __future__ import annotations

from typing import Any

from google.adk import Agent

from .skills import format_skills_compact


def generate_orchestrator_prompt(
    base_prompt: str,
    sub_agents: list[Agent],
    agent_metas: list[dict[str, Any]],
) -> str:
    """Build the orchestrator's system prompt by injecting sub-agent metadata.

    The base prompt from ``prompt.yaml`` is augmented with a structured
    listing of every sub-agent's name, description, tools, and skills so the
    orchestrator LLM can make informed routing decisions.
    """
    if not sub_agents:
        return base_prompt

    agent_sections: list[str] = []
    for agent, meta in zip(sub_agents, agent_metas):
        name = agent.name
        desc = meta.get("name", name)
        skills_text = format_skills_compact(meta.get("skills", []))

        section = f"### {name}\n"
        section += f"Description: {desc}\n"
        if meta.get("tools"):
            section += f"Tools: {', '.join(meta['tools'])}\n"
        if skills_text:
            section += f"Skills:\n{skills_text}\n"
        agent_sections.append(section)

    agents_block = "\n".join(agent_sections)

    dynamic_prompt = f"""{base_prompt}

## Specialized Agents

{agents_block}

## Delegation Rules

IMPORTANT: Analyze the user's query and decide which agent to delegate to based on their skills and tags.

- You can only delegate to one agent per query
- If the query matches an agent, delegate to THAT agent (not another)
- If the query is general knowledge, definitions, explanations, or creativity, respond directly
- If anyone could answer the query, prioritize responding yourself to avoid latency
- When multiple agents could handle the query, choose the one whose skills most precisely match

## Memory

You have conversation memory. The user may give short answers that complete previous queries. Use context to interpret ambiguous responses.

## Guidelines

- Be useful and direct in your responses
- Don't ask what the user wants to do if context makes it clear
- When delegating, don't repeat the sub-agent's response
- If no agent matches, respond directly with your general knowledge
"""

    return dynamic_prompt
