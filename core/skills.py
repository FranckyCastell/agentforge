"""Declarative skill loading and formatting for the dynamic orchestrator prompt."""

from __future__ import annotations

import os
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load a single YAML document from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_skills(abs_dir: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load skill definitions for an agent.

    Skills are discovered from two sources:
    1. YAML files in the agent's ``skills/`` directory.
    2. YAML files in external paths declared via ``skills_paths`` in config.

    Each YAML file may contain multiple documents separated by ``---``.
    """
    skills_info: list[dict[str, Any]] = []

    skills_dir = os.path.join(abs_dir, "skills")
    if os.path.isdir(skills_dir):
        for fname in sorted(os.listdir(skills_dir)):
            if fname.endswith((".yaml", ".yml")):
                fpath = os.path.join(skills_dir, fname)
                with open(fpath) as f:
                    for doc in yaml.safe_load_all(f):
                        if doc:
                            skills_info.append(doc)

    skills_paths = config.get("skills_paths", [])
    for sp in skills_paths:
        sp = os.path.expanduser(sp)
        if not os.path.isabs(sp):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sp = os.path.join(project_root, sp)
        if os.path.isdir(sp):
            for fname in sorted(os.listdir(sp)):
                if fname.endswith((".yaml", ".yml")):
                    fpath = os.path.join(sp, fname)
                    with open(fpath) as f:
                        for doc in yaml.safe_load_all(f):
                            if doc:
                                skills_info.append(doc)

    return skills_info


def format_skills_compact(skills: list[dict[str, Any]]) -> str:
    """Format skill definitions into a compact multi-line string for the orchestrator prompt."""
    if not skills:
        return ""
    parts: list[str] = []
    for s in skills:
        sid = s.get("id", "")
        desc = s.get("description", "").strip()
        tags = s.get("tags", [])
        examples = s.get("examples", [])[:2]
        line = f"  - {sid}: {desc}"
        if tags:
            line += f" [tags: {', '.join(tags)}]"
        if examples:
            line += f" [ej: {' / '.join(examples)}]"
        parts.append(line)
    return "\n".join(parts)
