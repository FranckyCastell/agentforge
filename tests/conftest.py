"""Shared pytest fixtures and path setup for AgentForge tests."""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def agents_dir(project_root: Path) -> Path:
    """Return the agents/ directory."""
    return project_root / "agents"


@pytest.fixture
def tmp_agent_dir(tmp_path: Path) -> Path:
    """Create a minimal agent directory structure in a temp location."""
    agent_dir = tmp_path / "test_agent"
    (agent_dir / "skills").mkdir(parents=True)
    (agent_dir / "mcps").mkdir(parents=True)
    return agent_dir
