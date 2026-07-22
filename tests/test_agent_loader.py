"""Tests for core.agent_loader — agent discovery and loading from YAML."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from core.agent_loader import discover_agents, load_orchestrator


class TestDiscoverAgents:
    def test_discovers_all_enabled_agents(self, project_root: Path):
        agents = discover_agents(str(project_root))
        agent_dirs = [a["dir"] for a in agents]
        assert "agents/weather_agent" in agent_dirs
        assert "agents/news_agent" in agent_dirs
        assert "agents/lawyer_agent" in agent_dirs
        assert "agents/code_agent" in agent_dirs

    def test_excludes_orchestrator(self, project_root: Path):
        agents = discover_agents(str(project_root))
        dirs = [a["dir"] for a in agents]
        assert "agents/orchestrator" not in dirs

    def test_excludes_disabled_dirs(self, project_root: Path, tmp_path: Path):
        disabled_dir = project_root / "agents" / "test_disabled.disabled"
        disabled_dir.mkdir(exist_ok=True)
        (disabled_dir / "config.yaml").write_text("name: Disabled\n")
        try:
            agents = discover_agents(str(project_root))
            dirs = [a["dir"] for a in agents]
            assert "agents/test_disabled.disabled" not in dirs
        finally:
            disabled_dir.rmdir() if not disabled_dir.exists() else None
            if disabled_dir.exists():
                import shutil
                shutil.rmtree(disabled_dir)

    def test_nonexistent_agents_dir(self, tmp_path: Path):
        agents = discover_agents(str(tmp_path))
        assert agents == []

    def test_dir_without_config_yaml_skipped(self, project_root: Path):
        dummy_dir = project_root / "agents" / "no_config"
        dummy_dir.mkdir(exist_ok=True)
        try:
            agents = discover_agents(str(project_root))
            dirs = [a["dir"] for a in agents]
            assert "agents/no_config" not in dirs
        finally:
            if dummy_dir.exists():
                dummy_dir.rmdir()

    def test_enabled_false_skips_agent(self, project_root: Path):
        test_dir = project_root / "agents" / "test_disabled_flag"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "config.yaml").write_text("name: TestDisabled\nenabled: false\n")
        try:
            agents = discover_agents(str(project_root))
            dirs = [a["dir"] for a in agents]
            assert "agents/test_disabled_flag" not in dirs
        finally:
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)


class TestLoadOrchestrator:
    def test_returns_three_elements(self, project_root: Path):
        result = load_orchestrator()
        assert len(result) == 3

    def test_root_agent_named_orchestrator(self):
        root_agent, _, _ = load_orchestrator()
        assert root_agent.name == "orchestrator"

    def test_sub_agents_loaded(self):
        root_agent, sub_agents, _ = load_orchestrator()
        assert len(sub_agents) >= 3

    def test_sub_agent_names(self):
        _, sub_agents, _ = load_orchestrator()
        names = [a.name for a in sub_agents]
        assert "weather_agent" in names
        assert "news_agent" in names
        assert "lawyer_agent" in names

    def test_metas_include_orchestrator_first(self):
        _, _, all_metas = load_orchestrator()
        assert len(all_metas) >= 4
        assert all_metas[0]["display_name"] == "Orchestrator"

    def test_meta_has_expected_fields(self):
        _, _, all_metas = load_orchestrator()
        for meta in all_metas:
            assert "display_name" in meta
            assert "model" in meta
            assert "provider" in meta
            assert "skills" in meta
            assert "tools" in meta

    def test_orchestrator_has_instruction(self):
        root_agent, _, _ = load_orchestrator()
        assert root_agent.instruction is not None
        assert len(root_agent.instruction) > 100
