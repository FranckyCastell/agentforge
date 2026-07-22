"""Tests for core.mcp_loader — MCP server registry and toolset loading."""

import os
from pathlib import Path

import pytest
import yaml

from core.mcp_loader import load_local_mcp_registry, load_mcp_registry, load_mcp_tools


class TestLoadMcpRegistry:
    def test_loads_central_registry(self, project_root: Path):
        registry = load_mcp_registry()
        assert "weather-mcp" in registry
        assert "duckduckgo-search" in registry

    def test_registry_has_weather_mcp_config(self):
        registry = load_mcp_registry()
        weather = registry["weather-mcp"]
        assert weather["type"] == "local"
        assert weather["command"] == "python3"

    def test_registry_has_duckduckgo_config(self):
        registry = load_mcp_registry()
        ddg = registry["duckduckgo-search"]
        assert ddg["type"] == "local"
        assert ddg["command"] == "npx"

    def test_registry_is_cached(self):
        r1 = load_mcp_registry()
        r2 = load_mcp_registry()
        assert r1 is r2


class TestLoadLocalMcpRegistry:
    def test_no_mcps_dir(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        assert load_local_mcp_registry(str(agent_dir)) == {}

    def test_loads_local_mcp(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        mcps_dir = agent_dir / "mcps"
        mcps_dir.mkdir(parents=True)
        mcp_file = mcps_dir / "custom.yaml"
        mcp_file.write_text(yaml.dump({
            "name": "custom-mcp",
            "type": "local",
            "command": "python3",
            "args": ["-m", "custom_server"],
        }))
        registry = load_local_mcp_registry(str(agent_dir))
        assert "custom-mcp" in registry

    def test_ignores_non_yaml_files(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        mcps_dir = agent_dir / "mcps"
        mcps_dir.mkdir(parents=True)
        (mcps_dir / "README.md").write_text("not yaml")
        (mcps_dir / ".gitkeep").write_text("")
        registry = load_local_mcp_registry(str(agent_dir))
        assert registry == {}


class TestLoadMcpTools:
    def test_no_mcps_returns_empty(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        tools, names = load_mcp_tools(str(agent_dir), {})
        assert tools == []
        assert names == []

    def test_unknown_mcp_ref_skipped(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        config = {"mcps": ["nonexistent-mcp"]}
        tools, names = load_mcp_tools(str(agent_dir), config)
        assert tools == []
        assert names == []

    def test_disabled_mcp_skipped(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        mcps_dir = agent_dir / "mcps"
        mcps_dir.mkdir(parents=True)
        mcp_file = mcps_dir / "disabled.yaml"
        mcp_file.write_text(yaml.dump({
            "name": "disabled-mcp",
            "type": "local",
            "command": "python3",
            "args": ["-m", "fake"],
            "enabled": False,
        }))
        config = {"mcps": ["disabled-mcp"]}
        tools, names = load_mcp_tools(str(agent_dir), config)
        assert tools == []
        assert names == []
