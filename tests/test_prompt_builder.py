"""Tests for core.prompt_builder — dynamic orchestrator prompt generation."""

from unittest.mock import MagicMock

from core.prompt_builder import generate_orchestrator_prompt


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _make_meta(name: str, tools=None, skills=None) -> dict:
    return {
        "name": name,
        "tools": tools or [],
        "skills": skills or [],
    }


class TestGenerateOrchestratorPrompt:
    def test_no_sub_agents_returns_base(self):
        result = generate_orchestrator_prompt("Base prompt", [], [])
        assert result == "Base prompt"

    def test_single_agent_injected(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent", tools=["weather-mcp"])
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "weather_agent" in result
        assert "Weather Agent" in result
        assert "weather-mcp" in result

    def test_multiple_agents_injected(self):
        agents = [_make_agent("weather_agent"), _make_agent("news_agent")]
        metas = [
            _make_meta("Weather Agent", tools=["weather-mcp"]),
            _make_meta("News Agent", tools=["duckduckgo-search"]),
        ]
        result = generate_orchestrator_prompt("Base", agents, metas)
        assert "weather_agent" in result
        assert "news_agent" in result

    def test_skills_included_in_prompt(self):
        agent = _make_agent("legal_agent")
        meta = _make_meta("Lawyer Agent", skills=[{
            "id": "legal_advice",
            "description": "Provide legal guidance",
            "tags": ["law", "legal"],
            "examples": ["Can I sue my landlord?"],
        }])
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "legal_advice" in result
        assert "Provide legal guidance" in result
        assert "law" in result

    def test_delegation_rules_section_present(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent")
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "## Delegation Rules" in result

    def test_memory_section_present(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent")
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "## Memory" in result

    def test_guidelines_section_present(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent")
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "## Guidelines" in result

    def test_specialized_agents_header(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent")
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "## Specialized Agents" in result

    def test_agent_without_tools_or_skills(self):
        agent = _make_agent("simple_agent")
        meta = _make_meta("Simple Agent")
        result = generate_orchestrator_prompt("Base", [agent], [meta])
        assert "simple_agent" in result
        assert "Simple Agent" in result

    def test_base_prompt_preserved(self):
        agent = _make_agent("weather_agent")
        meta = _make_meta("Weather Agent")
        custom_base = "You are a custom orchestrator with specific rules."
        result = generate_orchestrator_prompt(custom_base, [agent], [meta])
        assert result.startswith(custom_base)
