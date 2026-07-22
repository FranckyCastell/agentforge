"""Tests for core.skills — skill loading and formatting."""

from pathlib import Path

import yaml

from core.skills import format_skills_compact, load_skills


class TestFormatSkillsCompact:
    def test_empty_list(self):
        assert format_skills_compact([]) == ""

    def test_single_skill(self):
        skills = [{"id": "weather", "description": "Get weather forecasts"}]
        result = format_skills_compact(skills)
        assert "weather" in result
        assert "Get weather forecasts" in result

    def test_skill_with_tags(self):
        skills = [{
            "id": "legal",
            "description": "Legal advice",
            "tags": ["law", "legal"],
        }]
        result = format_skills_compact(skills)
        assert "law" in result
        assert "legal" in result

    def test_skill_with_examples(self):
        skills = [{
            "id": "news",
            "description": "Search news",
            "examples": ["What's happening in tech?", "Latest news about AI"],
        }]
        result = format_skills_compact(skills)
        assert "What's happening in tech?" in result

    def test_multiple_skills(self):
        skills = [
            {"id": "weather", "description": "Weather forecasts"},
            {"id": "news", "description": "News search"},
            {"id": "legal", "description": "Legal advice"},
        ]
        result = format_skills_compact(skills)
        assert result.count("\n") >= 2
        assert "weather" in result
        assert "news" in result
        assert "legal" in result

    def test_skill_missing_fields(self):
        skills = [{"id": "minimal"}]
        result = format_skills_compact(skills)
        assert "minimal" in result


class TestLoadSkills:
    def test_load_from_skills_dir(self, tmp_agent_dir: Path):
        skill_file = tmp_agent_dir / "skills" / "test.yaml"
        skill_file.write_text(yaml.dump({
            "id": "test_skill",
            "name": "Test Skill",
            "description": "A test skill",
            "tags": ["test"],
            "examples": ["example query"],
        }))
        config = {}
        skills = load_skills(str(tmp_agent_dir), config)
        assert len(skills) == 1
        assert skills[0]["id"] == "test_skill"

    def test_load_multiple_documents(self, tmp_agent_dir: Path):
        skill_file = tmp_agent_dir / "skills" / "multi.yaml"
        skill_file.write_text(
            "---\nid: skill_one\ndescription: First\n"
            "---\nid: skill_two\ndescription: Second\n"
        )
        config = {}
        skills = load_skills(str(tmp_agent_dir), config)
        assert len(skills) == 2
        assert skills[0]["id"] == "skill_one"
        assert skills[1]["id"] == "skill_two"

    def test_no_skills_dir(self, tmp_path: Path):
        agent_dir = tmp_path / "no_skills_agent"
        agent_dir.mkdir()
        config = {}
        skills = load_skills(str(agent_dir), config)
        assert skills == []

    def test_load_from_skills_paths(self, tmp_agent_dir: Path, tmp_path: Path):
        external_dir = tmp_path / "external_skills"
        external_dir.mkdir()
        skill_file = external_dir / "ext.yaml"
        skill_file.write_text(yaml.dump({
            "id": "external_skill",
            "description": "An external skill",
        }))
        config = {"skills_paths": [str(external_dir)]}
        skills = load_skills(str(tmp_agent_dir), config)
        assert len(skills) == 1
        assert skills[0]["id"] == "external_skill"

    def test_load_from_relative_skills_paths(self, tmp_agent_dir: Path, project_root: Path):
        config = {"skills_paths": ["agents/weather_agent/skills"]}
        skills = load_skills(str(tmp_agent_dir), config)
        assert len(skills) >= 1

    def test_nonexistent_skills_path_ignored(self, tmp_agent_dir: Path):
        config = {"skills_paths": ["/nonexistent/path/that/does/not/exist"]}
        skills = load_skills(str(tmp_agent_dir), config)
        assert skills == []
