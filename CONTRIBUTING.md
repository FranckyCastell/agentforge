# Contributing to AgentForge

Thanks for your interest in contributing! This guide covers the basics.

## Development Setup

```bash
git clone https://github.com/FranckyCastell/agentforge.git
cd agentforge
pip install -e ".[dev,weather]"
cp .env.example .env  # fill in your API keys
```

## Project Layout

```
core/           # Modular Python package (providers, loaders, runner, errors)
agents/         # YAML-defined agents (config.yaml + prompt.yaml + skills/)
tests/          # pytest suite (mirrors core/ module structure)
tui/            # TypeScript terminal UI (Bun + OpenTUI)
```

## Adding an Agent

1. Create `agents/my_agent/` with `config.yaml`, `prompt.yaml`, and `skills/`.
2. The orchestrator auto-discovers it at startup — no registration needed.
3. Add a test in `tests/test_agent_loader.py` if you change discovery logic.

## Running Tests

```bash
pytest
```

## Code Style

- Python 3.10+ — use type hints and docstrings on all public functions.
- No comments unless the code is genuinely non-obvious.
- Keep `orchestrator.py` as a thin entry point — all logic lives in `core/`.
- YAML configs use 2-space indentation.

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make sure `pytest` passes.
3. Keep commits focused — one logical change per commit.
4. Write a clear commit message (imperative mood, e.g. "Add code agent with Groq provider").

## Reporting Issues

Use [GitHub Issues](https://github.com/FranckyCastell/agentforge/issues).
Include the steps to reproduce, expected vs. actual behavior, and your Python version.
