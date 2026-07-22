# AgentForge

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Google ADK](https://img.shields.io/badge/ADK-2.5-green)](https://github.com/google/adk-python)
[![Bun](https://img.shields.io/badge/Bun-1.2-f9f1e1)](https://bun.sh)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> YAML-driven multi-agent orchestrator on Google ADK with real MCP tools, 11 LLM providers, and a terminal UI that visualizes delegation

Just drop a directory with YAML files and you've got a new agent — the orchestrator discovers it, describes it to the LLM, and routes queries automatically. No Python code, no registration.

## Quick start

```bash
# 1. Install Python deps
pip install -r requirements.txt
pip install mcp_weather_server          # weather MCP (Open-Meteo, no key)
# npx -y duckduckgo-mcp                # news MCP (no key, runs on demand)

# 2. Set your API key
cp .env.example .env
# Edit .env — get a free key at https://openrouter.ai/keys

# 3. Run the TUI
cd tui && bun install && cd .. && bun run tui/index.ts

# Or just the CLI
python orchestrator.py
```

## How it works

```
User query
     │
     ▼
┌──────────────────────────────────┐
│   Orchestrator (generic LLM)     │
│   Prompt auto-generated from     │
│   discovered agents' skills      │
└───────────┬──────────────────────┘
            │ delegates via ADK
   ┌────────┼────────┐
   ▼        ▼        ▼
┌──────┐┌──────┐┌──────┐
│Weather││ News ││Lawyer│
│ MCP  ││ MCP  ││ (LLM)│
└──────┘└──────┘└──────┘
```

The orchestrator's system prompt is **dynamically generated** at startup from the agents found in `agentes/`. It includes each agent's name, skills with tags and examples, and MCP tools — so the LLM knows exactly who to delegate to.

### Delegation flow in the TUI

```
you · 15:38
    What's the weather tomorrow in Palma?

● orchestrator · 15:38
  ┌─ ▾ reasoning ──────────────────────┐
  │ Weather query — delegating to      │
  │ weather agent.                     │
  └───────────────────────────────────┘
  │
  ▼ ● orchestrator → ◆ weather_agent
  ┌─ ▾ reasoning ──────────────────────┐
  │ I'll check tomorrow's forecast...  │
  └───────────────────────────────────┘
  │
  ├─ ✔ get_current_datetime
  │   ⎿ {"timezone_name": "Europe/Madrid"}
  │   ✔ 2026-07-22 15:38 CET
  │
  ├─ ✔ get_weather_byDateTimeRange
  │   ⎿ {"city": "Palma", "start_date": "2026-07-23", ...}
  │
  Tomorrow in Palma: 32°C, sunny, UV 8...
```

## Features

- **No-code agents** — create a directory with `config.yaml`, `prompt.yaml`, and `skills/*.yaml`. The orchestrator auto-discovers it
- **11 LLM providers** — OpenRouter, OpenAI, Anthropic, Gemini, xAI, DeepSeek, Mistral, Groq, Cerebras, OpenCode Zen, Local (Ollama)
- **Real MCP tools** — weather via Open-Meteo, search via DuckDuckGo (both key-less), or any MCP server
- **Central MCP registry** — define servers once in `mcp_servers.yaml`, reference by name in agent configs
- **Dynamic routing prompt** — skills with tags and examples injected into the orchestrator's prompt automatically
- **Terminal UI** — command palette (`Ctrl+P`), slash commands, delegation visualization with tree-style tool calls
- **Automatic retry** — 2 retries on upstream failures with 1.5s delay
- **Skills from external paths** — `skills_paths` in config loads YAML skills from anywhere
- **Multi-turn memory** — sliding window of 20 turns via `model_input_context`

## Adding an agent

```
agentes/agent_recipes/
├── config.yaml
├── prompt.yaml
└── skills/
    └── recipes.yaml
```

**config.yaml**
```yaml
nombre: "Recipe Agent"
provider:
  name: openrouter
  model: "deepseek/deepseek-v4-flash"
  api_key_env: "OPENROUTER_API_KEY"
mcps:
  - spoonacular        # references mcp_servers.yaml
skills_paths:
  - "~/.config/opencode/skills/recipes"
```

**prompt.yaml**
```yaml
system_prompt: |
  You are an expert chef. Propose detailed recipes and culinary advice.
  If the question is not about cooking, respond exactly:
  "ERROR: I cannot answer that. I only know about recipes."
```

**skills/recipes.yaml**
```yaml
id: recipe_search
nombre: Recipe Search
descripcion: Search recipes by ingredients, cuisine type, or occasion
tags:
  - recipes
  - cooking
  - ingredients
ejemplos:
  - "Give me a recipe with chicken and rice"
  - "What can I cook with avocado and quinoa"
```

That's it. The orchestrator discovers it on next run — no registration, no code changes.

> [!TIP]
> Disable an agent by renaming its directory with a `.disabled` suffix, or set `enabled: false` in its `config.yaml`.

## Providers

| Provider | `name` | Example model | Env var |
|---|---|---|---|
| OpenRouter | `openrouter` | `deepseek/deepseek-v4-flash` | `OPENROUTER_API_KEY` |
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| xAI (Grok) | `xai` | `grok-beta` | `XAI_API_KEY` |
| DeepSeek | `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| Mistral AI | `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| Groq | `groq` | `llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| Cerebras | `cerebras` | `llama3.1-8b` | `CEREBRAS_API_KEY` |
| OpenCode Zen | `opencode` | `deepseek-v4-flash-free` | `OPENCODE_ZEN_API_KEY` |
| Local | `local` | `llama3.2` | — (uses `base_url`) |

Local models (Ollama, LM Studio, vLLM):
```yaml
provider:
  name: local
  base_url: "http://localhost:11434/v1"
  model: "llama3.2"
```

## MCP tools

Define once in `mcp_servers.yaml`:
```yaml
weather-mcp:
  type: local
  command: python3
  args: ["-m", "mcp_weather_server"]

openrouter:
  type: remote
  url: https://mcp.openrouter.ai/mcp
```

Reference by name in any agent's `config.yaml`:
```yaml
mcps:
  - weather-mcp
```

## TUI shortcuts

| Key | Action |
|---|---|
| `Enter` | Send query |
| `Ctrl+P` | Command palette |
| `Ctrl+L` | Clear conversation |
| `Ctrl+Y` | Copy selected text |
| `Esc` | Quit |

| Command | Action |
|---|---|
| `/agents` | List connected agents with skills and tools |
| `/help` | Show commands |
| `/clear` | Clear conversation |
| `/exit` | Quit |

## Project structure

```
.
├── orchestrator.py              # ADK runner + dynamic prompt + retry + 11 providers
├── mcp_servers.yaml             # Central MCP registry
├── .env.example                 # All provider API key templates
├── agentes/
│   ├── orquestador/             # Root orchestrator (routing)
│   ├── agente_tiempo/           # Weather (MCP: Open-Meteo)
│   ├── agente_noticias/         # News (MCP: DuckDuckGo)
│   └── agente_abogado/          # Legal (LLM knowledge, no MCP)
└── tui/
    └── index.ts                 # OpenTUI terminal interface
```

## Requirements

- Python 3.10+
- [Bun](https://bun.sh) (for TUI)
- An API key for any supported provider

## License

[MIT](LICENSE)
