<div align="center">

# AgentForge

[![Google ADK](https://img.shields.io/badge/ADK-2.5-4285f4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/adk-python)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Bun](https://img.shields.io/badge/Bun-1.2-f9f1e1?style=for-the-badge&logo=bun&logoColor=white)](https://bun.sh)
[![11 Providers](https://img.shields.io/badge/11-LLM_Providers-8b5cf6?style=for-the-badge)](#providers)
[![License](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)](#license)

[Overview](#overview) • [Architecture](#architecture) • [Adding an Agent](#adding-an-agent) • [Providers](#providers) • [MCP Tools](#mcp-tools) • [TUI](#terminal-ui) • [Project Structure](#project-structure) • [Quick Start](#quick-start)

</div>

---

## Overview

**AgentForge** is a YAML-defined, multi-agent orchestrator built on [Google ADK](https://github.com/google/adk-python). Instead of writing Python to register sub-agents, you drop a directory with a few YAML files and the orchestrator discovers it, describes it to the LLM, and routes queries automatically.

It ships with three example agents (weather, news, legal) and supports **11 LLM providers** with per-agent configuration, **real MCP tools** (Open-Meteo, DuckDuckGo — zero API keys required), a **terminal UI** with delegation visualization, and automatic retry on upstream failures.

> [!NOTE]
> The orchestrator's system prompt is **dynamically generated** at startup by scanning `agents/`. Each agent's skills (with tags and example queries) and MCP tools are injected into the prompt so the LLM knows exactly when and where to delegate. No hardcoded routing.

---

## Architecture

```mermaid
flowchart TD
  subgraph User[" "]
    CLI["CLI / --json-events"]
    TUI["OpenTUI Terminal"]
  end

  subgraph Forge["AgentForge"]
    ORQ["Orchestrator<br/>Google ADK InMemoryRunner<br/>Dynamic prompt from agents/"]
    
    subgraph Agents["Discovered Agents"]
      W["Weather Agent<br/>nemotron-3-ultra-free<br/>MCP: Open-Meteo"]
      N["News Agent<br/>north-mini-code-free<br/>MCP: DuckDuckGo"]
      L["Lawyer Agent<br/>deepseek-v4-flash-free<br/>LLM Knowledge"]
    end
  end

  subgraph MCP["MCP Servers"]
    OM["Open-Meteo<br/>8 tools<br/>no API key"]
    DDG["DuckDuckGo<br/>web + news search<br/>no API key"]
    ANY["Any MCP Server<br/>(mcp_servers.yaml)"]
  end

  CLI --> ORQ
  TUI --> ORQ
  ORQ --> W
  ORQ --> N
  ORQ --> L
  W --> OM
  N --> DDG
  L --> ANY
```

The orchestrator uses **Google ADK's native delegation pattern**: a root agent receives all queries, analyzes intent, and delegates to the appropriate sub-agent. Sub-agents can use MCP tools, LLM internal knowledge, or both.

### Delegation flow

```
you · 15:38
    What's the weather tomorrow in Palma?

● orchestrator · 15:38
  ┌─ ▾ delegating to weather_agent ─────────────┐
  │ Weather query — matches weather agent's     │
  │ skills. Delegating.                         │
  └─────────────────────────────────────────────┘
  │
  ◆ weather_agent
  ├─ ✔ get_current_datetime → 2026-07-22 15:38 CET
  └─ ✔ get_weather_byDateTimeRange
          ⎿ city: Palma, forecast: 32°C, sunny
```

---

## Adding an Agent

Create a directory under `agents/` with three files:

```
agents/my_agent/
├── config.yaml
├── prompt.yaml
└── skills/
    └── myskill.yaml
```

**config.yaml** — model, provider, and MCP bindings:
```yaml
name: "My Agent"
provider:
  name: openrouter
  model: "deepseek/deepseek-v4-flash"
  api_key_env: "OPENROUTER_API_KEY"
mcps:
  - weather-mcp
skills_paths:
  - "~/.config/opencode/skills/recipes"
```

**prompt.yaml** — the agent's system prompt:
```yaml
system_prompt: |
  You are an expert chef. Propose detailed recipes and culinary advice.
  If the question is not about cooking, respond exactly:
  "ERROR: I cannot answer that. I only know about recipes."
```

**skills/myskill.yaml** — declarative skill definitions for the dynamic orchestrator prompt:
```yaml
id: recipe_search
name: Recipe Search
description: Search recipes by ingredients, cuisine, or occasion
tags:
  - recipes
  - cooking
examples:
  - "Give me a recipe with chicken and rice"
  - "What can I cook with avocado and quinoa"
```

> [!TIP]
> Disable an agent by renaming its directory with a `.disabled` suffix, or set `enabled: false` in `config.yaml`.

---

## Providers

Every agent can use a different provider. Set `provider.name` and `provider.model` in the agent's `config.yaml`:

| Provider | `name` | Example model | Env var |
| :--- | :--- | :--- | :--- |
| **OpenRouter** | `openrouter` | `deepseek/deepseek-v4-flash` | `OPENROUTER_API_KEY` |
| **OpenAI** | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `google` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| **xAI (Grok)** | `xai` | `grok-beta` | `XAI_API_KEY` |
| **DeepSeek** | `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| **Mistral AI** | `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| **Groq** | `groq` | `llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| **Cerebras** | `cerebras` | `llama3.1-8b` | `CEREBRAS_API_KEY` |
| **OpenCode Zen** | `opencode` | `deepseek-v4-flash-free` | `OPENCODE_ZEN_API_KEY` |
| **Local** | `local` | `llama3.2` | none (uses `base_url`) |

Local models (Ollama, LM Studio, vLLM):
```yaml
provider:
  name: local
  base_url: "http://localhost:11434/v1"
  model: "llama3.2"
```

> [!NOTE]
> Provider environments are isolated per-agent via `.env` files. The system loads `.env` from the project root first, then overrides with per-agent `.env` if present.

---

## MCP Tools

Define MCP servers once in `mcp_servers.yaml`:

```yaml
weather-mcp:
  type: local
  command: python3
  args: ["-m", "mcp_weather_server"]

openrouter-mcp:
  type: remote
  url: https://mcp.openrouter.ai/mcp
```

Reference by name in any agent's `config.yaml`:

```yaml
mcps:
  - weather-mcp
```

### Bundled (key-less) tools

| MCP Server | Tools | API Key |
| :--- | :--- | :--- |
| **Open-Meteo** (`mcp_weather_server`) | 8 tools: current weather, forecast, air quality, timezone, datetime | None |
| **DuckDuckGo** (`duckduckgo-mcp`) | web search, news search | None |

---

## Terminal UI

The TUI (TypeScript + OpenTUI, `bun`) provides a visual interface for the orchestrator with tool tree rendering and delegation visualization.

| Key | Action |
| :--- | :--- |
| `Ctrl+P` | Command palette |
| `Ctrl+L` | Clear conversation |
| `Ctrl+Y` | Copy selected text |
| `Esc` | Quit |

| Slash command | Action |
| :--- | :--- |
| `/agents` | List connected agents with skills and tools |
| `/help` | Show available commands |
| `/clear` | Clear conversation |
| `/exit` | Quit |

---

## Project Structure

```
.
├── orchestrator.py            # ADK runner + dynamic prompt + retry
├── mcp_servers.yaml           # Central MCP server registry
├── .env.example               # All provider API key templates
├── requirements.txt           # Python dependencies
│
├── agents/
│   ├── orchestrator/           # Root orchestrator (routing agent)
│   │   ├── config.yaml
│   │   ├── prompt.yaml
│   │   └── skills/router.yaml
│   ├── weather_agent/         # Weather (MCP: Open-Meteo)
│   │   ├── config.yaml
│   │   ├── prompt.yaml
│   │   └── skills/weather.yaml
│   ├── news_agent/       # News (MCP: DuckDuckGo)
│   │   ├── config.yaml
│   │   ├── prompt.yaml
│   │   └── skills/news.yaml
│   └── lawyer_agent/        # Legal (LLM knowledge, no MCP)
│       ├── config.yaml
│       ├── prompt.yaml
│       └── skills/legal.yaml
│
└── tui/
    ├── index.ts               # OpenTUI terminal interface
    ├── package.json
    └── tsconfig.json
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Bun](https://bun.sh) (for the TUI)
- An API key for any supported provider

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install mcp_weather_server
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set your API key (e.g., OPENROUTER_API_KEY)
```

### 3. Run

**CLI mode:**
```bash
python orchestrator.py
```

**JSON events (for programmatic use):**
```bash
python orchestrator.py --json-events
```

**Terminal UI:**
```bash
cd tui && bun install && cd .. && bun run tui/index.ts
```

---

<div align="center">

Built with [Google ADK](https://github.com/google/adk-python)

</div>
