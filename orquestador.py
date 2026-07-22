import asyncio
import json
import logging
import os
import sys
import uuid
import yaml

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk import Agent
from google.adk.agents import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

MAX_HISTORY_TURNS = 20
MAX_RETRIES = 2
RETRY_DELAY = 1.5
RETRYABLE_PATTERNS = ["upstream", "500", "502", "503", "429", "timeout", "connection"]

PROVIDER_CONFIGS = {
    "opencode":   {"prefix": "openai",     "base": "https://opencode.ai/zen/v1",         "key_env": "OPENCODE_ZEN_API_KEY"},
    "openrouter": {"prefix": "openai",     "base": "https://openrouter.ai/api/v1",       "key_env": "OPENROUTER_API_KEY"},
    "openai":     {"prefix": "openai",     "base": None,                                  "key_env": "OPENAI_API_KEY"},
    "anthropic":  {"prefix": "anthropic",  "base": None,                                  "key_env": "ANTHROPIC_API_KEY"},
    "google":     {"prefix": "gemini",     "base": None,                                  "key_env": "GEMINI_API_KEY"},
    "xai":        {"prefix": "openai",     "base": "https://api.x.ai/v1",                 "key_env": "XAI_API_KEY"},
    "deepseek":   {"prefix": "openai",     "base": "https://api.deepseek.com/v1",         "key_env": "DEEPSEEK_API_KEY"},
    "mistral":    {"prefix": "mistral",    "base": None,                                  "key_env": "MISTRAL_API_KEY"},
    "groq":       {"prefix": "groq",       "base": None,                                  "key_env": "GROQ_API_KEY"},
    "cerebras":   {"prefix": "cerebras",   "base": None,                                  "key_env": "CEREBRAS_API_KEY"},
    "local":      {"prefix": "openai",     "base": None,                                  "key_env": None},
}

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("orquestador_adk")

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

INTERNAL_TOOLS = {"transfer_to_agent", "transfer_agent"}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_api_key(prov: dict) -> str:
    if "api_key" in prov and prov["api_key"]:
        return prov["api_key"]
    val = prov.get("api_key_env", "")
    if val and val in os.environ:
        return os.environ[val]
    if val and (val.startswith("sk-") or val.startswith("sk_") or val.startswith("gsk_") or val.startswith("csk_") or val.startswith("AIza")):
        return val
    return ""


def _format_model_for_adk(prov: dict) -> str:
    provider_name = prov.get("name", "opencode")
    model_name = prov.get("model", "deepseek-v4-flash-free")
    base_url = prov.get("base_url")

    pconf = PROVIDER_CONFIGS.get(provider_name, PROVIDER_CONFIGS["opencode"])
    prefix = pconf["prefix"]

    if provider_name == "local" and base_url:
        os.environ["OPENAI_API_BASE"] = base_url
        api_key = _resolve_api_key(prov) or "ollama"
        os.environ["OPENAI_API_KEY"] = api_key
        return f"openai/{model_name}"

    api_key = _resolve_api_key(prov)
    if not api_key:
        key_env = pconf.get("key_env")
        if key_env:
            api_key = os.environ.get(key_env, "")

    if prefix == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if pconf.get("base"):
            os.environ["OPENAI_API_BASE"] = pconf["base"]
        elif base_url:
            os.environ["OPENAI_API_BASE"] = base_url
        return f"openai/{model_name}"
    elif prefix == "anthropic":
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        return f"anthropic/{model_name}"
    elif prefix == "gemini":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        return model_name
    else:
        if api_key:
            env_key = f"{prefix.upper()}_API_KEY"
            os.environ[env_key] = api_key
        return f"{prefix}/{model_name}"


_mcp_registry: dict | None = None


def _load_mcp_registry() -> dict:
    global _mcp_registry
    if _mcp_registry is not None:
        return _mcp_registry
    project_root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(project_root, "mcp_servers.yaml")
    if os.path.exists(path):
        _mcp_registry = _load_yaml(path) or {}
    else:
        _mcp_registry = {}
    return _mcp_registry


def _build_mcp_toolset(mcp_def: dict) -> object | None:
    from google.adk.tools import McpToolset
    from mcp.client.stdio import StdioServerParameters

    mcp_type = mcp_def.get("type", "local")
    if mcp_type == "remote":
        url = mcp_def.get("url", "")
        if not url:
            logger.warning("MCP remoto sin url: %s", mcp_def)
            return None
        try:
            return McpToolset(url=url)
        except Exception as e:
            logger.warning("Error cargando MCP remoto %s: %s", url, e)
            return None

    command = mcp_def.get("command", "")
    if isinstance(command, list):
        command = command[0] if command else ""
        args = mcp_def.get("command", [])[1:] if len(mcp_def.get("command", [])) > 1 else mcp_def.get("args", [])
    else:
        args = mcp_def.get("args", [])

    env = mcp_def.get("env") or mcp_def.get("environment") or None
    if env:
        resolved_env = {}
        for k, v in env.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                resolved_env[k] = os.environ.get(env_key, v)
            else:
                resolved_env[k] = v
        env = resolved_env

    try:
        params = StdioServerParameters(command=command, args=args, env=env)
        return McpToolset(connection_params=params)
    except Exception as e:
        logger.warning("Error cargando MCP local %s: %s", command, e)
        return None


def _cargar_tools_mcp(abs_dir: str, config: dict) -> tuple[list, list[str]]:
    tools = []
    tool_names = []

    mcp_refs = config.get("mcps", [])
    if mcp_refs:
        registry = _load_mcp_registry()
        for ref in mcp_refs:
            if ref in registry:
                mcp_def = registry[ref]
                if mcp_def.get("enabled") is False:
                    continue
                toolset = _build_mcp_toolset(mcp_def)
                if toolset:
                    tools.append(toolset)
                    tool_names.append(ref)
                    logger.info("  └─ MCP '%s' cargado desde registry central", ref)
            else:
                logger.warning("  └─ MCP '%s' no encontrado en mcp_servers.yaml", ref)

    mcps_dir = os.path.join(abs_dir, "mcps")
    if os.path.isdir(mcps_dir):
        from google.adk.tools import McpToolset
        from mcp.client.stdio import StdioServerParameters

        for fname in sorted(os.listdir(mcps_dir)):
            if fname.endswith((".yaml", ".yml")):
                mcp_cfg = _load_yaml(os.path.join(mcps_dir, fname))
                name = mcp_cfg.get("name", fname)
                toolset = _build_mcp_toolset(mcp_cfg)
                if toolset:
                    tools.append(toolset)
                    tool_names.append(name)
                    logger.info("  └─ MCP '%s' cargado desde mcps/%s", name, fname)

    return tools, tool_names


def _cargar_skills(abs_dir: str, config: dict) -> list[dict]:
    skills_info = []

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
            sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), sp)
        if os.path.isdir(sp):
            for fname in sorted(os.listdir(sp)):
                if fname.endswith((".yaml", ".yml")):
                    fpath = os.path.join(sp, fname)
                    with open(fpath) as f:
                        for doc in yaml.safe_load_all(f):
                            if doc:
                                skills_info.append(doc)

    return skills_info


def _format_skills_compact(skills: list[dict]) -> str:
    if not skills:
        return ""
    parts = []
    for s in skills:
        sid = s.get("id", "")
        desc = s.get("descripcion", "").strip()
        tags = s.get("tags", [])
        ejemplos = s.get("ejemplos", [])[:2]
        line = f"  - {sid}: {desc}"
        if tags:
            line += f" [tags: {', '.join(tags)}]"
        if ejemplos:
            line += f" [ej: {' / '.join(ejemplos)}]"
        parts.append(line)
    return "\n".join(parts)


def _generate_orchestrator_prompt(base_prompt: str, sub_agents: list[Agent], agent_metas: list[dict]) -> str:
    if not sub_agents:
        return base_prompt

    agent_sections = []
    for agent, meta in zip(sub_agents, agent_metas):
        name = agent.name
        desc = meta.get("nombre", name)
        skills_text = _format_skills_compact(meta.get("skills", []))

        section = f"### {name}\n"
        section += f"Descripcion: {desc}\n"
        if meta.get("tools"):
            section += f"Herramientas: {', '.join(meta['tools'])}\n"
        if skills_text:
            section += f"Skills:\n{skills_text}\n"
        agent_sections.append(section)

    agents_block = "\n".join(agent_sections)

    dynamic_prompt = f"""{base_prompt}

## Agentes especializados

{agents_block}

## Reglas de delegacion

IMPORTANTE: Analiza la consulta del usuario y decide a que agente delegar segun sus skills y tags.

- Solo puedes delegar a un agente por consulta
- Si la consulta encaja con un agente, delega a ESE agente (no a otro)
- Si la consulta es cultura general, definiciones, explicaciones o creatividad, responde tu directamente
- Si la consulta podria responderla cualquiera, prioriza responder tu para evitar latencia

## Memoria

Tienes memoria de conversacion. El usuario puede dar respuestas cortas que completan consultas anteriores. Usa el contexto para interpretar respuestas ambiguas.

## Directrices

- Se util y directo en tus respuestas
- No preguntes al usuario que quiere hacer si el contexto lo deja claro
- Cuando delegues, no repitas la respuesta del subagente
"""

    return dynamic_prompt


def cargar_agente_desde_dir(agent_dir: str) -> tuple[Agent, dict]:
    abs_dir = os.path.join(os.path.dirname(__file__), agent_dir)
    env_file = os.path.join(abs_dir, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    cfg_path = os.path.join(abs_dir, "config.yaml")
    prompt_path = os.path.join(abs_dir, "prompt.yaml")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No existe config.yaml en {abs_dir}")

    config = _load_yaml(cfg_path)
    prompt_cfg = _load_yaml(prompt_path) if os.path.exists(prompt_path) else {}
    system_prompt = prompt_cfg.get("system_prompt", "")

    skills_info = _cargar_skills(abs_dir, config)

    desc_skills = ""
    if skills_info:
        skills_parts = []
        for s in skills_info:
            sid = s.get("id", "")
            desc = s.get("descripcion", "")
            tags = ", ".join(s.get("tags", []))
            ejemplos = ", ".join(s.get("ejemplos", [])[:2])
            skills_parts.append(
                f"[{sid}] {desc}"
                + (f" (Tags: {tags})" if tags else "")
                + (f" (Ejemplos: {ejemplos})" if ejemplos else "")
            )
        desc_skills = " Skills: " + "; ".join(skills_parts)

    tools, tool_names = _cargar_tools_mcp(abs_dir, config)
    model_adk = _format_model_for_adk(config.get("provider", {}))
    agent_name = config.get("nombre", agent_dir).replace(" ", "_").lower()

    agent = Agent(
        name=agent_name,
        description=f"{config.get('nombre')} - {desc_skills}",
        instruction=system_prompt,
        model=model_adk,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(max_output_tokens=4096),
    )

    meta = {
        "nombre": config.get("nombre", agent_name),
        "model": config.get("provider", {}).get("model", "?"),
        "provider": config.get("provider", {}).get("name", "?"),
        "skills": skills_info,
        "tools": tool_names,
    }

    return agent, meta


def _descubrir_agentes(base_dir: str) -> list[dict]:
    agentes = []
    agentes_dir = os.path.join(base_dir, "agentes")
    if not os.path.isdir(agentes_dir):
        return agentes
    for entry in sorted(os.listdir(agentes_dir)):
        full = os.path.join(agentes_dir, entry)
        cfg = os.path.join(full, "config.yaml")
        if not os.path.isdir(full) or not os.path.isfile(cfg):
            continue
        if entry.endswith(".disabled"):
            logger.info("  └─ Agente '%s' omitido (sufijo .disabled)", entry)
            continue
        if entry == "orquestador":
            continue
        agent_cfg = _load_yaml(cfg)
        if agent_cfg.get("enabled") is False:
            logger.info("  └─ Agente '%s' omitido (enabled: false)", entry)
            continue
        agentes.append({"dir": f"agentes/{entry}"})
    return agentes


def cargar_orquestador(orq_dir: str = "agentes/orquestador") -> tuple[Agent, list[Agent], list[dict]]:
    project_root = os.path.dirname(os.path.abspath(__file__))
    abs_dir = os.path.join(project_root, orq_dir)
    config = _load_yaml(os.path.join(abs_dir, "config.yaml"))
    prompt_cfg = _load_yaml(os.path.join(abs_dir, "prompt.yaml"))

    agentes_cfg = config.get("agentes")
    if not agentes_cfg:
        agentes_cfg = _descubrir_agentes(project_root)
        if agentes_cfg:
            logger.info("Agentes descubiertos automaticamente: %s", [a["dir"] for a in agentes_cfg])

    sub_agents = []
    agent_metas = []
    for info in (agentes_cfg or []):
        try:
            sub, meta = cargar_agente_desde_dir(info["dir"])
            sub_agents.append(sub)
            agent_metas.append(meta)
            logger.info("Cargado subagente ADK: '%s' (%s)", sub.name, sub.model)
        except Exception as e:
            logger.warning("Error cargando agente desde %s: %s", info["dir"], e)

    model_adk = _format_model_for_adk(config.get("provider", {}))
    base_prompt = prompt_cfg.get(
        "system_prompt",
        "Eres el orquestador principal. Delega en los subagentes especializados segun la consulta.",
    )

    dynamic_prompt = _generate_orchestrator_prompt(base_prompt, sub_agents, agent_metas)

    root_agent = Agent(
        name="orquestador",
        description="Agente orquestador generico",
        instruction=dynamic_prompt,
        model=model_adk,
        sub_agents=sub_agents,
        generate_content_config=types.GenerateContentConfig(max_output_tokens=4096),
    )

    orq_meta = {
        "nombre": config.get("nombre", "Orquestador"),
        "model": config.get("provider", {}).get("model", "?"),
        "provider": config.get("provider", {}).get("name", "?"),
        "skills": _cargar_skills(abs_dir, config),
        "tools": [],
    }

    all_metas = [orq_meta] + agent_metas

    return root_agent, sub_agents, all_metas


async def main_async():
    root_agent, sub_agents, _ = cargar_orquestador()
    runner = InMemoryRunner(agent=root_agent)

    user_id = "cli_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        user_id=user_id, session_id=session_id, app_name=runner.app_name
    )

    print("=" * 60)
    print(f"  Google ADK Multi-Agent Orchestrator")
    if sub_agents:
        print(f"  Subagentes conectados: {', '.join(a.name for a in sub_agents)}")
    print("  Escribe 'salir' para terminar")
    print("=" * 60)
    print()

    history_contents: list[types.Content] = []
    history_turn_count = 0

    while True:
        try:
            texto = input("  Tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not texto:
            continue
        if texto.lower() in ("salir", "exit", "quit"):
            break

        user_msg = types.Content(role="user", parts=[types.Part.from_text(text=texto)])
        turn_assistant_contents: list[types.Content] = []

        run_config = None
        if history_contents:
            run_config = RunConfig(model_input_context=list(history_contents))

        try:
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=user_msg,
                run_config=run_config,
            ):
                if getattr(event, "content", None):
                    author = getattr(event, "author", "orquestador")
                    for part in event.content.parts:
                        if part.text:
                            print(f"  {author} > {part.text}")
                    if event.content.role != 'user':
                        turn_assistant_contents.append(event.content)
        except Exception as e:
            logger.error("Error en ejecucion de ADK runner: %s", e)

        history_contents.append(user_msg)
        history_contents.extend(turn_assistant_contents)
        history_turn_count += 1

        while history_turn_count > MAX_HISTORY_TURNS and history_contents:
            history_contents.pop(0)
            while history_contents and history_contents[0].role != 'user':
                history_contents.pop(0)
            history_turn_count -= 1

        print()


def _clean_error_msg(msg: str) -> str:
    if "BadRequestError" in msg and "Upstream request failed" in msg:
        return "El proveedor del modelo no esta disponible. Intenta de nuevo."
    if "RateLimitError" in msg:
        return "Limite de peticiones alcanzado. Espera unos segundos."
    if "AuthenticationError" in msg:
        return "Error de autenticacion. Revisa tu API key en .env"
    if "Timeout" in msg:
        return "Tiempo de espera agotado. Intenta de nuevo."
    return msg


def _is_retryable(err_str: str) -> bool:
    err_lower = err_str.lower()
    return any(p in err_lower for p in RETRYABLE_PATTERNS)


def _json_event(**kwargs):
    sys.stdout.write(json.dumps(kwargs, default=str) + "\n")
    sys.stdout.flush()


async def main_json_events_async():
    root_agent, sub_agents, all_metas = cargar_orquestador()

    agent_list = []
    for i, meta in enumerate(all_metas):
        is_orq = i == 0
        agent_name = "orquestador" if is_orq else sub_agents[i - 1].name if sub_agents else "orquestador"
        skills_summary = []
        for s in meta.get("skills", []):
            skills_summary.append({
                "id": s.get("id", ""),
                "nombre": s.get("nombre", ""),
                "descripcion": s.get("descripcion", "").strip(),
                "tags": s.get("tags", []),
                "ejemplos": s.get("ejemplos", [])[:3],
            })
        agent_list.append({
            "name": agent_name,
            "nombre": meta.get("nombre", agent_name),
            "model": meta.get("model", "?"),
            "provider": meta.get("provider", "?"),
            "is_orchestrator": is_orq,
            "skills": skills_summary,
            "tools": meta.get("tools", []),
        })

    _json_event(type="init", agents=agent_list)

    runner = InMemoryRunner(agent=root_agent)
    user_id = "cli_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        user_id=user_id, session_id=session_id, app_name=runner.app_name
    )

    history_contents: list[types.Content] = []
    history_turn_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _json_event(type="error", text="Invalid JSON input")
            continue

        query = req.get("query", "")
        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit"):
            break

        user_msg = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        prev_author = "orquestador"
        turn_assistant_contents: list[types.Content] = []

        run_config = None
        if history_contents:
            run_config = RunConfig(model_input_context=list(history_contents))

        success = False
        for attempt in range(MAX_RETRIES + 1):
            prev_author = "orquestador"
            turn_assistant_contents = []
            try:
                async for event in runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=user_msg,
                    run_config=run_config,
                ):
                    if not getattr(event, "content", None):
                        continue
                    author = getattr(event, "author", "orquestador")
                    if author != prev_author:
                        _json_event(type="delegate", from_agent=prev_author, to_agent=author)
                        prev_author = author

                    for part in event.content.parts:
                        if part.text:
                            _json_event(type="response", agent=author, text=part.text)
                        if getattr(part, "function_call", None):
                            fc = part.function_call
                            if fc.name in INTERNAL_TOOLS:
                                continue
                            _json_event(type="tool_call", agent=author, tool=fc.name, args=fc.args)
                        if getattr(part, "function_response", None):
                            fr = part.function_response
                            if fr.name in INTERNAL_TOOLS:
                                continue
                            _json_event(type="tool_result", agent=author, tool=fr.name, result=fr.response)

                    if event.content.role != 'user':
                        turn_assistant_contents.append(event.content)

                _json_event(type="done")
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES and _is_retryable(err_str):
                    _json_event(type="retry", attempt=attempt + 1, max_retries=MAX_RETRIES)
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                _json_event(type="error", text=_clean_error_msg(err_str))
                _json_event(type="done")
                success = True
                break

        if not success:
            pass

        history_contents.append(user_msg)
        history_contents.extend(turn_assistant_contents)
        history_turn_count += 1

        while history_turn_count > MAX_HISTORY_TURNS and history_contents:
            history_contents.pop(0)
            while history_contents and history_contents[0].role != 'user':
                history_contents.pop(0)
            history_turn_count -= 1


def main():
    if "--json-events" in sys.argv:
        logging.disable(logging.CRITICAL)
        asyncio.run(main_json_events_async())
    else:
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
