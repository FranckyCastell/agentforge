"""AgentForge runner: CLI and JSON-events modes with retry and history management."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from google.adk.agents import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent_loader import INTERNAL_TOOLS, load_orchestrator
from .errors import RETRY_DELAY, RETRYABLE_PATTERNS, clean_error_msg, is_retryable

logger = logging.getLogger("orchestrator_adk")

MAX_HISTORY_TURNS: int = 20
MAX_RETRIES: int = 2


def _trim_history(
    history_contents: list[types.Content],
    history_turn_count: int,
) -> tuple[list[types.Content], int]:
    """Evict the oldest conversation turn once the history exceeds the limit."""
    while history_turn_count > MAX_HISTORY_TURNS and history_contents:
        history_contents.pop(0)
        while history_contents and history_contents[0].role != "user":
            history_contents.pop(0)
        history_turn_count -= 1
    return history_contents, history_turn_count


async def main_async() -> None:
    """Run the orchestrator in interactive CLI mode."""
    root_agent, sub_agents, _ = load_orchestrator()
    runner = InMemoryRunner(agent=root_agent)

    user_id = "cli_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        user_id=user_id, session_id=session_id, app_name=runner.app_name
    )

    print("=" * 60)
    print("  Google ADK Multi-Agent Orchestrator")
    if sub_agents:
        print(f"  Connected sub-agents: {', '.join(a.name for a in sub_agents)}")
    print("  Type 'exit' to quit")
    print("=" * 60)
    print()

    history_contents: list[types.Content] = []
    history_turn_count = 0

    while True:
        try:
            text = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in (("exit", "quit")):
            break

        user_msg = types.Content(role="user", parts=[types.Part.from_text(text=text)])
        turn_assistant_contents: list[types.Content] = []

        run_config = None
        if history_contents:
            run_config = RunConfig(model_input_context=list(history_contents))

        success = False
        for attempt in range(MAX_RETRIES + 1):
            turn_assistant_contents = []
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=user_msg,
                    run_config=run_config,
                ):
                    if getattr(event, "content", None):
                        author = getattr(event, "author", "orchestrator")
                        for part in event.content.parts:
                            if part.text:
                                print(f"  {author} > {part.text}")
                        if event.content.role != "user":
                            turn_assistant_contents.append(event.content)
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES and is_retryable(err_str):
                    logger.warning("Retry %d/%d after error: %s", attempt + 1, MAX_RETRIES, clean_error_msg(err_str))
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                logger.error("Error executing ADK runner: %s", clean_error_msg(err_str))
                print(f"  [error] {clean_error_msg(err_str)}")
                success = True
                break

        history_contents.append(user_msg)
        history_contents.extend(turn_assistant_contents)
        history_turn_count += 1
        history_contents, history_turn_count = _trim_history(history_contents, history_turn_count)

        print()


def _json_event(**kwargs: Any) -> None:
    """Write a single JSON event to stdout."""
    sys.stdout.write(json.dumps(kwargs, default=str) + "\n")
    sys.stdout.flush()


async def main_json_events_async() -> None:
    """Run the orchestrator in JSON-events mode (stdin/stdout protocol)."""
    root_agent, sub_agents, all_metas = load_orchestrator()

    agent_list: list[dict[str, Any]] = []
    for i, meta in enumerate(all_metas):
        is_orq = i == 0
        agent_name = "orchestrator" if is_orq else sub_agents[i - 1].name if sub_agents else "orchestrator"
        skills_summary = [
            {
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "description": s.get("description", "").strip(),
                "tags": s.get("tags", []),
                "examples": s.get("examples", [])[:3],
            }
            for s in meta.get("skills", [])
        ]
        agent_list.append({
            "name": agent_name,
            "display_name": meta.get("name", agent_name),
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
        if query.lower() in (("exit", "quit")):
            break

        user_msg = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        prev_author = "orchestrator"
        turn_assistant_contents: list[types.Content] = []

        run_config = None
        if history_contents:
            run_config = RunConfig(model_input_context=list(history_contents))

        success = False
        for attempt in range(MAX_RETRIES + 1):
            prev_author = "orchestrator"
            turn_assistant_contents = []
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=user_msg,
                    run_config=run_config,
                ):
                    if not getattr(event, "content", None):
                        continue
                    author = getattr(event, "author", "orchestrator")
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

                    if event.content.role != "user":
                        turn_assistant_contents.append(event.content)

                _json_event(type="done")
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES and is_retryable(err_str):
                    _json_event(type="retry", attempt=attempt + 1, max_retries=MAX_RETRIES)
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                _json_event(type="error", text=clean_error_msg(err_str))
                _json_event(type="done")
                success = True
                break

        history_contents.append(user_msg)
        history_contents.extend(turn_assistant_contents)
        history_turn_count += 1
        history_contents, history_turn_count = _trim_history(history_contents, history_turn_count)


def main() -> None:
    """Entry point: dispatch to CLI or JSON-events mode based on argv."""
    if "--json-events" in sys.argv:
        logging.disable(logging.CRITICAL)
        asyncio.run(main_json_events_async())
    else:
        asyncio.run(main_async())
