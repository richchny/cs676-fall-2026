"""
tools.py: what the model is allowed to ask this app to do.

Two halves:
  TOOLS         the descriptions Claude reads to decide which tool to call.
  execute_tool  the code that runs a tool when Claude asks for it.

Claude never runs anything itself. It only asks; this module does the work, by
calling the same functions the regex router used to call.

Every tool returns a plain string. Errors are returned as strings too (never
raised), so Claude can read them and recover, for example by asking the user
which of two matching personas they meant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM
from .persona import find_personas, generate_persona, list_personas, save_persona

DEFAULT_TURNS = 6
MAX_TURNS = 20  # each turn is a billed API call, so cap what the model can request

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "create_persona",
        "description": (
            "Create a new character and save it as a persona file. Use this when the user "
            "wants a new person, e.g. 'a skeptical data scientist named Ada'. Call it once "
            "per character."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What kind of person to create, in the user's own words.",
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every persona that currently exists, with name, role, and summary.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_conversation",
        "description": (
            "Have two or more existing personas talk to each other. Every participant must "
            "already exist; create missing ones first. If a name is ambiguous, ask the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of the personas who will talk, at least two.",
                },
                "topic": {"type": "string", "description": "What they talk about."},
                "turns": {
                    "type": "integer",
                    "description": f"Total lines of dialogue. Default {DEFAULT_TURNS}, maximum {MAX_TURNS}.",
                },
            },
            "required": ["participants", "topic"],
        },
    },
]


def execute_tool(
    name: str,
    args: Dict[str, Any],
    llm: LLM,
    persona_dir: Path,
    on_turn: Optional[Callable[[Turn], None]] = None,
    on_start: Optional[Callable[[str], None]] = None,
) -> str:
    """Run one tool by name and return its result as text for the model to read."""
    try:
        if name == "create_persona":
            return _create_persona(args, llm, persona_dir)
        if name == "list_personas":
            return _list_personas(persona_dir)
        if name == "start_conversation":
            return _start_conversation(args, llm, persona_dir, on_turn, on_start)
        return f"Error: there is no tool called '{name}'."
    except KeyError as exc:
        return f"Error: missing required argument {exc}."
    except Exception as exc:  # noqa: BLE001 - a failing tool should be reported, not crash the session
        return f"Error: {type(exc).__name__}: {exc}"


def _create_persona(args: Dict[str, Any], llm: LLM, persona_dir: Path) -> str:
    existing = [s.name for s in list_personas(persona_dir)]
    spec = generate_persona(args["description"], llm, existing_names=existing)
    path = save_persona(spec, persona_dir)
    return f"Created {spec.name} ({spec.role}): {spec.summary} Saved to {path}."


def _list_personas(persona_dir: Path) -> str:
    specs = list_personas(persona_dir)
    if not specs:
        return "There are no personas yet."
    return "\n".join(f"- {s.name} ({s.role}): {s.summary}" for s in specs)


def _start_conversation(
    args: Dict[str, Any],
    llm: LLM,
    persona_dir: Path,
    on_turn: Optional[Callable[[Turn], None]],
    on_start: Optional[Callable[[str], None]],
) -> str:
    names: List[str] = args["participants"]
    topic: str = args["topic"]
    turns = max(1, min(int(args.get("turns", DEFAULT_TURNS)), MAX_TURNS))

    specs = []
    for name in names:
        matches = find_personas(name, persona_dir)
        if not matches:
            return f"Error: no persona matches '{name}'. Create it first, or check the name with list_personas."
        if len(matches) > 1:
            options = ", ".join(m.name for m in matches)
            return f"Error: '{name}' is ambiguous; it matches {options}. Ask the user which one they mean."
        specs.append(matches[0])

    if len({s.name for s in specs}) < len(specs):
        return "Error: the same persona was named twice. A persona cannot talk to itself."
    if len(specs) < 2:
        return "Error: a conversation needs at least two personas."

    agents = [agent_from_spec(s, llm) for s in specs]
    if on_start is not None:
        on_start(topic)
    conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
    path = save_transcript(conversation, persona_dir / "transcripts")
    if conversation.error:
        return (
            f"The conversation stopped early after {len(conversation.turns)} of {turns} turns: "
            f"{conversation.error}. The partial transcript was saved to {path}."
        )
    return f"The conversation finished ({len(conversation.turns)} turns) and was saved to {path}."