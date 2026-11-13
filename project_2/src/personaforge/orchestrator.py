"""
orchestrator.py: the agent you talk to.

The model decides what you meant. Each request goes to Claude along with the tool
definitions in tools.py; Claude asks for tools to be run, and the loop in
_handle_with_tools executes them and feeds the results back until Claude has a
final answer.

Models without tool support (the --offline stub) fall back to the original
regular-expression router, which is kept so the app runs for free with no key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM
from .persona import (
    DEFAULT_PERSONA_DIR,
    PersonaSpec,
    generate_persona,
    list_personas,
    save_persona,
    find_personas,
)

from .tools import TOOLS, execute_tool

_STAGE_MANAGER_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You help them
decide who to create and what those characters should discuss.

Be brief — two or three sentences. When the user seems ready to act, remind them they
can say things like "create a persona doctor", "list personas", or "have them talk
about the test results"."""

_TOOL_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You have tools
to create personas, list them, and start conversations between them. Use them to do
what the user asks, in as many steps as it takes: if they want a conversation between
people who do not exist yet, create those people first.

If the request is unclear (for example, a conversation without a topic, or a name that
matches several personas), ask a short question instead of guessing. After using tools,
tell the user briefly what happened. Do not repeat dialogue that the user has already
seen printed.

Never create a persona the user refers to by name if one with that name already exists
in the list below; use the existing one."""

MAX_TOOL_ROUNDS = 8  # safety stop so a confused model can't loop forever


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None


class Orchestrator:
    """Routes what the user typed to the right part of the package."""

    def __init__(self, llm: LLM, persona_dir: Path = DEFAULT_PERSONA_DIR) -> None:
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.history: List[dict] = []
        self.tool_history: List[dict] = []

    # -- intents ------------------------------------------------------------

    def _create_persona(self, description: str) -> Result:
        """Author a persona, write it to disk, and report where it landed."""
        if not description.strip():
            return Result(reply="Tell me what kind of persona — for example, 'create a persona doctor'.")

        existing = [s.name for s in list_personas(self.persona_dir)]
        spec = generate_persona(description, self.llm, existing_names=existing)
        
        path = save_persona(spec, self.persona_dir)
        return Result(
            reply=f"Created **{spec.name}** ({spec.role}) — {spec.summary}",
            events=[f"wrote {path}"],
        )

    def _list_personas(self) -> Result:
        """Show what is currently in the persona folder."""
        specs = list_personas(self.persona_dir)
        if not specs:
            return Result(reply="No personas yet. Try: create a persona patient")

        lines = [f"- **{s.name}** ({s.role}) — {s.summary}" for s in specs]
        return Result(reply=f"{len(specs)} persona(s) in `{self.persona_dir}`:\n" + "\n".join(lines))

    def _converse(
        self,
        topic: str,
        who: Optional[List[str]] = None,
        turns: int = 6,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Wake up two personas and let them talk."""
        specs: List[PersonaSpec] = []

        if who:
            for name in who:
                matches = find_personas(name, self.persona_dir)
                if not matches:
                    return Result(reply=f"I couldn't find a persona matching '{name}'.")
                if len(matches) > 1:
                    options = ", ".join(m.name for m in matches)
                    return Result(reply=f"'{name}' matches more than one persona: {options}. Which one do you mean?")
                specs.append(matches[0])

            if len({s.name for s in specs}) < len(specs):
                return Result(reply="A persona can't talk to itself. Name two different personas.")
        else:
            specs = list_personas(self.persona_dir)[:2]

        if len(specs) < 2:
            return Result(reply="I need at least two personas before they can talk. Create another one first.")

        agents = [agent_from_spec(s, self.llm) for s in specs]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
        path = save_transcript(conversation, self.persona_dir / "transcripts")

        reply = f"{len(conversation.turns)} turns between {' and '.join(s.name for s in specs)}."
        if conversation.error:
            reply = f"Stopped early after {len(conversation.turns)} turns: {conversation.error}"

        return Result(reply=reply, events=[f"wrote {path}"], conversation=conversation)

    def _chat(self, text: str) -> Result:
        """Nothing matched, so just talk to the user."""
        self.history.append({"role": "user", "content": text})
        reply = self.llm.complete(_STAGE_MANAGER_SYSTEM, self.history, max_tokens=512)
        self.history.append({"role": "assistant", "content": reply})
        return Result(reply=reply)

    # -- routing ------------------------------------------------------------

    def _handle_with_tools(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Let the model decide which tools to run, looping until it has a final answer."""
        self.tool_history.append({"role": "user", "content": text})
        events: List[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            roster = "\n".join(
                f"- {s.name} ({s.role}): {s.summary}" for s in list_personas(self.persona_dir)
            ) or "(none yet)"
            system = f"{_TOOL_SYSTEM}\n\nPersonas that exist right now:\n{roster}"

            response = self.llm.complete_with_tools(system, self.tool_history, TOOLS, max_tokens=1024)

            # Echo the model's reply into the history exactly as it came back.
            self.tool_history.append({"role": "assistant", "content": response.raw_content})

            if response.stop_reason != "tool_use":
                return Result(reply=response.text or "(done)", events=events)

            # Run every tool it asked for, then send all the results back together.
            results = []
            for call in response.tool_calls:
                output = execute_tool(call.name, call.input, self.llm, self.persona_dir, on_turn, on_start)
                events.append(f"{call.name}: {output.splitlines()[0][:100]}")
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})

            self.tool_history.append({"role": "user", "content": results})

        return Result(reply="I got stuck in a loop and stopped. Try rephrasing the request.", events=events)

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """
        Work out what the user meant and do it.

        If the model supports tools, the tool loop decides. Otherwise (offline mode) a
        stack of regular expressions is tried in order and the first match wins.
        """

        if hasattr(self.llm, "complete_with_tools"):
            return self._handle_with_tools(text, on_turn=on_turn, on_start=on_start)
        
        lowered = text.strip().lower()

        # "create a persona doctor" / "make a persona who is a nurse"
        create = re.search(
            r"(?:create|make|add|build|generate)\s+(?:a\s+|an\s+|new\s+)*persona\s*(?:for|of|who is|that is|:)?\s*(.*)",
            lowered,
        )
        if create:
            return self._create_persona(create.group(1).strip())

        # "list personas" / "who do I have"
        if re.search(r"\b(list|show|what|who)\b.*\bpersona", lowered) or lowered in {"list", "ls"}:
            return self._list_personas()

        # "have them talk about X" / "start a conversation between A and B"
        converse = re.search(
            r"\b(?:talk|conversation|converse|chat|discuss|speak)\b",
            lowered,
        )
        if converse:
            topic_match = re.search(r"\babout\s+(.*)$", lowered)
            topic = topic_match.group(1).strip() if topic_match else "an initial consultation"

            who_match = re.search(r"\bbetween\s+(.*?)(?:\s+about\b|$)", lowered)
            who = None
            if who_match:
                who = [w.strip() for w in re.split(r"\s+and\s+|,", who_match.group(1)) if w.strip()]

            return self._converse(topic=topic, who=who, on_turn=on_turn, on_start=on_start)

        return self._chat(text)


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. REPLACE THE REGEX ROUTER WITH REAL TOOL USE. This is the main event.
#     Claude supports tool calling: you describe the tools, the model picks one
#     and fills in the arguments. Roughly:
#
#         tools = [
#             {
#                 "name": "create_persona",
#                 "description": "Create and save a new persona character.",
#                 "input_schema": {
#                     "type": "object",
#                     "properties": {
#                         "description": {"type": "string",
#                                         "description": "What kind of person to create"},
#                     },
#                     "required": ["description"],
#                 },
#             },
#             {"name": "list_personas", ...},
#             {"name": "start_conversation", ...},   # participants, topic, turns
#         ]
#
#     Then loop: send the message with `tools=`, and while
#     `response.stop_reason == "tool_use"`, run the requested tool, append a
#     `tool_result` block, and send it back. The model handles "actually, make
#     it two doctors and a nurse, and have them argue about the diagnosis" —
#     which the regexes below will never manage.  [DONE]
#
#  2. The router cannot ask a clarifying question. "Create a persona" with no
#     description gets a canned string instead of "what kind of person?". [DONE]
#
#  3. Pattern order decides meaning. "list the personas then have them talk"
#     only ever lists. [ONLY APPLIES TO OFFLINE]
#
#  4. `_converse` silently picks the first two personas when you do not name
#     anyone, even if you have six. [FIXED FOR NAMED PARTICIPANTS]
#
#  5. `turns` is hardcoded to 6 and cannot be set from the conversation. [DONE]
#
#  6. `self.history` grows without limit and is only used by `_chat`, so the
#     orchestrator has no memory of the personas it created.
