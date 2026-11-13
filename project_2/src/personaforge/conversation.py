"""
conversation.py — put two or more agents in a room and let them talk.

The loop is deliberately dumb: agents take strict turns in the order given, for
a fixed number of turns, and then it stops. No agent decides who speaks next and
no agent can interrupt. Making that smarter is one of the more interesting
things you can do with this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import re
from datetime import datetime
from pathlib import Path

from .agent import PersonaAgent

@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)
    error: Optional[str] = None  # set if the conversation stopped early

    def transcript(self) -> str:
        """Render as plain text, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        text = header + "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)
        if self.error:
            text += f"\n\n*[Stopped early after {len(self.turns)} turns: {self.error}]*"
        return text
    


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = 6,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a round-robin conversation between the given agents.

    :param agents:  two or more PersonaAgents. They speak in list order.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   total lines of dialogue, split across the agents.
    :param on_turn: called after each line, so a UI can print as it goes rather
                    than waiting for the whole conversation to finish.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting turn from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    #modified so the scene is framed as instructions instead of a bare topic
    history: List[Turn] = [Turn(speaker="Scene", text=f"You are in a conversation. The subject is: {topic}. Begin speaking in character now; do not ask for clarification or describe the setup.")]

    #modified so a failing turn does not lose the whole conversation
    for i in range(turns):
        speaker = agents[i % len(agents)]

        try:
            text = speaker.respond(history)
            # An empty reply is a failure too: it was the symptom of the
            # duplicate-name bug, and recording blank turns helps nobody.
            if not text.strip():
                raise ValueError("the model returned an empty reply")
        except Exception as exc:  # noqa: BLE001 - keep what was said, report why we stopped
            conversation.error = f"{speaker.name} failed on turn {i + 1} ({type(exc).__name__}: {exc})"
            break

        turn = Turn(speaker=speaker.name, text=text)

        history.append(turn)
        conversation.turns.append(turn)

        if on_turn is not None:
            on_turn(turn)

    return conversation

#moved transcript to a new folder so the conversation.md is not read as a persona
def save_transcript(conversation: Conversation, directory, filename: Optional[str] = None) -> Path:
    """
    Write the transcript to disk and return its path.

    With no filename, one is generated from the time and topic, e.g.
    `2026-11-08_223045_the-mri-results.md`, so a new conversation never
    overwrites an earlier one. An explicit filename is used as given.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if filename is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        topic_slug = re.sub(r"[^a-z0-9]+", "-", conversation.topic.lower()).strip("-")[:40] or "conversation"
        filename = f"{stamp}_{topic_slug}.md"

    path = directory / filename
    path.write_text(conversation.transcript(), encoding="utf-8")
    return path


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. Turn-taking is strict round-robin. Real conversations do not alternate
#     politely. Who should decide who speaks next — a rule, or a model?
#
#  2. Conversations end after N turns regardless of whether anything was
#     resolved. There is no notion of the discussion being finished.
#
#  3. The "Scene" turn is a hack: a fake speaker that exists only to inject the
#     topic. Is there a cleaner way to set a scene?
#
#  4. Nobody observes the conversation. A third agent could summarize it, judge
#     it, or score whether the doctor actually answered the patient's question —
#     which is the natural bridge to the AI-judge requirement in the final
#     presentation rubric. [DONE]
#
#  5. `save_transcript` always writes to the same filename, so each conversation
#     overwrites the last one. [DONE]
#
#  6. If one agent's reply fails, the whole conversation raises and everything
#     said so far is lost. [DONE]
