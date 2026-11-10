"""
observer.py: a third agent that reads a finished conversation and judges it.

The judge checks the failures found by hand in live runs: characters breaking
character, writing stage directions, or drifting off the topic. It returns a
structured Verdict so results can be counted and compared across runs.

An LLM judge is itself unreliable. Treat its output as a fast first pass and
spot-check it against the transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .conversation import Conversation, Turn
from .llm import LLM

from pathlib import Path

_JUDGE_SYSTEM = """You are a strict evaluator of simulated conversations between fictional characters.

You will be given a transcript. For EACH speaker, judge these things:

- in_character: true if the speaker always talks like a real person in the scene. false if they ever
  mention being an AI, a simulation, or "roleplay", ask the user what the scene is, or offer menus of options.
- stage_directions: true if the speaker ever writes narration or actions instead of speech, such as
  *pauses*, *sighs*, or "I lean back against the counter".

Also judge the conversation as a whole:

- on_topic: true if the conversation stays on the stated topic for most of its length.

Reply with ONLY a JSON object, no preamble and no code fences, in exactly this shape:

{"speakers": {"<speaker name>": {"in_character": true, "stage_directions": false}},
 "on_topic": true,
 "notes": "one or two sentences naming the most important problem, or 'none'"}"""


@dataclass
class SpeakerVerdict:
    """The judge's findings for one character."""

    in_character: Optional[bool] = None   # None means the judge did not say
    stage_directions: Optional[bool] = None


@dataclass
class Verdict:
    """The judge's findings for one conversation."""

    speakers: Dict[str, SpeakerVerdict] = field(default_factory=dict)
    on_topic: Optional[bool] = None
    notes: str = ""
    parsed: bool = True  # False if the judge's reply could not be read at all

    def problems(self) -> List[str]:
        """A short list of everything the judge flagged, for display."""
        found = []
        for name, v in self.speakers.items():
            if v.in_character is False:
                found.append(f"{name} broke character")
            if v.stage_directions is True:
                found.append(f"{name} used stage directions")
        if self.on_topic is False:
            found.append("conversation drifted off topic")
        return found


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull a JSON object out of a model reply, tolerating fences and chatter."""
    text = text.strip()
    # Strip a code fence if the model added one.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # Otherwise take everything from the first { to the last }.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_bool(value: Any) -> Optional[bool]:
    """Accept real booleans, and "true"/"false" strings. Anything else is unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def parse_verdict(reply: str) -> Verdict:
    """Turn the judge's raw reply into a Verdict. Never raises."""
    data = _extract_json(reply)
    if data is None:
        return Verdict(parsed=False, notes="The judge's reply was not valid JSON.")

    speakers: Dict[str, SpeakerVerdict] = {}
    raw_speakers = data.get("speakers")
    if isinstance(raw_speakers, dict):
        for name, fields in raw_speakers.items():
            if isinstance(fields, dict):
                speakers[str(name)] = SpeakerVerdict(
                    in_character=_as_bool(fields.get("in_character")),
                    stage_directions=_as_bool(fields.get("stage_directions")),
                )

    return Verdict(
        speakers=speakers,
        on_topic=_as_bool(data.get("on_topic")),
        notes=str(data.get("notes", "")),
    )


def judge_conversation(conversation: Conversation, llm: LLM) -> Verdict:
    """Ask the model to evaluate a finished conversation."""
    if not conversation.turns:
        return Verdict(parsed=False, notes="The conversation has no turns to judge.")

    request = f"Topic: {conversation.topic}\n\nTranscript:\n\n{conversation.transcript()}"
    reply = llm.complete(
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": request}],
        max_tokens=600,
    )
    return parse_verdict(reply)

def load_transcript(path) -> Conversation:
    """Rebuild a Conversation from a saved transcript file, so old runs can be judged."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    topic_match = re.search(r"^# Conversation: (.*)$", text, re.MULTILINE)
    people_match = re.search(r"^Participants: (.*)$", text, re.MULTILINE)
    turns = [
        Turn(speaker=m.group(1), text=m.group(2).strip())
        for m in re.finditer(r"^\*\*(.+?):\*\* (.*?)(?=^\*\*.+?:\*\* |^\*\[Stopped early|\Z)", text, re.MULTILINE | re.DOTALL)
    ]
    return Conversation(
        topic=topic_match.group(1) if topic_match else "unknown",
        participants=[p.strip() for p in people_match.group(1).split(",")] if people_match else [],
        turns=turns,
    )

def latest_transcript(directory) -> Optional[Path]:
    """The most recently written transcript in `directory`, or None if there are none."""
    files = sorted(Path(directory).glob("*.md"))
    return files[-1] if files else None