"""
Tests for the conversation loop and the orchestrator's intent routing.

Everything runs against ScriptedLLM, so the whole suite is offline and free.
"""

import pytest

from personaforge import (
    Orchestrator,
    PersonaSpec,
    agent_from_spec,
    run_conversation,
    save_persona,
    save_transcript,
    list_personas,
)
from personaforge.llm import ScriptedLLM
from personaforge.llm import ScriptedToolLLM, ToolCall, ToolResponse
from personaforge.tools import execute_tool

PERSONA_MD = """---
name: {name}
role: {role}
summary: a summary
---

# {name}
Body text.
"""


def two_agents(replies=None):
    llm = ScriptedLLM(replies=replies or ["line A", "line B"])
    a = agent_from_spec(PersonaSpec(name="Alice", role="doctor", summary="", body="A"), llm)
    b = agent_from_spec(PersonaSpec(name="Bob", role="patient", summary="", body="B"), llm)
    return a, b, llm


def test_conversation_alternates_speakers():
    a, b, _ = two_agents()
    convo = run_conversation([a, b], topic="the results", turns=4)

    assert [t.speaker for t in convo.turns] == ["Alice", "Bob", "Alice", "Bob"]


def test_conversation_respects_turn_count():
    a, b, _ = two_agents()
    assert len(run_conversation([a, b], topic="x", turns=5).turns) == 5


def test_conversation_needs_two_agents():
    a, _, _ = two_agents()
    with pytest.raises(ValueError):
        run_conversation([a], topic="x", turns=2)


def test_on_turn_fires_per_turn():
    a, b, _ = two_agents()
    seen = []
    run_conversation([a, b], topic="x", turns=3, on_turn=seen.append)

    assert len(seen) == 3
    assert seen[0].speaker == "Alice"


def test_topic_reaches_the_agents():
    """The scene turn must be visible to the first speaker."""
    a, b, llm = two_agents()
    run_conversation([a, b], topic="UNIQUE_TOPIC_MARKER", turns=1)

    first_call = llm.calls[0]["messages"]
    assert any("UNIQUE_TOPIC_MARKER" in m["content"] for m in first_call)


def test_transcript_contains_every_line():
    a, b, _ = two_agents(replies=["hello", "hi there"])
    convo = run_conversation([a, b], topic="greetings", turns=2)
    text = convo.transcript()

    assert "Alice" in text and "Bob" in text
    assert "hello" in text and "hi there" in text


def test_save_transcript_writes_a_file(tmp_path):
    a, b, _ = two_agents()
    convo = run_conversation([a, b], topic="x", turns=2)
    path = save_transcript(convo, tmp_path)

    assert path.exists()
    assert "Alice" in path.read_text(encoding="utf-8")


# -- orchestrator routing -----------------------------------------------------


def test_create_persona_writes_a_file(tmp_path):
    llm = ScriptedLLM(replies=[PERSONA_MD.format(name="Maria Delgado", role="patient")])
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    result = orch.handle("create a persona patient with back pain")

    assert "Maria Delgado" in result.reply
    assert (tmp_path / "maria-delgado.md").exists()


def test_list_personas_reports_what_exists(tmp_path):
    save_persona(PersonaSpec(name="Maria Delgado", role="patient", summary="s", body="b"), tmp_path)
    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)

    assert "Maria Delgado" in orch.handle("list personas").reply


def test_list_personas_when_empty(tmp_path):
    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)
    assert "No personas yet" in orch.handle("list personas").reply


def test_conversation_requires_two_personas(tmp_path):
    save_persona(PersonaSpec(name="Solo", role="patient", summary="", body="b"), tmp_path)
    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)

    assert "at least two" in orch.handle("have them talk about the weather").reply


def test_conversation_runs_with_two_personas(tmp_path):
    for name, role in [("Alice Adams", "doctor"), ("Bob Brown", "patient")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(replies=["a line"]), persona_dir=tmp_path)
    result = orch.handle("have them talk about the test results")

    assert result.conversation is not None
    assert len(result.conversation.turns) == 6
    assert list((tmp_path / "transcripts").glob("*.md"))


def test_named_participants_are_resolved(tmp_path):
    for name, role in [("Alice Adams", "doctor"), ("Bob Brown", "patient"), ("Carol Chen", "nurse")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(replies=["x"]), persona_dir=tmp_path)
    result = orch.handle("start a conversation between Carol and Bob about the chart")

    assert result.conversation is not None
    assert set(result.conversation.participants) == {"Carol Chen", "Bob Brown"}


def test_unknown_participant_is_reported(tmp_path):
    for name in ["Alice Adams", "Bob Brown"]:
        save_persona(PersonaSpec(name=name, role="x", summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)
    result = orch.handle("start a conversation between Zaphod and Bob about lunch")

    assert "couldn't find" in result.reply


def test_unmatched_input_falls_through_to_chat(tmp_path):
    llm = ScriptedLLM(replies=["Tell me who you want to build."])
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    result = orch.handle("hello, what is this thing?")

    assert result.reply == "Tell me who you want to build."
    assert result.conversation is None


def test_transcripts_do_not_overwrite_each_other(tmp_path):
    a, b, _ = two_agents()
    first = save_transcript(run_conversation([a, b], topic="first topic", turns=2), tmp_path)
    second = save_transcript(run_conversation([a, b], topic="second topic", turns=2), tmp_path)

    assert first != second
    assert first.exists() and second.exists()


def test_transcripts_are_not_loaded_as_personas(tmp_path):
    save_persona(PersonaSpec(name="Alice Adams", role="doctor", summary="", body="b"), tmp_path)
    save_persona(PersonaSpec(name="Bob Brown", role="patient", summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(replies=["a line"]), persona_dir=tmp_path)
    orch.handle("have them talk about the results")

    names = [s.name for s in list_personas(tmp_path)]
    assert "Unnamed" not in names
    assert len(names) == 2

# -- tool-use routing ---------------------------------------------------------


def tool_turn(name, args, call_id="t1"):
    """A scripted model reply that asks for one tool."""
    return ToolResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, input=args)],
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": call_id, "name": name, "input": args}],
    )


def final(text):
    return ToolResponse(text=text, stop_reason="end_turn", raw_content=[{"type": "text", "text": text}])


def test_plain_answer_needs_no_tools(tmp_path):
    llm = ScriptedToolLLM(responses=[final("Who should I create?")])
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    result = orch.handle("create a persona")

    assert result.reply == "Who should I create?"


def test_tool_call_is_executed_and_result_returned_to_model(tmp_path):
    # execute_tool will call the same fake model, so give it a persona to author.
    llm = ScriptedToolLLM(
        responses=[tool_turn("list_personas", {}), final("There are none yet.")]
    )
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    result = orch.handle("who do I have?")

    assert result.reply == "There are none yet."
    # The second request must carry a tool_result quoting the call's id.
    last_message = llm.calls[1]["messages"][-1]
    assert last_message["role"] == "user"
    assert last_message["content"][0]["tool_use_id"] == "t1"
    assert "no personas" in last_message["content"][0]["content"]


def test_tool_errors_go_back_to_the_model(tmp_path):
    llm = ScriptedToolLLM(
        responses=[
            tool_turn("start_conversation", {"participants": ["Zaphod", "Bob"], "topic": "x"}),
            final("I couldn't find Zaphod."),
        ]
    )
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    orch.handle("have Zaphod talk to Bob")

    fed_back = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert fed_back.startswith("Error")


def test_runaway_tool_loop_is_stopped(tmp_path):
    from personaforge.orchestrator import MAX_TOOL_ROUNDS

    llm = ScriptedToolLLM(responses=[tool_turn("list_personas", {}, f"t{i}") for i in range(MAX_TOOL_ROUNDS)])
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    result = orch.handle("loop forever")

    assert "stuck" in result.reply


# -- failure handling ---------------------------------------------------------


class FailingLLM:
    """A fake model that answers `ok_calls` times and then raises, like an API outage."""

    def __init__(self, ok_calls):
        self.ok_calls = ok_calls
        self.n = 0

    def complete(self, system, messages, max_tokens=2048):
        self.n += 1
        if self.n > self.ok_calls:
            raise RuntimeError("API down")
        return "a line"


def test_failed_turn_keeps_earlier_lines_and_records_why():
    llm = FailingLLM(ok_calls=2)
    a = agent_from_spec(PersonaSpec(name="Alice", role="doctor", summary="", body="A"), llm)
    b = agent_from_spec(PersonaSpec(name="Bob", role="patient", summary="", body="B"), llm)

    convo = run_conversation([a, b], topic="x", turns=4)

    assert len(convo.turns) == 2
    assert "API down" in convo.error
    assert "Stopped early" in convo.transcript()


def test_empty_reply_stops_the_conversation():
    a, b, _ = two_agents(replies=["hello", ""])
    convo = run_conversation([a, b], topic="x", turns=4)

    assert len(convo.turns) == 1
    assert "empty reply" in convo.error


def test_conversation_tool_saves_a_partial_transcript(tmp_path):
    for name, role in [("Alice Adams", "doctor"), ("Bob Brown", "patient")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)

    result = execute_tool(
        "start_conversation",
        {"participants": ["Alice Adams", "Bob Brown"], "topic": "x", "turns": 4},
        FailingLLM(ok_calls=1),
        tmp_path,
    )

    assert "stopped early after 1 of 4 turns" in result
    assert list((tmp_path / "transcripts").glob("*.md"))

def test_model_is_told_which_personas_exist(tmp_path):
    save_persona(PersonaSpec(name="Ada Chen", role="analyst", summary="skeptic", body="b"), tmp_path)
    llm = ScriptedToolLLM(responses=[final("ok")])

    Orchestrator(llm=llm, persona_dir=tmp_path).handle("have Ada talk to Elena")

    assert "Ada Chen" in llm.calls[0]["system"]

def test_ambiguous_participant_is_reported(tmp_path):
    for name in ["Marcus Chen", "Marcus Webb"]:
        save_persona(PersonaSpec(name=name, role="patient", summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)
    result = orch.handle("start a conversation between marcus and marcus about the pain")

    assert "more than one" in result.reply
    assert result.conversation is None


def test_same_persona_twice_is_refused(tmp_path):
    for name, role in [("Marcus Chen", "patient"), ("Dr. Reyes", "doctor")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)

    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)
    result = orch.handle("start a conversation between Marcus Chen and marcus chen about the pain")

    assert "can't talk to itself" in result.reply
    assert result.conversation is None