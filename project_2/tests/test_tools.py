"""Tests for the tool-calling layer of the LLM protocol."""

from personaforge.llm import ScriptedToolLLM, ToolCall, ToolResponse
from personaforge import PersonaSpec, save_persona
from personaforge.llm import ScriptedLLM
from personaforge.tools import TOOLS, execute_tool
from personaforge.observer import latest_transcript

def test_scripted_tool_llm_replays_responses_in_order():
    llm = ScriptedToolLLM(
        responses=[
            ToolResponse(
                text="",
                tool_calls=[ToolCall(id="t1", name="list_personas", input={})],
                stop_reason="tool_use",
            ),
            ToolResponse(text="Done.", stop_reason="end_turn"),
        ]
    )

    first = llm.complete_with_tools("sys", [{"role": "user", "content": "hi"}], tools=[])
    second = llm.complete_with_tools("sys", [{"role": "user", "content": "hi"}], tools=[])

    assert first.stop_reason == "tool_use"
    assert first.tool_calls[0].name == "list_personas"
    assert second.text == "Done."


def test_scripted_tool_llm_records_the_tools_it_was_given():
    llm = ScriptedToolLLM(responses=[ToolResponse(text="ok")])
    tools = [{"name": "list_personas"}]

    llm.complete_with_tools("SYSTEM_MARKER", [{"role": "user", "content": "x"}], tools=tools)

    assert llm.calls[0]["system"] == "SYSTEM_MARKER"
    assert llm.calls[0]["tools"] == tools

PERSONA_MD = """---
name: Ada Park
role: scientist
summary: a skeptical data scientist
---

# Ada Park
Body.
"""


def test_every_tool_has_a_name_description_and_schema():
    for tool in TOOLS:
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"


def test_create_persona_tool_writes_a_file(tmp_path):
    llm = ScriptedLLM(replies=[PERSONA_MD])
    result = execute_tool("create_persona", {"description": "a skeptic"}, llm, tmp_path)

    assert "Ada Park" in result
    assert (tmp_path / "ada-park.md").exists()


def test_list_personas_tool_when_empty(tmp_path):
    assert "no personas" in execute_tool("list_personas", {}, ScriptedLLM(), tmp_path)


def test_unknown_tool_returns_an_error_string(tmp_path):
    result = execute_tool("make_coffee", {}, ScriptedLLM(), tmp_path)
    assert result.startswith("Error")


def test_missing_argument_returns_an_error_string(tmp_path):
    result = execute_tool("create_persona", {}, ScriptedLLM(), tmp_path)
    assert result.startswith("Error")


def test_conversation_tool_reports_an_ambiguous_name(tmp_path):
    for name in ["Marcus Chen", "Marcus Webb"]:
        save_persona(PersonaSpec(name=name, role="patient", summary="", body="b"), tmp_path)

    result = execute_tool(
        "start_conversation", {"participants": ["marcus", "marcus"], "topic": "pain"}, ScriptedLLM(), tmp_path
    )
    assert "ambiguous" in result


def test_conversation_tool_refuses_the_same_persona_twice(tmp_path):
    save_persona(PersonaSpec(name="Marcus Chen", role="patient", summary="", body="b"), tmp_path)
    save_persona(PersonaSpec(name="Dr. Reyes", role="doctor", summary="", body="b"), tmp_path)

    result = execute_tool(
        "start_conversation",
        {"participants": ["Marcus Chen", "marcus chen"], "topic": "pain"},
        ScriptedLLM(),
        tmp_path,
    )
    assert "cannot talk to itself" in result


def test_conversation_tool_runs_and_caps_turns(tmp_path):
    for name, role in [("Alice Adams", "doctor"), ("Bob Brown", "patient")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)

    llm = ScriptedLLM(replies=["a line"])
    result = execute_tool(
        "start_conversation",
        {"participants": ["Alice Adams", "Bob Brown"], "topic": "x", "turns": 999},
        llm,
        tmp_path,
    )
    assert "20 turns" in result

GOOD_VERDICT = '{"speakers": {"Alice": {"in_character": true, "stage_directions": true}}, "on_topic": true, "notes": "Alice narrates."}'


def write_transcript(tmp_path, name="2026-11-08_100000_x.md"):
    folder = tmp_path / "transcripts"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(
        "# Conversation: x\n\nParticipants: Alice, Bob\n\n**Alice:** hello\n\n**Bob:** hi", encoding="utf-8"
    )


def test_evaluate_tool_reports_the_verdict(tmp_path):
    write_transcript(tmp_path)
    result = execute_tool("evaluate_conversation", {}, ScriptedLLM(replies=[GOOD_VERDICT]), tmp_path)

    assert "Alice used stage directions" in result


def test_evaluate_tool_with_no_transcripts(tmp_path):
    result = execute_tool("evaluate_conversation", {}, ScriptedLLM(), tmp_path)
    assert result.startswith("Error")


def test_evaluate_tool_cannot_escape_the_transcripts_folder(tmp_path):
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    write_transcript(tmp_path)

    result = execute_tool("evaluate_conversation", {"filename": "../secret.md"}, ScriptedLLM(), tmp_path)
    assert result.startswith("Error")


def test_evaluate_tool_unreadable_verdict_is_not_a_pass(tmp_path):
    write_transcript(tmp_path)
    result = execute_tool("evaluate_conversation", {}, ScriptedLLM(replies=["looks fine!"]), tmp_path)

    assert "could not produce" in result
    assert "no problems flagged" not in result