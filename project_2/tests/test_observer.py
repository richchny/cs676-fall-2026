"""Tests for the observer. The judge's replies are scripted, so everything is offline."""

from personaforge import PersonaSpec, agent_from_spec, run_conversation
from personaforge.llm import ScriptedLLM
from personaforge.observer import judge_conversation, parse_verdict

GOOD_REPLY = """{"speakers": {"Alice": {"in_character": true, "stage_directions": false},
                           "Bob": {"in_character": false, "stage_directions": true}},
 "on_topic": true, "notes": "Bob narrates."}"""


def short_conversation():
    llm = ScriptedLLM(replies=["line A", "line B"])
    a = agent_from_spec(PersonaSpec(name="Alice", role="doctor", summary="", body="A"), llm)
    b = agent_from_spec(PersonaSpec(name="Bob", role="patient", summary="", body="B"), llm)
    return run_conversation([a, b], topic="the results", turns=2)


def test_parses_a_clean_reply():
    v = parse_verdict(GOOD_REPLY)

    assert v.parsed
    assert v.speakers["Alice"].in_character is True
    assert v.speakers["Bob"].stage_directions is True
    assert v.on_topic is True


def test_parses_a_code_fenced_reply():
    v = parse_verdict("```json\n" + GOOD_REPLY + "\n```")
    assert v.parsed and "Bob" in v.speakers


def test_parses_a_reply_with_chatter_around_it():
    v = parse_verdict("Sure, here is my evaluation:\n" + GOOD_REPLY + "\nHope that helps!")
    assert v.parsed and "Alice" in v.speakers


def test_garbage_reply_does_not_raise():
    v = parse_verdict("I think it went well!")
    assert v.parsed is False


def test_missing_fields_are_unknown_not_false():
    v = parse_verdict('{"speakers": {"Alice": {"in_character": true}}}')

    assert v.speakers["Alice"].stage_directions is None
    assert v.on_topic is None


def test_problems_lists_only_what_was_flagged():
    problems = parse_verdict(GOOD_REPLY).problems()

    assert "Bob broke character" in problems
    assert "Bob used stage directions" in problems
    assert not any("Alice" in p for p in problems)


def test_judge_sends_the_transcript_to_the_model():
    judge_llm = ScriptedLLM(replies=[GOOD_REPLY])
    judge_conversation(short_conversation(), judge_llm)

    request = judge_llm.calls[0]["messages"][0]["content"]
    assert "the results" in request
    assert "line A" in request


def test_judge_refuses_an_empty_conversation():
    from personaforge.conversation import Conversation

    judge_llm = ScriptedLLM(replies=[GOOD_REPLY])
    v = judge_conversation(Conversation(topic="x", participants=["A", "B"]), judge_llm)

    assert v.parsed is False
    assert judge_llm.calls == []  # no API call was wasted