import json

from conftest import FakeTokenizer

from posttrain.tools import (
    IGNORE_INDEX, WEATHER_TOOLS, get_weather, glaive_to_turns, parse_tool_call,
    run_conversation, tokenize_conversation,
)

CALL = '<call>{"name": "get_weather", "arguments": {"city": "Paris"}}</call>'


def test_loss_covers_assistant_turns_only():
    tok = FakeTokenizer()
    turns = [("human", "Weather in Paris?"), ("assistant", CALL),
             ("result", '{"weather": "rain"}'), ("assistant", "It rains.")]
    example = tokenize_conversation(tok, WEATHER_TOOLS, turns)
    targets = tok.decode([i for i, l in zip(example["input_ids"], example["labels"]) if l != IGNORE_INDEX])
    assert targets == f" {CALL}{tok.eos_token} It rains.{tok.eos_token}"
    assert '"weather": "rain"' not in targets  # tool results are context, never targets


def test_parse_tool_call():
    assert parse_tool_call(CALL) == {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert parse_tool_call("No tool needed.") is None
    assert parse_tool_call("<call>not json</call>") is None
    nested = '<call>{"name": "f", "arguments": "{\\"x\\": 1}"}</call>'
    assert parse_tool_call(nested) == {"name": "f", "arguments": {"x": 1}}


def test_glaive_conversion_wraps_function_calls():
    turns = glaive_to_turns([
        {"from": "human", "value": "hi"},
        {"from": "function_call", "value": '{"name": "f", "arguments": {}}'},
        {"from": "observation", "value": "{}"},
        {"from": "gpt", "value": "done"},
    ])
    assert [role for role, _ in turns] == ["human", "assistant", "result", "assistant"]
    assert turns[1][1].startswith("<call>") and turns[1][1].endswith("</call>")


def test_loop_executes_the_call_and_feeds_the_result_back():
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return CALL if len(prompts) == 1 else "It is 12°C with light rain."

    turns = run_conversation(generate, WEATHER_TOOLS, {"get_weather": get_weather}, "Weather in Paris?")
    assert [role for role, _ in turns] == ["human", "assistant", "result", "assistant"]
    assert json.loads(turns[2][1]) == {"city": "Paris", "weather": "12°C, light rain"}
    assert "<result>:" in prompts[1] and prompts[1].endswith("<assistant>:")


def test_loop_stops_on_plain_answer_and_survives_bad_calls():
    turns = run_conversation(lambda p: "I cannot book flights.", WEATHER_TOOLS, {}, "Book a flight")
    assert len(turns) == 2

    unknown = '<call>{"name": "book_flight", "arguments": {}}</call>'
    turns = run_conversation(lambda p: unknown, WEATHER_TOOLS, {"get_weather": get_weather}, "Book", max_assistant_turns=1)
    assert "unknown tool" in turns[-1][1]

    bad_args = '<call>{"name": "get_weather", "arguments": {"town": "Paris"}}</call>'
    turns = run_conversation(lambda p: bad_args, WEATHER_TOOLS, {"get_weather": get_weather}, "Weather", max_assistant_turns=1)
    assert "error" in json.loads(turns[-1][1])
