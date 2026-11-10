import pytest

from posttrain.evaluate import score_responses
from posttrain.grpo import build_messages, to_training_batch
from posttrain.rewards import (
    correctness_reward, extract_reference_answer, extract_tagged_answer, format_reward, proximity_reward,
)


def wrap(*texts):
    return [[{"role": "assistant", "content": t}] for t in texts]


GOOD = "48 / 2 = 24, and 48 + 24 = 72.\n<answer>72</answer>"


def test_extract_tagged_answer():
    assert extract_tagged_answer(GOOD) == "72"
    assert extract_tagged_answer("<answer> 18 </answer>") == "18"
    assert extract_tagged_answer("The answer is 72") is None
    assert extract_tagged_answer("<answer>72") is None


def test_extract_reference_answer():
    assert extract_reference_answer("48/2 = 24\n#### 72") == "72"
    assert extract_reference_answer("no marker") is None


def test_format_reward_requires_reasoning_then_integer_on_last_line():
    rewards = format_reward(wrap(GOOD, "<answer>72</answer>", "Reasoning.\n<answer>$72</answer>", GOOD + "\nextra"))
    assert rewards == [1.0, 0.0, 0.0, 0.0]


def test_correctness_reward_is_exact_match():
    assert correctness_reward(wrap(GOOD, GOOD, "none"), answer=["72", "73", "72"]) == [2.0, 0.0, 0.0]


def test_proximity_reward_gives_partial_credit():
    exact, near, far, missing = proximity_reward(
        wrap(GOOD, "x\n<answer>70</answer>", "x\n<answer>7</answer>", "no answer"),
        answer=["72"] * 4,
    )
    assert exact == pytest.approx(2.0)
    assert 0.0 < far < near < exact
    assert missing == 0.0


def test_score_responses_separates_accuracy_from_format():
    metrics = score_responses([GOOD, "<answer>72</answer>", "72"], ["72", "72", "72"])
    assert metrics["correct"] == 2 and metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["tagged_rate"] == pytest.approx(2 / 3)
    assert metrics["strict_format_rate"] == pytest.approx(1 / 3)


def test_prompt_is_system_one_shot_then_question():
    messages = build_messages("How many?")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "How many?"
    batch = to_training_batch({"question": ["Q"], "answer": ["work\n#### 5"]})
    assert batch["answer"] == ["5"] and batch["prompt"][0][-1]["content"] == "Q"
