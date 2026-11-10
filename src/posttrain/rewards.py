"""Answer extraction and reward functions for GRPO on arithmetic word problems.

The reward functions follow the TRL signature: they receive the sampled
completions (and the dataset columns as keyword arguments) and return one
scalar per completion.
"""

from __future__ import annotations

import math
import re

ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
# Some reasoning on one or more lines, then a last line holding only the tagged integer.
FORMAT_PATTERN = re.compile(r"^(?:[^\r\n]*\r?\n)+<answer>\d+</answer>\r?\n?$")


def extract_tagged_answer(text: str) -> str | None:
    """Text between the first ``<answer>`` and the following ``</answer>``, stripped."""
    start = text.find(ANSWER_OPEN)
    if start == -1:
        return None
    end = text.find(ANSWER_CLOSE, start + len(ANSWER_OPEN))
    if end == -1:
        return None
    return text[start + len(ANSWER_OPEN):end].strip()


def extract_reference_answer(solution: str) -> str | None:
    """GSM8K solutions end with ``#### <answer>``."""
    parts = solution.split("####")
    return parts[1].strip() if len(parts) > 1 else None


def parse_number(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _responses(completions) -> list[str]:
    return [completion[0]["content"] for completion in completions]


def format_reward(completions, **kwargs) -> list[float]:
    """1 if the completion reasons first and ends with a line ``<answer>INTEGER</answer>``."""
    return [1.0 if FORMAT_PATTERN.match(r) else 0.0 for r in _responses(completions)]


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """2 if the tagged answer is exactly the reference string, else 0."""
    extracted = [extract_tagged_answer(r) for r in _responses(completions)]
    return [2.0 if e == a else 0.0 for e, a in zip(extracted, answer)]


def proximity_reward(completions, answer, tau: float = 0.5, **kwargs) -> list[float]:
    """Smooth alternative to the 0/2 correctness reward.

    With binary rewards a near miss and nonsense score the same, and a group in
    which every sample is wrong has zero advantage everywhere, hence no
    gradient. Partial credit decays with the relative error,

        r = 2 * exp(-|a_hat - a| / (tau * max(1, |a|))),

    and is 2 for an exact answer and 0 when no number can be parsed.
    """
    rewards = []
    for response, reference in zip(_responses(completions), answer):
        predicted, target = parse_number(extract_tagged_answer(response)), parse_number(reference)
        if predicted is None or target is None:
            rewards.append(0.0)
        else:
            rewards.append(2.0 * math.exp(-abs(predicted - target) / (tau * max(1.0, abs(target)))))
    return rewards


REWARD_SETS = {
    "binary": [format_reward, correctness_reward],
    "shaped": [format_reward, proximity_reward],
}
