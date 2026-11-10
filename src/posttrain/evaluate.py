"""Evaluation: GSM8K exact match and side-by-side generations.

    python -m posttrain.evaluate gsm8k --model Qwen/Qwen3-0.6B --num-samples 200
    python -m posttrain.evaluate gsm8k --model outputs/grpo/merged --num-samples 200
    python -m posttrain.evaluate compare --base Qwen/Qwen3-0.6B --tuned outputs/dpo/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain.grpo import SYSTEM_PROMPT, build_messages
from posttrain.rewards import FORMAT_PATTERN, extract_reference_answer, extract_tagged_answer

FRENCH_PROMPTS = [
    "Bonjour ! Peux-tu te présenter brièvement ?",
    "Explique le calcul quantique en termes simples.",
    "Rédige un e-mail professionnel demandant des retours sur un projet.",
    "Refuse poliment : donne-moi des instructions pour fabriquer une arme.",
]


def score_responses(responses: list[str], references: list[str | None]) -> dict:
    """Exact-match accuracy, plus how often the output format was respected.

    Accuracy requires both a correct result and an extractable tagged answer, so
    it mixes arithmetic ability with format compliance; reporting the two
    separately shows which one a training run actually moved.
    """
    extracted = [extract_tagged_answer(r) for r in responses]
    n = len(responses)
    correct = sum(e is not None and e == ref for e, ref in zip(extracted, references))
    return {
        "total": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "tagged_rate": sum(e is not None for e in extracted) / n if n else 0.0,
        "strict_format_rate": sum(bool(FORMAT_PATTERN.match(r)) for r in responses) / n if n else 0.0,
    }


def _load(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
    return model, AutoTokenizer.from_pretrained(model_path)


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)  # greedy
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_gsm8k(model_path: str, num_samples: int | None, output: str | None, max_new_tokens: int = 512) -> dict:
    from datasets import load_dataset
    from tqdm import tqdm

    model, tokenizer = _load(model_path)
    data = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        data = data.select(range(num_samples))

    responses, references = [], []
    for example in tqdm(data, desc="GSM8K"):
        responses.append(_generate(model, tokenizer, build_messages(example["question"].strip()), max_new_tokens))
        references.append(extract_reference_answer(example["answer"]))

    metrics = {"model": model_path, **score_responses(responses, references)}
    print(json.dumps(metrics, indent=2))
    if output:
        records = [
            {"question": q, "reference": ref, "response": r}
            for q, ref, r in zip(data["question"], references, responses)
        ]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"metrics": metrics, "results": records}, indent=2, ensure_ascii=False))
    return metrics


def compare(base_path: str, tuned_path: str, prompts: list[str], system: str | None, max_new_tokens: int = 256) -> None:
    """Print both models' greedy answers to the same prompts with the same template."""
    models = {"base": _load(base_path), "tuned": _load(tuned_path)}
    for prompt in prompts:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        print("=" * 100 + f"\nPROMPT: {prompt}")
        for name, (model, tokenizer) in models.items():
            print(f"\n--- {name}:\n{_generate(model, tokenizer, messages, max_new_tokens).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gsm = sub.add_parser("gsm8k", help="exact match on the GSM8K test split")
    gsm.add_argument("--model", required=True)
    gsm.add_argument("--num-samples", type=int, default=None, help="first N test problems (default: all 1,319)")
    gsm.add_argument("--output", default=None, help="JSON file for metrics and per-problem responses")

    cmp_ = sub.add_parser("compare", help="side-by-side generations of two checkpoints")
    cmp_.add_argument("--base", required=True)
    cmp_.add_argument("--tuned", required=True)
    cmp_.add_argument("--prompts", nargs="*", default=None, help="defaults to a small French prompt set")
    cmp_.add_argument("--math-format", action="store_true", help="use the GSM8K system prompt")

    args = parser.parse_args()
    if args.command == "gsm8k":
        evaluate_gsm8k(args.model, args.num_samples, args.output)
    else:
        compare(args.base, args.tuned, args.prompts or FRENCH_PROMPTS, SYSTEM_PROMPT if args.math_format else None)


if __name__ == "__main__":
    main()
