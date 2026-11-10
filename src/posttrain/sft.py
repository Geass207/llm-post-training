"""Supervised fine-tuning of a base model for French news summarization.

The model only learns to write the summary: prompt tokens are masked out of the
loss (completion-only fine-tuning).

    python -m posttrain.sft --max-steps 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE_INDEX = -100
INSTRUCTION = "Résumez l’article suivant:"
ASSISTANT_TAG = "<assistant>:"


def build_prompt(article: str, summary: str | None = None) -> str:
    """``<human>: instruction + article`` followed by the assistant tag (and the summary)."""
    prompt = f"<human>: {INSTRUCTION}\n{article}\n{ASSISTANT_TAG}"
    return prompt if summary is None else f"{prompt} {summary}"


def tokenize_example(tokenizer, article: str, summary: str) -> dict:
    """Tokenize one example with labels restricted to the summary and the EOS token.

    Without the EOS token in the targets the model never learns where to stop.
    """
    text = build_prompt(article, summary) + tokenizer.eos_token
    input_ids = list(tokenizer(text)["input_ids"])
    n_prompt = len(tokenizer(build_prompt(article))["input_ids"])
    labels = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def prepare_dataset(tokenizer, dataset_name: str, max_length: int, eval_size: int, seed: int):
    from datasets import load_dataset

    data = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    data = data.map(
        lambda ex: tokenize_example(tokenizer, ex["text"], ex["reference-summary"]),
        remove_columns=data.column_names,
    )
    data = data.filter(lambda ex: len(ex["input_ids"]) <= max_length)
    split = data.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def generate_summary(model, tokenizer, article: str, max_new_tokens: int = 200) -> str:
    import torch

    encoding = tokenizer(build_prompt(article), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoding.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA summarization fine-tuning (OrangeSum)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--dataset", default="giuliadc/orangesum_5k")
    parser.add_argument("--output-dir", default="outputs/sft_summarization")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import transformers
    from peft import get_peft_model

    from posttrain.common import count_parameters, load_model_and_tokenizer, lora_config, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, four_bit=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(model, lora_config(r=args.lora_rank))
    trainable, total = count_parameters(model)
    print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_data, eval_data = prepare_dataset(
        tokenizer, args.dataset, args.max_length, args.eval_size, args.seed
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            lr_scheduler_type="cosine",
            warmup_steps=0.01,
            optim="paged_adamw_8bit",
            fp16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_total_limit=3,
            report_to="none",
            seed=args.seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    model.config.use_cache = False

    loss_before = trainer.evaluate()["eval_loss"]
    trainer.train()
    loss_after = trainer.evaluate()["eval_loss"]

    out = Path(args.output_dir)
    model.save_pretrained(out / "lora_adapter")
    summary = {
        "model": args.model,
        "max_steps": args.max_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "heldout_examples": len(eval_data),
        "heldout_loss_before": loss_before,
        "heldout_loss_after": loss_after,
        "relative_reduction": (loss_before - loss_after) / loss_before,
        "log_history": trainer.state.log_history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"held-out loss: {loss_before:.4f} -> {loss_after:.4f}")


if __name__ == "__main__":
    main()
