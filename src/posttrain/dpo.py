"""Direct Preference Optimization on French preference pairs.

Off-policy alignment: for each prompt the policy is trained to prefer the
chosen answer over the rejected one, relative to a frozen reference model. With
LoRA the reference is the same network with the adapters disabled, so no second
copy of the model is kept in memory.

    python -m posttrain.dpo --limit 1000
"""

from __future__ import annotations

import argparse


def build_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """Chat-formatted prompt ending with the assistant header (thinking mode disabled)."""
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def to_preference_example(tokenizer, example: dict) -> dict:
    return {
        "prompt": build_prompt(tokenizer, example["question"], example.get("system")),
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def fits(tokenizer, example: dict, max_length: int) -> bool:
    """Keep pairs whose prompt plus longest answer fits in the context budget."""
    n_prompt = len(tokenizer.encode(example["prompt"]))
    n_answer = max(len(tokenizer.encode(example["chosen"])), len(tokenizer.encode(example["rejected"])))
    return n_prompt + n_answer < max_length


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO with LoRA on French preference pairs")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="AIffl/french_orca_dpo_pairs")
    parser.add_argument("--output-dir", default="outputs/dpo")
    parser.add_argument("--limit", type=int, default=1000, help="number of preference pairs sampled")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--report-to", default="none", help='"wandb" to log the run')
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    from posttrain.common import load_model_and_tokenizer, lora_config, merge_adapter, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, four_bit=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    data = load_dataset(args.dataset, split="train").shuffle(seed=args.seed).select(range(args.limit))
    data = data.map(lambda ex: to_preference_example(tokenizer, ex), remove_columns=data.column_names)
    data = data.filter(lambda ex: fits(tokenizer, ex, args.max_length))
    print(f"preference pairs after length filtering: {len(data)}")

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,
            optim="paged_adamw_8bit",
            fp16=True,
            logging_steps=1,
            save_strategy="steps",
            save_steps=100,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=args.report_to,
            seed=args.seed,
        ),
        peft_config=lora_config(r=args.lora_rank),
        processing_class=tokenizer,
        train_dataset=data,
    )
    # fp16 mixed precision needs float32 trainable weights
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainer.model.print_trainable_parameters()
    trainer.train()

    adapter_dir = f"{args.output_dir}/lora_adapter"
    trainer.model.save_pretrained(adapter_dir)
    merge_adapter(args.model, adapter_dir, f"{args.output_dir}/merged")


if __name__ == "__main__":
    main()
