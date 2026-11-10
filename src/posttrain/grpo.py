"""Group Relative Policy Optimization on GSM8K with programmatic rewards.

On-policy: for each problem the model samples a group of completions, each is
scored by reward functions, and the advantage of a completion is its reward
standardised within its group, so no value network is needed.

    python -m posttrain.grpo --limit 200
"""

from __future__ import annotations

import argparse

from posttrain.rewards import REWARD_SETS, extract_reference_answer

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The assistant first thinks shortly about the reasoning process in the mind and then provides the user "
    "with the answer in a new line between <answer> and </answer>.\n"
    "The answer must be a single integer."
)
ONE_SHOT = [
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": "To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.\n<answer>4</answer>",
    },
]


def build_messages(question: str) -> list[dict]:
    """System instruction, one worked example showing the format, then the problem."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *ONE_SHOT, {"role": "user", "content": question}]


def to_training_batch(batch: dict) -> dict:
    return {
        "prompt": [build_messages(q) for q in batch["question"]],
        "answer": [extract_reference_answer(a) for a in batch["answer"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO with LoRA on GSM8K")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--limit", type=int, default=200, help="training problems (0 = full train split)")
    parser.add_argument("--reward", choices=sorted(REWARD_SETS), default="binary")
    parser.add_argument("--num-generations", type=int, default=2, help="completions sampled per problem")
    parser.add_argument("--batch-size", type=int, default=8, help="must be divisible by --num-generations")
    parser.add_argument("--max-completion-length", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.0, help="weight of the KL term to the reference policy")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--four-bit", action="store_true", help="quantize the base model (larger checkpoints)")
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    from posttrain.common import ATTENTION_MODULES, MLP_MODULES, load_model_and_tokenizer, lora_config, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, four_bit=args.four_bit)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model, lora_config(r=args.lora_rank, target_modules=ATTENTION_MODULES + MLP_MODULES)
    )
    model.print_trainable_parameters()

    split = f"train[:{args.limit}]" if args.limit else "train"
    data = load_dataset("openai/gsm8k", "main", split=split).map(to_training_batch, batched=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_SETS[args.reward],
        args=GRPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=args.epochs,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            chat_template_kwargs={"enable_thinking": False},
            beta=args.kl_beta,
            epsilon=0.28,
            temperature=1,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,
            optim="paged_adamw_8bit",
            fp16=True,
            logging_steps=1,
            save_steps=25,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=args.report_to,
            seed=args.seed,
        ),
        train_dataset=data,
        processing_class=tokenizer,
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{args.output_dir}/merged")
    tokenizer.save_pretrained(f"{args.output_dir}/merged")


if __name__ == "__main__":
    main()
