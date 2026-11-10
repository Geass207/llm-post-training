"""Tool calling: conversation template, loss masking and the execution loop.

The model never executes anything. It writes a call, stops, and the application
runs the function and appends the result to the conversation:

    <tools>: [{"name": "get_weather", ...}]
    <human>: What's the weather like in Paris right now?
    <assistant>: <call>{"name": "get_weather", "arguments": {"city": "Paris"}}</call>
    <result>: {"city": "Paris", "weather": "12°C, light rain"}
    <assistant>: The weather in Paris is 12°C with light rain.

    python -m posttrain.tools --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Callable

IGNORE_INDEX = -100
CALL_PATTERN = re.compile(r"<call>(.*?)</call>", flags=re.DOTALL)
GLAIVE_ROLES = {"human": "human", "gpt": "assistant", "function_call": "assistant", "observation": "result"}

Turn = tuple[str, str]  # (role, content), role in {"human", "assistant", "result"}


def format_conversation(tools: list[dict], turns: list[Turn], eos_token: str) -> list[tuple[str, bool]]:
    """Render a conversation as ``(text, is_target)`` segments.

    Only the content of assistant turns, each closed by the EOS token, is a
    training target. The EOS token is what teaches the model to hand control
    back to the application after writing a call.
    """
    segments = [(f"<tools>: {json.dumps(tools, ensure_ascii=False)}\n", False)]
    for role, content in turns:
        if role == "assistant":
            segments += [("<assistant>:", False), (f" {content}{eos_token}", True), ("\n", False)]
        else:
            segments.append((f"<{role}>: {content}\n", False))
    return segments


def tokenize_conversation(tokenizer, tools: list[dict], turns: list[Turn]) -> dict:
    """Tokenize segment by segment; tool descriptions, user turns and tool results are masked."""
    input_ids: list[int] = []
    labels: list[int] = []
    for text, is_target in format_conversation(tools, turns, tokenizer.eos_token):
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        input_ids += ids
        labels += ids if is_target else [IGNORE_INDEX] * len(ids)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def glaive_to_turns(conversations: list[dict]) -> list[Turn]:
    """Convert a Glaive function-calling conversation to template turns."""
    turns = []
    for message in conversations:
        content = message["value"]
        if message["from"] == "function_call":
            content = f"<call>{content}</call>"
        turns.append((GLAIVE_ROLES[message["from"]], content))
    return turns


def parse_tool_call(text: str) -> dict | None:
    """Return ``{"name": ..., "arguments": {...}}`` if the text contains a well-formed call."""
    match = CALL_PATTERN.search(text)
    if match is None:
        return None
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):  # some datasets serialise the arguments twice
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return {"name": call["name"], "arguments": arguments}


def run_conversation(
    generate: Callable[[str], str],
    tools: list[dict],
    functions: dict[str, Callable],
    question: str,
    eos_token: str = "",
    max_assistant_turns: int = 3,
) -> list[Turn]:
    """Alternate between the model and the tools until the model answers in plain text.

    ``generate`` maps a prompt to the model's next assistant turn, which keeps
    this loop independent of any model and easy to test.
    """
    turns: list[Turn] = [("human", question)]
    for _ in range(max_assistant_turns):
        prompt = "".join(text for text, _ in format_conversation(tools, turns, eos_token)) + "<assistant>:"
        answer = generate(prompt).strip()
        turns.append(("assistant", answer))
        call = parse_tool_call(answer)
        if call is None:
            break
        if call["name"] not in functions:
            result = {"error": f"unknown tool: {call['name']}"}
        else:
            try:
                result = functions[call["name"]](**call["arguments"])
            except TypeError as error:
                result = {"error": str(error)}
        turns.append(("result", json.dumps(result, ensure_ascii=False)))
    return turns


# --- Demonstration tool: fixed values, so that the origin of the answer is unambiguous ---

def get_weather(city: str) -> dict:
    weather = {"Paris": "12°C, light rain", "Abu Dhabi": "35°C, sunny"}
    return {"city": city, "weather": weather.get(city, "20°C, cloudy")}


WEATHER_TOOLS = [{
    "name": "get_weather",
    "description": "Get the current weather in a given city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The name of the city"}},
        "required": ["city"],
    },
}]
DEMO_QUESTIONS = [
    "What's the weather like in Paris right now?",
    "Should I take an umbrella in Abu Dhabi today?",
    "Can you book a flight to Paris for me?",  # no tool can help: the model should say so
]


def make_generate(model, tokenizer, max_new_tokens: int = 100) -> Callable[[str], str]:
    import torch

    def generate(prompt: str) -> str:
        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, encoding.input_ids.shape[1]:], skip_special_tokens=True)

    return generate


def demo(model, tokenizer) -> list[list[Turn]]:
    model.eval()
    transcripts = []
    for question in DEMO_QUESTIONS:
        turns = run_conversation(
            make_generate(model, tokenizer), WEATHER_TOOLS, {"get_weather": get_weather},
            question, eos_token=tokenizer.eos_token,
        )
        for role, content in turns:
            print(f"<{role}>: {content}")
        print()
        transcripts.append(turns)
    return transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for tool calling (Glaive)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--dataset", default="llamafactory/glaive_toolcall_en")
    parser.add_argument("--output-dir", default="outputs/sft_tools")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import transformers
    from datasets import load_dataset
    from peft import get_peft_model

    from posttrain.common import load_model_and_tokenizer, lora_config, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, four_bit=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("=== before fine-tuning ===")
    demo(model, tokenizer)

    data = load_dataset(args.dataset, split="train")
    data = data.filter(lambda ex: ex["tools"] != "[]")  # keep conversations where tools are available
    data = data.shuffle(seed=args.seed).map(
        lambda ex: tokenize_conversation(tokenizer, json.loads(ex["tools"]), glaive_to_turns(ex["conversations"])),
        remove_columns=data.column_names,
    )

    model = get_peft_model(model, lora_config(r=args.lora_rank))
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
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
            report_to="none",
            seed=args.seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(f"{args.output_dir}/lora_adapter")

    print("=== after fine-tuning ===")
    demo(model, tokenizer)


if __name__ == "__main__":
    main()
