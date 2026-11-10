"""Model loading, 4-bit quantization and LoRA configuration shared by all experiments."""

from __future__ import annotations

import random

# Linear layers adapted by LoRA. Qwen3.5 interleaves standard attention blocks
# with Gated DeltaNet (linear attention) blocks, whose projections have their
# own names; Qwen3 only has the standard ones. PEFT ignores absent names.
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LINEAR_ATTENTION_MODULES = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_TARGET_MODULES = ATTENTION_MODULES + LINEAR_ATTENTION_MODULES + MLP_MODULES


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantization_config():
    """NF4 4-bit weights, bfloat16 compute, double quantization of the constants.

    Double quantization stores the per-block scaling constants in 8 bits instead
    of 32, cutting their overhead from 0.5 to about 0.127 bit per weight.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config(r: int = 32, alpha: int | None = None, dropout: float = 0.1, target_modules=None):
    """LoRA on every attention and MLP projection; alpha defaults to 2r."""
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        target_modules=list(target_modules or ALL_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_name: str, four_bit: bool = True):
    """Load a causal LM (optionally 4-bit) and its tokenizer with a padding token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=quantization_config() if four_bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts.

    bitsandbytes packs two 4-bit weights into each stored uint8 element, so the
    element count of a ``Params4bit`` tensor is doubled to get the true count.
    """
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        n = param.numel()
        if param.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def merge_adapter(base_model_name: str, adapter_dir: str, output_dir: str) -> None:
    """Merge a LoRA adapter into a 16-bit copy of the base model.

    Merging into the 4-bit weights would re-quantize ``W + (alpha / r) BA``; the
    update is small relative to the quantization step and part of it is rounded
    away, so the adapter is merged into full-precision weights instead.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(output_dir)
