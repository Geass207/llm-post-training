from conftest import FakeTokenizer

from posttrain.sft import IGNORE_INDEX, build_prompt, tokenize_example


def test_only_summary_and_eos_are_training_targets():
    tok = FakeTokenizer()
    example = tokenize_example(tok, article="Un article.", summary="Un résumé.")
    targets = [i for i, label in zip(example["input_ids"], example["labels"]) if label != IGNORE_INDEX]
    assert tok.decode(targets) == " Un résumé." + tok.eos_token


def test_prompt_tokens_are_masked_and_lengths_match():
    tok = FakeTokenizer()
    example = tokenize_example(tok, article="A", summary="B")
    n_prompt = len(build_prompt("A"))
    assert example["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])


def test_inference_prompt_ends_with_assistant_tag():
    assert build_prompt("texte").endswith("<assistant>:")
