# LLM Post-Training: Fine-Tuning, Tool Use and Alignment

Four post-training experiments on sub-1B open-weight language models, each run
on a single 16 GB GPU with 4-bit quantization and LoRA adapters:

| Experiment | Method | Model | Data | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| Reasoning with a verifiable reward | **GRPO** (on-policy RL) | Qwen3-0.6B | GSM8K | exact match **10.0% → 49.5%** on 200 test problems |
| French news summarization | **SFT**, completion-only loss | Qwen3.5-0.8B-Base | OrangeSum | held-out loss 2.331 → 2.174 (**−6.7%**) |
| Tool calling | **SFT** on multi-turn tool conversations | Qwen3.5-0.8B-Base | Glaive function calling | calls an unseen tool and answers from its result |
| Preference alignment | **DPO** (off-policy) | Qwen3-0.6B | French Orca preference pairs | qualitative comparison on matched prompts |

These are separate experiments on separate adapters, not one model taken
through four stages.

## Results

### GRPO on GSM8K

The model must reason briefly, then give an integer on a last line between
`<answer>` tags. For each problem it samples a group of completions; each is
scored by two programs — format (0/1) and correctness (0/2) — and its advantage
is its reward standardised within the group. No critic and no learned reward
model are involved.

| Qwen3-0.6B | Correct | Exact match |
| :--- | ---: | ---: |
| Before | 20 / 200 | 10.0% |
| After GRPO (50 steps, 200 training problems, 18 min) | 99 / 200 | 49.5% |

How to read this number:

- **It mixes two skills.** Exact match needs a correct result *and* an answer
  that can be extracted from the tags. The base model often answers at once,
  without reasoning, or puts a unit inside the tags (`<answer> $18 </answer>`),
  so part of the gain is format compliance and part is the reasoning that the
  format induces. `posttrain.evaluate` now reports the tagged-answer rate and
  the strict-format rate next to accuracy to separate the two.
- **It is a 200-problem estimate.** The 95% interval is roughly ±7 points
  around 49.5% and ±4 around 10.0%: the improvement is not in doubt, its exact
  size is. The full test split has 1,319 problems.
- **One seed, greedy decoding**, no KL term to the reference policy (`β = 0`).

### Summarization (SFT)

LoRA (rank 32, 21.6M trainable parameters, 2.8% of the model) on a 4-bit base
model that has only been pretrained. The loss is computed on the summary and
the end-of-sequence token only; instruction and article are masked.

| Optimizer step | 0 | 25 | 50 | 75 | 100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Held-out loss (50 articles) | 2.331 | 2.295 | 2.300 | 2.212 | 2.174 |

100 steps cover 800 of the 4,934 training articles (16% of one epoch) and the
loss is still falling. This is a likelihood result: summary quality (ROUGE,
BERTScore, human judgement) was not measured.

### Tool calling (SFT)

A second adapter is trained for 50 steps on conversations that contain tool
descriptions, calls and results. The model writes
`<call>{"name": ..., "arguments": ...}</call>` and stops; the application
executes the function, appends the result, and the model continues. Two design
choices carry the behaviour: tool results are context and never loss targets,
and every assistant turn ends with the end-of-sequence token, which is what
makes the model hand control back after a call.

Before training, the model invents the weather or role-plays both sides of the
dialogue. After training, it calls a weather tool that never appears in the
training data, answers with the returned values, and declines a flight-booking
request that no available tool can serve — see
[results/samples.md](results/samples.md). The tool returns fixed values and
this is a three-question demonstration, not a tool-use benchmark.

### DPO

981 French preference pairs (answers from a stronger model preferred over
answers from a weaker one), `β = 0.1`, rank-32 LoRA, 246 optimizer steps. Base
and adapted models are compared on identical prompts with greedy decoding. The
adapted model stops inventing a biography when asked to introduce itself and
gives more substantive explanations, but one refusal becomes weaker. No
preference-accuracy or safety metric was computed, so no gain is claimed.

All figures above are stored in
[results/recorded_runs.json](results/recorded_runs.json).

## Implementation notes

- **Quantization.** 4-bit NF4 weights with bfloat16 compute and double
  quantization: quantizing the per-block constants to 8 bits reduces their
  overhead from 0.5 to about 0.127 bit per weight.
- **Memory.** In 16-bit precision with Adam, full fine-tuning of the 0.77B
  model needs about 12 bytes per parameter for weights, gradients and moments
  (9.3 GB); with LoRA only the 21.6M adapter parameters carry gradients and
  optimizer state (1.8 GB) — before activations, in both cases.
- **Hybrid attention.** Qwen3.5 interleaves Gated DeltaNet (linear attention)
  blocks with standard attention blocks; the adapters target the projections
  of both, and the MLPs.
- **Merging.** Adapters are merged into a 16-bit copy of the base model.
  Merging into the 4-bit weights re-quantizes `W + (α/r)·BA` and rounds away
  part of an update that is small relative to the quantization step.
- **Rewards.** `posttrain.rewards` also implements a shaped correctness reward,
  `2·exp(−|â − a| / (τ·max(1, |a|)))`, which gives partial credit to near
  misses. With binary rewards, a group in which every sample is wrong has zero
  advantage everywhere and contributes no gradient.

## Usage

```bash
pip install -e ".[dev]"
python -m pytest                         # unit tests, no GPU or download needed

python -m posttrain.sft   --max-steps 100            # summarization adapter
python -m posttrain.tools --max-steps 50             # tool-calling adapter + demo
python -m posttrain.dpo   --limit 1000               # preference alignment
python -m posttrain.grpo  --limit 200                # GRPO, binary rewards
python -m posttrain.grpo  --limit 200 --reward shaped

python -m posttrain.evaluate gsm8k --model Qwen/Qwen3-0.6B --num-samples 200
python -m posttrain.evaluate gsm8k --model outputs/grpo/merged --num-samples 200
python -m posttrain.evaluate compare --base Qwen/Qwen3-0.6B --tuned outputs/dpo/merged
```

```
src/posttrain/
  common.py     quantization, LoRA configuration, parameter counting, adapter merging
  sft.py        summarization: prompt template, completion-only labels, training
  tools.py      tool-calling template, loss masking, call parsing, execution loop
  dpo.py        preference-pair preparation and DPO training
  rewards.py    answer extraction and reward functions
  grpo.py       prompt construction and GRPO training
  evaluate.py   GSM8K exact match, format rates, side-by-side generations
tests/          label masking, call parsing, tool loop, rewards, scoring
results/        recorded metrics and sample generations
```

The reported numbers come from the original notebook runs of these experiments
on a T4. This package factors the same pipeline into tested modules; the
training entry points have not been re-run end to end since the refactoring, so
expect small differences in exact figures.

## Next experiments

The runs above are deliberately small. What would turn them into conclusions:

1. **GRPO at scale and under control.** Full GSM8K test set, several seeds, the
   full training split, larger groups (8–16 completions) — and a format-only
   baseline (a few SFT steps on the output format) to isolate how much of the
   +39.5 points is reasoning.
2. **Binary versus shaped reward** at equal compute, tracking the fraction of
   groups with zero reward variance.
3. **Reward hacking checks.** Does accuracy hold when the one-shot example or
   the tag names change at test time?
4. **Summarization quality.** ROUGE-L and BERTScore on the OrangeSum test set,
   one full epoch, against the post-trained chat model as a reference.
5. **Tool-use accuracy.** Held-out Glaive conversations: call/no-call decision,
   function name and argument exact match.
6. **DPO measured.** Preference accuracy on held-out pairs (how often the
   chosen answer gets the higher likelihood) before and after, and a sweep over
   `β`.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), 2024.
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K), 2021.
- Kamal Eddine et al., *BARThez: a Skilled Pretrained French Sequence-to-Sequence Model* (OrangeSum), EMNLP 2021.

The experimental setup follows the lab sessions of the ALTeGraD course
(M. Vazirgiannis) of the MVA master's programme, ENS Paris-Saclay.
