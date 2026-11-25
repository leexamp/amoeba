---
license: apache-2.0
base_model:
  - openbmb/MiniCPM5-2B        # replace per variant
datasets:
  - in-house synthetic tool-use dialogues (self-generated from base models)
library_name: transformers
pipeline_tag: text-generation
tags:
  - amoeba
  - speculative-decoding
  - tool-calling
  - on-device
---

# Amoeba-{SIZE}

Amoeba-{SIZE} is an independent research derivative of **{BASE}** (Apache-2.0). It is
post-trained for tool calling and, for 2B/1B, ships parallel draft heads for
Amoeba-DA verified speculative decoding. This is **not** an official model of the base
model's organization.

## What was changed

- Tool-use SFT on synthetic self-generated tool-call dialogues (calculator, weather,
  search, time), merged into a standalone checkpoint.
- For 2B/1B only: K parallel bottleneck draft heads (`medusa_chat.pt`) trained on
  model-generated chat data; the backbone weights are unchanged.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(".")
model = AutoModelForCausalLM.from_pretrained(".", dtype="auto", device_map="auto")
# tool calls require a tool-definition system message (see repo agent_infer.py)
# recommended sampling: temperature=0.8, top_p=0.95, repetition_penalty=1.05
```

## Reported results

- Tool-call success: {TOOL}
- Amoeba-DA speedup: {SPEED} (single RTX 4080S, HF generate, greedy)
- Greedy decoding may loop on long answers; use the sampling settings above.

## Limitations

Amoeba-DA output matches greedy decoding up to bf16 near-ties (statistically near-lossless,
not bit-exact). Speedup depends on draft acceptance rate; on Qwen3-8B the same method
gives only 1.1–1.4×. An earlier early-exit line was rejected and is not shipped.
Released tool SFT used only in-house synthetic dialogues; CCI3 and OpenMathReasoning
were used in abandoned experiments only.

## License and attribution

Apache-2.0. Derivative of {BASE} (Apache-2.0). No LayerSkip code was copied
(CC-BY-NC); its ideas were implemented from the paper. Independent research project,
not affiliated with the base model providers.
