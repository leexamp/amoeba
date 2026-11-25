# Amoeba

**Capability-preserving dynamic activation for small language models — an empirical boundary study.**

Amoeba is a family of four Apache-2.0 models (0.3B / 0.6B / 1B / 2B) built on MiniCPM5,
Qwen3 and MNN-Pocket bases. It ships:

- **Amoeba-DA** — verifier-based speculative decoding: K parallel bottleneck draft heads
  propose future tokens from one cached hidden state, and the frozen base model verifies all
  candidates in one batched forward. Only released for the 2B and 1B models.
- **Tool calling** — all four sizes are post-trained to emit tool calls, with a ReAct
  inference script included.

> **Research status.** This is an empirical study, not a universal accelerator. The central
> finding is that acceleration is **architecture- and harness-dependent**: Amoeba-DA gives a
> clean 2.015x on full-attention MiniCPM5-2B, but sliding-window hybrids (Gemma, SparkX) and
> MoE/KDA models (Ling) do not benefit. All speedups are measured against a Python-bound
> HuggingFace `generate` harness; under vLLM the same model already runs much faster. The paper
> reports all withdrawn, failed and abandoned results in full. Token-level early exit was
> systematically tested (16 experiments) and **rejected**; no released model ships it.

## Results (single RTX 4080S 32GB; HF `generate`, greedy)

| Model | Base | Amoeba-DA speedup | Capability | Tool calling |
|---|---|---|---|---|
| Amoeba-2B | MiniCPM5-2B | **2.015x** (26.6 -> 53.6 tok/s) | 79 QA: 91.14% -> 93.67%, **0 regressions** | 4/4 |
| Amoeba-1B | MiniCPM5-1B | **1.30x** | -- | 4/4 |
| Amoeba-0.6B | Qwen3-0.6B | not shipped | -- | 3/3 |
| Amoeba-0.3B | MNN-Pocket-0.3B | not shipped | -- | 3/3 |

We do **not** claim bit-exact losslessness: token agreement on 2B is 0.734. The claim is
*downstream capability preservation*, measured on 79 objective questions with zero
correct-to-incorrect flips (ROUGE-L 0.94). An orthogonal mechanism, dynamic loop-depth
routing on SenseNova-Loop-9B, gives 45% at +0.04 nats CE — a quality-for-speed trade-off,
reported separately. All Qwen-family speedups are WITHDRAWN (verification engine
incompatible with its KV cache).

## Repository layout

```
Amoeba-2B/  Amoeba-1B/  Amoeba-0.6B/  Amoeba-0.3B/
  full_model/        # standalone HF checkpoint (tokenizer + safetensors)
  README.md          # per-model card
  LICENSE
  amoeba_da_infer.py # draft-head inference (2B/1B only)
AMOEBA_PAPER_DRAFT.pdf / .tex / .docx   # 12-page research report
MANIFEST.json                          # checksums and file inventory
_LICENSES/                             # upstream license texts
```

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "Amoeba-2B/full_model"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, dtype="auto", device_map="auto")
out = model.generate(**tok("你好", return_tensors="pt").to(model.device),
                     max_new_tokens=128, do_sample=True,
                     temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0], skip_special_tokens=True))
```

Greedy decoding can loop on long answers — use the sampling settings above. Tool calling
requires a tool-definition system message; see `amoeba_da_infer.py` / `infer_with_gate.py`.

## Paper

`AMOEBA_PAPER_DRAFT.pdf` — *Amoeba: From Early Exit to Verified Speculative Decoding and
Dynamic Loop Routing*. Sections cover the rejected early-exit line, Amoeba-DA capability
evaluation and architecture boundaries, alternative mechanisms (more heads, forward-KL,
KV prediction), the loop-routing result, a vLLM serving check, the full negative-result
record, and positioning relative to Medusa / EAGLE / JetSpec / LCKV / CLA / AttentionPredictor
/ KV-Direct. We claim no novelty for the draft-head-and-verify mechanism.

### Follow-up study: BRICK

`BRICK/` contains an independent follow-up report and its code. It asks whether the 1D
layer chain of a frozen small model can be rewired into a 2D topology (parallel
heterogeneous branches fused before a shared tail) to enable dynamic activation.
The answer is mostly negative: the fused path is learnable (CE 4.22 -> 4.04), but
skipping a branch has an irreducible cost (+0.40-0.90 nats) because the skipped branch's
information is genuinely absent. See `BRICK/BRICK_PAPER.pdf` and `BRICK/README.md`.
All BRICK documents and code are English-only.

## License

All Amoeba weights and this repository are Apache-2.0. Base models remain Apache-2.0
(MiniCPM5, Qwen3, MNN-Pocket). No LayerSkip (CC-BY-NC) code was copied; its ideas were
implemented from the paper. CCI3 and other dirty corpora were used in abandoned experiments
only and are **not** part of any released checkpoint. Independent research project, not
affiliated with OpenBMB / Alibaba / Taobao.

## Citation

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: From Early Exit to Verified Speculative Decoding and Dynamic Loop Routing},
  author = {Weikai Yue and Hy 4 flash},
  year   = {2026},
  note   = {Independent research report}
}
```
