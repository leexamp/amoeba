---
license: apache-2.0
base_model:
  - openbmb/MiniCPM5-2B
  - openbmb/MiniCPM5-1B
  - Qwen/Qwen3-0.6B
  - taobao-mnn/MNN-Pocket-0.3B
datasets:
  - in-house synthetic tool-use dialogues (self-generated from base models)
library_name: transformers
pipeline_tag: text-generation
tags:
  - dynamic-activation
  - amoeba-da
  - verified-speculative-decoding
  - tool-calling
  - on-device
  - efficient-inference

---

# Amoeba: Verified Dynamic Activation for Language Models

**Amoeba** is a family of language models with **Amoeba-DA** (verified dynamic activation) and tool-calling (agent) capability. Amoeba-DA uses K parallel draft heads to propose tokens `t+2..t+K+1` from a single hidden state; the untouched base model verifies all candidates in one batched forward. A candidate is accepted only if it matches the target model's own greedy argmax, so decoding is statistically equivalent to full-depth greedy decoding.

> **Status (updated).** The earlier token-level early-exit (halt head) line was systematically rejected — 16 measured routes all failed on this 42-layer deep-narrow architecture. The shipped technique is **Amoeba-DA**: **2.02×** on Amoeba-2B (27.1 → 54.7 tok/s, RTX 4080S) and **1.30×** on Amoeba-1B. Tool calling is included (2B/1B LoRA SFT; 0.6B/0.3B native tool templates). See §5 and `docs/动态激活_完整实验矩阵.md`.

---

## 1. Introduction

Conventional transformers apply the same number of layers to every token. This is wasteful on device and edge scenarios, where a large fraction of tokens (function words, punctuation, easy continuations) could be resolved with far less computation.

Amoeba augments a pretrained base model with a lightweight **per-layer halt head**. At each layer, the model asks a learned binary question: *"can I stop here?"* A token exits as soon as its **cumulative exit probability** crosses a confidence threshold. The result is a model whose effective depth adapts per token and per prompt, controlled by a single inference-time `threshold`.

The design is deliberately base-agnostic. The same wrapper has been applied across three model families (MiniCPM5, Qwen3, MNN-Pocket) and four scales, from 2B down to 120M, to study how the quality–speed tradeoff scales with model size.

---

## 2. Core Innovation

> **Historical scope.** Sections 2–4 and 6 document the *early-exit* research line, which was investigated and **rejected** (see §5.2b and the experiment matrix). It is retained as the rationale for the design choice, not as a claimed capability of the released checkpoints. The shipped technique is Amoeba-DA (parallel draft heads + full-model verification).

### 2.1 Per-layer halt heads

For a base model with `L` transformer layers and hidden size `H`, Amoeba attaches one halt head after every layer:

```python
halt_heads = nn.ModuleList([
    nn.Linear(H + 1, 1) for _ in range(L)
])
```

The `+1` input is a **token-level uncertainty scalar** derived from the next-token distribution at that layer:

```
unc = tanh(-log(p_max))     # p_max = max softmax probability of the layer's logits
score_l = halt_heads[l](concat(h_l, unc))     # h_l = hidden state at layer l
p_l = sigmoid(score_l)
```

Intuitively, a token that the layer is already confident about (high `p_max`, low `unc`) is a better candidate for early exit. The uncertainty signal lets the halt head condition its decision on both the representation and the model's confidence.

### 2.2 Exit decision by cumulative probability

Exit is not decided by a single layer in isolation. The halt head emits a per-layer probability `p_l`, and the model tracks the **cumulative exit probability**:

```
survival = Π_{j<l} (1 - p_j)        # probability of having survived to layer l
P(exit by layer l) = 1 - survival
```

A token exits at the first layer where `1 - survival >= threshold`. A **higher `threshold` is more conservative** — the token traverses more layers before stopping; `threshold = 1.0` effectively forces full-depth evaluation. This gives practitioners a single knob to trade speed for quality at inference time.

### 2.3 The halt loss

The halt heads are trained with a composite objective designed to avoid confident-but-wrong early exits:

| Term | Purpose |
|---|---|
| **Uniform prior** | Encourage an approximately uniform distribution over exit layers so that every exit point is trained, rather than collapsing onto a single shallow layer. |
| **Semantic consistency** | Constrain intermediate-layer outputs to agree semantically with the final layer (via KL divergence and/or cosine similarity on logits / normalized hidden states). This is the key to *losing less quality* at early exits. |
| **Entropy regularization** | Penalize premature collapse of the halt distribution toward one-hot decisions, stabilizing training. |
| **Route loss** | Weight each layer's language-model loss by the probability of routing to it, so the halt head is optimized for token-level language modeling. |

The distinguishing claim of this design is that a learned halt head with semantic and entropy constraints should **lose less quality than a plain confidence threshold** — a hypothesis we are quantifying rather than asserting.

### 2.4 Per-layer reinforcement training

An early failure taught us that training only the halt heads (with a frozen base) is insufficient: the per-layer language-model loss becomes a *read-only probe* with no path to update the decoder, so intermediate layers never become "exitable." The remedy is **per-layer reinforcement training**:

- **Unfreeze the decoder** so that the per-layer cross-entropy gradients actually reach the layers that produce each intermediate hidden state.
- **Per-layer independent LM loss**, computed at every enabled exit with **uniform or inverse-CE exit weights** (rather than a quadratic weighting that starves the shallow layers).
- **Final-layer anchor** (`L_anchor`) to prevent catastrophic forgetting of full-depth quality.
- **Layer dropout** (LayerSkip-style, `p_max ≈ 0.1–0.2`): stochastically skip deeper layers during training so that shallow layers learn to carry predictive information themselves.
- **Self-distillation** from the final layer to intermediate exits (KL on logits + cosine on representations), used as a refinement after per-layer CE is under control.

After the base is strengthened, the halt heads are **recalibrated** on the new hidden-state distribution. This "train the athlete, then train the judge" ordering is central to the method.

### 2.5 Ideas, not copied code

Layer dropout, early-exit language-model loss, and self-distillation are **ideas** inspired by published research (LayerSkip, Elhoushi et al. 2024; CALM, Schuster et al. 2022). Our implementation is written from scratch. **No LayerSkip source code was copied** — see [License & Acknowledgments](#7-license--acknowledgments).

---

## 3. Model Family

| Variant | Base | Arch | Layers | Hidden | Vocab | Status |
|---|---|---|---:|---:|---:|---|
| **Amoeba-2B** | MiniCPM5-2B | `LlamaForCausalLM` | 42 | 2048 | 130560 | Main line; active development (exit-collapse fix in progress) |
| **Amoeba-1B** | MiniCPM5-1B | `LlamaForCausalLM` | 24 | 1536 | 130560 | Weights prepared; adaptation smoke-tested |
| **Amoeba-0.6B** | Qwen3-0.6B | `Qwen3ForCausalLM` | 28 | 1024 | 151936 | Configs validated; training in progress |
| **Amoeba-0.3B** | MNN-Pocket-0.3B | `Qwen3ForCausalLM` | 28 | 896 | 65536 | Configs validated; training in progress |
| **Amoeba-120M** | Custom tiny testbed | Qwen3-like | 16 | 768 | 4104 | Testbed only — mechanism validated, base too weak for quality evaluation |

**Notes on the variants**

- **2B / 1B** share the MiniCPM5 tokenizer, vocabulary (130560), and special tokens (`eos = [1, 130073]`, `pad = 1`). The 1B model has `hidden_size = 1536` (not 2048), so all dimensions are read dynamically from `config`.
- **0.6B / 0.3B** are native `Qwen3ForCausalLM` models (not "Llama-like") with Qwen3's `q_norm`/`k_norm` inside the attention layers. They are loaded with `AutoModelForCausalLM`; `trust_remote_code` is not required. Their EOS/PAD ids differ (`0.6B: 151645 / 151643`, `0.3B: 65512 / 65510`) and corpora must be re-tokenized to each base's vocabulary.
- **120M** is a small testbed built during an earlier, now-deprecated small-vocabulary experiment. Its dynamic-activation mechanism works (depth responds smoothly to `threshold`), but generation quality is too poor to evaluate early-exit quality loss; it is included only as a mechanistic demonstration.
- The family spans **three base-model families and four scales**, enabling a scaling study of "dynamic-activation benefit vs. model size." The 2B line is the validation target; the smaller variants follow once the method is confirmed.

---

## 4. Usage

Amoeba does not modify the base model's modeling code. It uses a **thin wrapper** that holds the base model plus the halt heads and steps through layers manually. This preserves `generate()` behavior and avoids breaking KV-cache handling in `transformers`.

### 4.1 Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AmoebaModel:
    """Base LM + one halt head per layer. No changes to modeling_*.py."""

    def __init__(self, base, halt_heads):
        self.base, self.halt_heads = base, halt_heads

    @classmethod
    def from_pretrained(cls, base_path, halt_path, attn_implementation="sdpa"):
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,  # wrapper relies on SDPA causal masking
        ).cuda().eval()
        base.config.use_cache = True
        cfg = base.config
        heads = torch.nn.ModuleList([
            torch.nn.Linear(cfg.hidden_size + 1, 1).to(base.device)
            for _ in range(cfg.num_hidden_layers)
        ])
        ckpt = torch.load(halt_path, map_location="cpu", weights_only=False)
        heads.load_state_dict(ckpt["halt_heads"])
        return cls(base, heads)

model = AmoebaModel.from_pretrained(
    "openbmb/MiniCPM5-2B", "path/to/halt_heads.pt"
)
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B")
```

### 4.2 Dynamic inference with `threshold`

`threshold` is compared against the **cumulative exit probability**. Higher values are more conservative (deeper exit); lower values exit earlier.

```python
@torch.no_grad()
def generate(model, tok, prompt, threshold=0.7, max_new_tokens=64,
             forced_exit=None):
    """
    threshold in [0, 1]: higher => more conservative => more layers.
      Illustrative depth on the 120M testbed: 0.5 -> ~3.8 layers,
      0.8 -> ~7.1 layers, 1.0 -> all 16 layers.
    forced_exit: int in [0, L-1] => evaluate a fixed exit layer (no routing).
                 Useful for quality-vs-depth evaluation.
    """
    base, heads = model.base, model.halt_heads
    L = len(heads)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    past = None
    for _ in range(max_new_tokens):
        h = base.model.embed_tokens(ids)
        pos = torch.arange(h.shape[1], device=h.device)[None, :]
        cos, sin = base.model.rotary_emb(h, pos)
        survival, exit_at = 1.0, L - 1
        for i, layer in enumerate(base.model.layers):
            h = layer(h, position_embeddings=(cos, sin), past_key_value=past,
                      use_cache=True)[0]
            if forced_exit is not None:
                if i == forced_exit:
                    exit_at = i
                    break
                continue
            logits = base.lm_head(base.model.norm(h[:, -1:]))
            p_max = logits.float().softmax(-1).max(-1).values
            unc = torch.tanh(-p_max.log()).clamp_min(0)
            p_l = heads[i](torch.cat([h[:, -1, :], unc[:, None]], -1)).sigmoid()
            survival *= (1.0 - p_l.item())
            if 1.0 - survival >= threshold:
                exit_at = i
                break
        logits = base.lm_head(base.model.norm(h[:, -1:]))
        ids = torch.cat([ids, logits.argmax(-1)[:, None]], -1)
    return tok.decode(ids[0], skip_special_tokens=True), exit_at

text, layer = generate(model, tok, "The capital of France is", threshold=0.7)
print(f"exited at layer {layer}: {text}")
```

> The snippet is **illustrative** — it exposes the routing logic (`threshold`, cumulative probability, `forced_exit`). A production implementation should handle the KV cache and per-layer `past_key_value` objects explicitly; incorrect cache handling can erase or even reverse the theoretical speedup.

### 4.3 Forced-exit evaluation

To measure quality at a fixed depth, pass `forced_exit=i`. Sweeping `i = 0 … L-1` yields the **forced-exit CE/PPL curve**, the standard diagnostic for how "exitable" each intermediate layer is. A healthy model shows a roughly flat curve close to the final-layer CE; the current checkpoints do not yet (see below).

---

## 5. Performance & Current Status

### 5.1 Measured speed (Amoeba-DA)

| Model | Baseline greedy | Amoeba-DA | Speedup | Avg accepted tokens/step |
|---|---:|---:|---:|---:|
| Amoeba-2B | 27.1 tok/s | 54.7 tok/s | **2.02×** | 2.61 |
| Amoeba-1B | 44.7 tok/s | 58.0 tok/s | **1.30×** | — |

Draft-head top-1 agreement on chat holdout (2B): 0.469 / 0.348 / 0.293 / 0.240 for offsets 2–5. Crucially, training the heads on **on-policy self-generated chat text** (CE 1.31) instead of off-policy web text (CE 3.71) raised α₂ from 0.138 to 0.469 and speedup from 1.13× to 2.02×.

### 5.2 Quality / exactness statement

Amoeba-DA accepts a candidate only when it matches the target model's own argmax at that position. In **bf16**, batched verification and single-token greedy round near-ties (logits within ~0.01) differently, so output is **not literally bit-identical**: 6/20 prompts matched exactly, with **83.7% positional token match** and an average matching prefix of 75.9 tokens. Quality is therefore *statistically lossless*, not bit-exact — do not claim otherwise. See `medusa_agreement.json`.

### 5.2b Why token-level early exit was rejected

Token-level early exit, layer skipping, attention skipping, MLP skipping and confidence-gated exits were all measured and failed (16 experiments). The 42-layer deep-narrow MiniCPM5 architecture has **no inter-layer redundancy**: keeping 76% of layers still yields CE 11.7 (≈ ln V, i.e. random). Gate cost is ~1.25 layers (an `lm_head` call), while a usable exit depth is ≥40 of 42 — ceiling ≈1.16×. Full data: `docs/动态激活_完整实验矩阵.md`.

### 5.3 Dialogue quality measurement (8 fixed prompts, greedy)

To answer "how much does dialogue quality drop", we compared three modes on the same 8 fixed prompts: **original base, full depth** / **Amoeba full depth (τ=1, no early exit)** / **Amoeba quality gate (conf≥0.95)**. Lower 3-gram repetition and garble rates are better; a smaller exit layer means earlier exit.

| Model | Mode | Avg length | 3-gram repeat | Garble | Avg exit layer |
|---|---|---:|---:|---:|---:|
| **Amoeba-2B** | Original full | 234 | 0.000 | 0.007 | 42 |
| | Amoeba full | 238 | 0.006 | 0.009 | 42 |
| | Amoeba gated | 128 | 0.179 | 0.002 | 36.2 |
| **Amoeba-1B** | Original full | 264 | 0.308 | 0.000 | 24 |
| | Amoeba full | 302 | 0.159 | 0.000 | 24 |
| | Amoeba gated | 258 | 0.589 | 0.000 | 18.6 |
| **Amoeba-0.6B** | Original full | 205 | 0.000 | 0.017 | 28 |
| | Amoeba full | 208 | 0.000 | 0.013 | 28 |
| | Amoeba gated | 187 | 0.013 | 0.019 | 24.2 |
| **Amoeba-0.3B** | Original full | 92 | 0.000 | 0.001 | 28 |
| | Amoeba full | 93 | 0.000 | 0.003 | 28 |
| | Amoeba gated | 161 | 0.140 | 0.000 | 10.5 |

**Takeaways:** (1) **Amoeba full depth ≈ original** — length, repetition, and garble are nearly identical, confirming the modification itself is safe. (2) **The quality gate has a visible cost** — e.g. 2B 3-gram repetition rises 0.000→0.179 and answers shorten (234→128); 0.3B exits at only 10.5/28 layers with repetition 0.140. (3) **Smaller models are more aggressive and lose more** — consistent with weaker intermediate representations. (4) **Acceleration remains limited** — 2B exits at 36.2/42 layers, saving little; 0.3B is more aggressive but pays a repetition penalty. These are local-protocol numbers, not official evaluation-stack results.

---



---

## 5.5 Usage notes & honest limitations (read before deploying)

1. **Greedy decoding loops on long answers.** On 2B, greedy generation repeated a
   sentence 20+ times in a long role-play answer. Use the recommended sampling:
   `temperature=0.8, top_p=0.95, repetition_penalty=1.05`. All reported capability
   deltas were measured with the recovered official chat template
   (`<think>` enabled); our short-subset greedy scores are **not comparable** to the
   official model-card numbers (MATH-500 94.6 etc., sampled at temperature 1.0).
2. **Tool calls require the tool-definition system message.** `agent_infer.py` supplies
   it. Asking for weather with a bare chat template makes the model say it has no
   real-time data — this is a prompt issue, not a capability failure.
3. **Training data correction.** Early experiments used CCI3-Data, but it is a noisy
   Chinese web/news corpus (CE 3.7 vs 1.31 on self-generated chat) and **was not used
   for the released checkpoints**. The released tool-calling SFT used in-house
   synthetic tool-use dialogues generated from the base models. The math-CoT SFT
   experiment using OpenMathReasoning degraded MMLU and was **discarded**; no released
   model includes it.
4. **0.6B tool-call checkpoint:** the first 2-epoch run overfit (loss 0.0008) and
   produced empty replies. The released `ckpt_06b_agent_v2` uses 1 epoch / lr 5e-5.
5. **Draft heads are only shipped for 2B and 1B.** On 0.6B/0.3B they measured 0.60× /
   0.36× (slower) and are not distributed as a speed feature.
6. **vLLM is not usable in this environment** (v0.11 ABI-incompatible with torch
   2.15+cu132); all timings use HuggingFace `generate` and are CPU/Python-bound.

---

## 6. Training Method (Overview)

The training pipeline proceeds in stages, each with a gate before continuing:

1. **Baseline reproduction.** Record the untouched base model's CE/accuracy and its forced-exit curve; confirm the final layer is the quality ceiling.
2. **Method viability.** Use overfit tests to verify gradients reach both the halt heads (`halt_grad > 0`) and the decoder (`base_grad > 0`), and that per-layer CE can be reduced at a target layer without hurting the final layer.
3. **Per-layer reinforcement.** Unfreeze the decoder; apply per-layer independent LM loss (uniform or inverse-CE weights) plus a final-layer anchor. Optionally add layer dropout (`p_max ≈ 0.1–0.2`) and self-distillation (KL at `T = 2`, cosine on normalized representations) with a low learning rate (base `1e-5`–`5e-5`, heads `1e-4`–`2e-4`).
4. **Halt-head recalibration.** With the strengthened base fixed, retrain the halt heads with the full halt objective and calibrate the exit threshold (rather than relying on a hard-coded value).
5. **Capability post-training (planned).** Add tool-use and reasoning ability with agent data (OpenThoughts-Agent-SFT-100K, Mixture-of-Thoughts, OpenMathReasoning), re-tokenized into each base's vocabulary.

**Research lineage.** The layer-dropout + early-exit-loss recipe follows the ideas of **LayerSkip** (Elhoushi et al., 2024) and the confidence-adaptive per-token halting of **CALM** (Schuster et al., 2022). We implement these ideas independently; see the licensing note below.

---

## 7. License & Acknowledgments

**Amoeba weights and code are released under Apache-2.0.** Amoeba is a derivative work of the base models listed below, with added tool-calling SFT and **Amoeba-DA** (verified dynamic activation via parallel draft heads). This is an independent research project, **not affiliated with or endorsed by OpenBMB, Alibaba/Qwen, or Taobao/MNN**.

### Base models

| Base | Organization | License |
|---|---|---|
| MiniCPM5-2B, MiniCPM5-1B | OpenBMB | Apache-2.0 |
| Qwen3-0.6B | Alibaba / Qwen | Apache-2.0 |
| MNN-Pocket-0.3B | Taobao / MNN | Apache-2.0 |

### Training data

- **BAAI/CCI3-Data** — Apache-2.0
- **open-thoughts/OpenThoughts-Agent-SFT-100K** — Apache-2.0
- **nvidia/OpenMathReasoning** — CC-BY-4.0 (attribution required; we gratefully acknowledge this dataset)
- Reasoning traces in Mixture-of-Thoughts are distilled from **DeepSeek-R1** (MIT, which permits蒸馏 / derivative use); the upstream data-card license field is absent, so we treat this source conservatively.

### Research references

- **LayerSkip** — Elhoushi et al., 2024, *"LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"* (Meta). Ideas only (layer dropout, early-exit loss, self-distillation). The original LayerSkip **code is CC-BY-NC** and **was not copied**; our implementation is independent and uses no LayerSkip source files.
- **CALM** — Schuster et al., 2022, *"Confident Adaptive Language Modeling"* (Google). Confidence-adaptive per-token halting; reference implementation is Apache-2.0.
- We preserve all upstream copyright notices and include a `NOTICE` file listing base-model attributions, as required by Apache-2.0.

---

## 8. Limitations

1. **Under active tuning.** Dynamic-activation quality is not yet reliable. Exit collapse in early checkpoints means current weights should be treated as research artifacts, not deployable models; no validated quality–speed curve is published.
2. **Text-first.** The current work targets pure text generation. Multimodal dynamic activation is planned as a future extension and is not part of these checkpoints.
3. **Threshold calibration.** The exit `threshold` is task- and model-dependent. A single hard-coded value (e.g. 0.7) is not universally appropriate; the soft expected depth used in training does not necessarily match the hard exit taken at inference. Users should calibrate the threshold on their target task.
4. **Overhead can erode gains.** Per-layer `lm_head`/halt-head calls, Python-level layer iteration, and KV-cache handling can reduce or negate theoretical speedups if not implemented carefully.
5. **Small-model sensitivity.** Smaller bases (0.3B, 120M) have limited capacity and are more prone to exit collapse; their halt-loss hyper-parameters need more conservative settings. The 120M testbed's tiny legacy vocabulary (4104) makes it unsuitable for quality evaluation.

---

## 9. Citation

If you use Amoeba in research, please cite this project and the base models / methods it builds upon:

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Verified Dynamic Activation for Language Models},
  year   = {2026},
  note   = {Independent research project; Apache-2.0}
}

@inproceedings{elhoushi2024layerskip,
  title  = {LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
  author = {Elhoushi, Mostafa and Chen, Weijie and others},
  year   = {2024}
}

@inproceedings{schuster2022calm,
  title  = {Confident Adaptive Language Modeling},
  author = {Schuster, Tal and Fisch, Amit and others},
  year   = {2022}
}
```

Please also cite the relevant base-model papers (MiniCPM5, Qwen3, MNN-Pocket) and datasets used in your variant.

---

## 10. Disclaimer

Amoeba is an independent research project. It is **not** an official model from OpenBMB, Alibaba/Qwen, or Taobao/MNN, and carries no endorsement from them. The models are provided for research and educational purposes; the quality limitations described above apply.
