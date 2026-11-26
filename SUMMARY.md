# Amoeba / BRICK / CROSS — Research Summary

Three independent empirical reports, 2026-11-24 to 2026-11-26.
Amoeba (13 pages) · BRICK (5 pages) · CROSS (5 pages).
All reports, code, and documents are English-only. All weights and code Apache-2.0.

---

## 0. What we do NOT claim (read this first)

Across all three reports we explicitly decline the following claims:

- No novelty for **draft-head-and-verify** speculative decoding (Medusa, EAGLE, JetSpec).
- No novelty for **early exit / layer dropout** (CALM, LayerSkip, BitSkip).
- No novelty for **dense-to-MoE conversion** (CMoE, ToMoE, ExpertWeaver, DOT-MoE).
- No novelty for **program-of-layers** (PoLar, ICML 2026).
- No novelty for **KV quantization / layer pruning / cross-layer KV** (JetSpec, LCKV,
  MiniCache, AttentionPredictor, KV-Direct, ReplaySSM, Curse of Depth).
- No **bit-exact losslessness** claims. Token agreement on 2B is 0.734; the claim is
  downstream capability preservation, not identical decoding.
- No universal accelerator claims. Every reported speedup is architecture- and
  harness-dependent, with withdrawn results reported in full.

The contribution of this body of work is **not a new mechanism**. It is an empirical
boundary map, an evaluation standard, and a complete negative-result record.

---

## 1. Amoeba — Capability-preserving dynamic activation

**Paper**: `AMOEBA_PAPER_DRAFT.pdf`. Models: 2B / 1B released on HuggingFace with
draft heads; 0.6B / 0.3B ship tool-calling only.

### Contributions

1. **Capability-preservation as the acceptance criterion** for speculative decoding.
   Instead of bit-exactness or token agreement, we use 79 objective questions with
   zero correct-to-incorrect regressions and ROUGE-L 0.94. On 2B: 91.14% -> 93.67%,
   zero regressions, with 2.015x speedup (26.6 -> 53.6 tok/s). This standard —
   "downstream capability preserved + faster" — is the methodological core.

2. **A five-regime architecture boundary map** that no prior speculative-decoding
   paper reports: full-attention dense (works), sliding-window hybrids (numerical
   non-associativity), MoE/KDA (cache rollback blocked), loop-structured (separate
   positive preliminary result), and Qwen family (all speedups withdrawn due to
   verifier/KV-cache incompatibility). Negative results are as much a contribution
   as the positive one.

3. **The harness-dependence finding**. Same 2B model: 27 tok/s in the Python-bound
   HF `generate` harness, 64 tok/s single-prompt under vLLM, 640-1280 tok/s batched.
   vLLM's built-in n-gram speculative decoding gives 1.806x, showing the mechanism
   is framework-independent while the measured absolute numbers are not. This
   explains why published speedups across harnesses are not comparable.

4. **A complete negative-result chain**: 16 failed token-level early-exit experiments
   (~1.16x structural ceiling, last two layers contribute ~1.3 nats CE), K=8 draft
   heads (acceptance collapse), v2 confidence layering (miscalibrated confidence),
   v3 forward-KL (alpha2 0.469 -> 0.020), v4 KV prediction (QA +5pp but decode
   0.973x), Gemma/SparkX/Ling experiments, Qwen withdrawal, and the loop result
   kept separate (45% speedup at +0.04 nats CE, explicitly a lossy trade-off).

5. **Honest withdrawal discipline**: the Qwen numbers were retracted after they had
   circulated; the paper documents the retraction rather than hiding it.

---

## 2. BRICK — 2D topology negative result

**Paper**: `BRICK/BRICK_PAPER.pdf`.

### Contributions

1. **A systematic negative result for 2D rewiring of a frozen model**. Rearranging
   the 1D layer chain into an 8-layer prelude, two parallel heterogeneous branches
   (8 and 6 layers), and a 2-layer coda is learnable for the *fused* path (CE 4.22
   -> 4.04) but **skipping a branch costs +0.40 nats irrecoverably**: the skipped
   branch's information is genuinely absent. Exposing fusion to single-branch
   inputs makes results worse. This is a structural boundary, not an implementation
   defect.

2. **Scale-dependent layer importance**. On 1B, layer 0 (+6.51 nats ablation) and
   layer 4 (+5.62) are cornerstones while the last layer contributes +3.62; on Amoeba's
   2B, the last layer alone contributes ~1.3 nats. Importance shifts from early
   cornerstone layers toward the final readout as scale grows — a boundary that
   affects where dynamic computation can be placed.

3. **Ablation-derived design rules**. 1D layer ablation shows a U-shaped profile
   (critical at both ends, near-zero middle layers); 2D topology ablation shows
   skipping prelude layers 4-7 is catastrophic (CE 3.79 -> 15.66) while skipping a
   branch costs +0.31. Therefore prelude and coda must execute fully; only the
   branch pair is a legal dynamic-activation scope. We also separate single-layer
   residual bypass (later layers compensate) from whole-segment skipping.

4. **Seven bugs caught by local CPU testing before any GPU run**: transformers 5.x
   layer-call signature (`position_embeddings` required), double-zero-init zero
   gradient, representation-distance oracle collapse, softmax argmax router collapse,
   fp32/bf16 dtype mismatch, wrong residual anchor, and LR too high. Each would have
   wasted a remote GPU session.

---

## 3. CROSS — Program-as-expert, with an honest self-audit

**Paper**: `CROSS/CROSS_PAPER.pdf`.

### Contributions

1. **The program-as-expert intersection (the only mechanism-level specificity we
   claim)**. Dense-to-MoE conversion is not novel (CMoE/ToMoE/ExpertWeaver operate
   on FFN neurons); PoLar selects one skip/repeat program. CROSS treats an entire
   layer-execution program as one expert and fuses multiple full program outputs
   with a token-level router. We claim only this specific variant, not MoEfication.

2. **A theory-backed clean fusion result**. With fixed mixing weights (so the router
   is not selecting), probability-level fusion reaches CE 3.719, below the Jensen
   weighted-member bound 3.966 and below the best single program 3.880; hidden-level
   fusion is 3.827 with no guarantee. This is the single surviving positive result
   after auditing, and it has a mathematical guarantee rather than just an
   empirical gap.

3. **A three-stage self-audit that removed two false positives**:
   - In-sample fusion (4.755 vs 4.840, -0.085) was rejected as fitting: 492K
     parameters on 32-64 supervised tokens with labels on both sides.
   - Held-out "complementarity" was rejected: fusion weights collapsed to one-hot,
     so the difference came from selecting different programs, not fusing.
   - A 14-window paired bootstrap (210 tokens) gave fusion-P0 = -0.075, 95% CI
     [-0.156, +0.012] crossing zero, so "better-than-identity programs" was also
     withdrawn as sampling noise.
   The paper reports only what survived: fixed-weight probability fusion (Jensen),
   and negative results for branch skipping and execution-time selection.

4. **Execution-time selector negative result**. A distilled selector halves layer
   executions (93 -> 46, ~50%) but raises CE by +0.28 to +0.93 and is unstable
   across teacher budgets (3x swing between 6 and 10 steps). Selection is not a
   free efficiency knob on a lightly trained teacher.

---

## 4. The common thread (what is actually novel across the three)

The unifying contribution is a **research method and a set of boundaries**, not a
model or a mechanism:

- **Capability over bit-exactness**: 79 questions, zero regressions, ROUGE-L — an
  acceptance standard others in speculative decoding do not use.
- **Boundary mapping over method proposals**: we report five architecture regimes,
  two topology regimes (1D vs 2D), two scales (1B vs 2B), and three harness regimes
  (HF / vLLM single / vLLM batched).
- **Negative results as first-class output**: 16 failed early exits, two withdrawn
  result sets (Qwen; CROSS selection), one falsified remedy (BRICK single-branch
  fusion training), one falsified mechanism (CROSS execution top-k).
- **Self-audit before release**: CROSS's withdrawals were made before publication,
  unlike Amoeba's Qwen retraction which happened after numbers had circulated.
- **Small-model / low-resource scope**: 0.3B-2B, single consumer GPU or CPU-only —
  a regime large-lab papers do not systematically cover.

## 5. Future work noted but not executed

- Training dedicated layers and inserting them into CROSS's program pool.
- A3 Frozen Joint-KV from BRICK (LoRA K/V alignment across branches).
- A6 depth-coordinate interpolation (multi-budget inference without skipping).
- Full GPU convergence and 79-question evaluation for BRICK and CROSS.
