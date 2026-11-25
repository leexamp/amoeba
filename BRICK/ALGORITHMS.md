# BRICK — Algorithm Designs and Full Negative-Result Record (2026-11-25)

> Three hard constraints (from Amoeba's empirical findings, inviolable):
> 1. **Readout is indivisible**: the final layer alone contributes ~1.3 nats CE; it always
>    stays in the coda.
> 2. **KV cache must stay causally consistent**: our speed numbers are extremely harness-
>    sensitive (27 vs 640 tok/s); any algorithm breaking cache causality fails end-to-end.
> 3. **Frozen backbone**: train only lightweight modules (<10M params); otherwise this
>    becomes another from-scratch-training work with no positioning.

---

## A0. Baseline: Residual-Aligned Fusion (correction to the original PoC)

**Problem**: the original `BrickFusion` fed the equal-weight average of branch outputs
directly into the coda. But branch1 (layers 31-39 in the 42-layer config) receives a
"layer-15 output" distribution rather than the pretrained "layer-30 output", so the
average can differ greatly from `h_pre`, damaging the residual stream from step one.

**Algorithm**: use a residual correction, zero-initialized:
```
h_coda = h_pre + alpha * Fusion(branch_0(h_pre), branch_1(h_pre))
```
where `alpha` is learnable (init 0) and Fusion's output layer is zero-initialized.
The start point is strictly the original model; training can only improve or preserve.

**Why it may work**: LoRA-style zero-initialized residual — the optimization start point
is the identity map, gradients are clean. Amoeba's draft heads also use a zero-initialized
`fc2` (medusa_train4.py), validated as stable.

**Risk (partially validated on real 1B, 2026-11-25)**:
- On local 1B (24 layers, 8+[8,6]+2), the 2D-rearrangement start MSE is as high as
  **19.6** (fusion identity, 2D parallel output vs 1D sequential teacher). After 4 steps
  MSE drops to 9.8, but **alpha stays ~0 and goes negative** (-0.0011): a small residual
  correction converges too slowly under a large offset.
- **Diagnosis**: `h_pre + alpha*proj(branches)` anchors at the prelude output, but after
  2D rearrangement branch outputs differ hugely from the prelude; small alpha cannot
  compensate quickly.
- **Fix (verified by three-mode comparison + LR sweep on real 1B, 2026-11-25)**:
  - **Residual anchor**: `residual_branch_avg` (anchor = branch-output mean) starts at
    MSE **5.72**, vs `residual_prelude` (h_pre anchor) **9.92** — the branch mean has
    passed all middle layers and is closer to the teacher.
  - **`direct` mode falsified**: equal-average init without a residual anchor explodes
    to MSE 19.15.
  - **Initialization**: must be alpha=1 + zero-initialized proj (start = branch-mean
    identity, but proj keeps full gradient). alpha=0 + normal proj gives ~0 alpha
    gradient and no learning.
  - **LR=3e-5 is the sweet spot** (swept 1e-5/3e-5/1e-4): 8 steps MSE 5.72->4.45,
    stable; 3e-4 explodes (->19), 1e-4 explodes then recovers (->5.37, unstable).
  - Fixed in `brick_config.py`: `FUSION_MODE="residual_branch_avg"`, `LR=3e-5`.
- On the tiny model the start MSE is only 0.014 (small layer differences with random
  weights), so this issue was invisible — another "tiny cannot reveal, real model reveals" case.

**Prior work**: zero-initialized residual is a generic trick (LoRA, ControlNet); no novelty claimed.

**Priority P0** (implemented).

---

## A1. Oracle-Routed Branching (teacher-signal-distilled token-level router)

**Core problem**: in `all` mode both branches are computed, no speedup. In routed mode
only one branch per token, but "which one" has no supervision — our v2 lesson was that
self-confidence is miscalibrated (draft confidence decoupled from correctness), so the
model's own confidence cannot be the routing signal.

**Algorithm**: use teacher signals as the oracle:
```
for each token t: teacher = hidden after sequential execution of the original model h_t*
                  branch_k output = h_t,k
                  oracle label y_t = argmin_k ||h_t,k - h_t*||^2
                  train router r(h_pre) -> p(k|h_pre) to fit y_t (cross-entropy)
```
At inference `k_t = argmax r(h_pre,t)`, compute only branch k_t.

**Why it may work**: bypasses unreliable self-confidence — the teacher directly says
which branch is closer to the true representation. Self-supervised routing in
representation space, not confidence routing.

**Risk**:
- Teacher labels are teacher-forcing; autoregressive inputs shift distribution (Amoeba's
  on-policy chat corpus mitigates this).
- **Observed locally**: branch collapse to [1.0, 0.0]. The teacher is the full middle-stack
  output, so a deeper branch is naturally closer than a shallow one — the oracle picks the
  deep branch at every token. A mini version of "last layer dominates".
  **Fix direction**: use CE decrease / relative improvement instead of absolute distance,
  so the shallow branch is chosen on tokens where it is already good enough (early-exit
  logic), and the deep branch only when the shallow one is insufficient.
  A5's balance loss must also be enabled.

  **Controlled experiment verified (test_cpu_router_fix.py, tiny CPU, 2026-11-25)**:

  | Exp | Config | oracle label dist | router choice dist |
  |---|---|---|---|
  | E1 | repr oracle + hetero (4,2) | [1.00, 0.00] | [1.00, 0.00] |
  | E2 | CE oracle + hetero (4,2) | [0.54, 0.46] | [0.99, 0.01] |
  | E3 | repr + hetero + balance | [1.00, 0.00] | [1.00, 0.00] |
  | E4 | repr + equal (3,3) | [0.97, 0.03] | [1.00, 0.00] |
  | E5 | CE + equal (3,3) | [0.53, 0.47] | [0.91, 0.09] |
  | E6 | CE + equal + balance | [0.53, 0.47] | [0.85, 0.15] |

  **Two independent problems separated**:
  1. **Supervision definition**: repr oracle labels 100% biased to the deep branch; CE
     oracle labels are balanced (~50/50). Balance loss cannot save wrong labels (E3 proves).
     Oracle must be "representation distance" -> "CE" (`ce_oracle` implemented and moved
     into `brick_router.py`).
  2. **Router training dynamics (solved)**: even with balanced labels, the router still
     favors one branch (E2 0.99). **Gumbel-Softmax experiment (test_cpu_gumbel.py, 7
     controls) identifies the root cause**:
     - plain router (softmax+CE): [0.19, 0.81] — collapse reproduced
     - Gumbel-Softmax (soft, tau 1.0->0.1): **[0.48, 0.52]** — perfectly balanced
     - extra entropy / balance terms (G4-G7) give no further improvement — Gumbel's
       random exploration is the remedy itself
     - **cost**: agree drops 0.58 -> 0.52 (balance bought with exploration) — an honest
       speed-accuracy trade-off.
     - Root cause is winner-take-all from non-differentiable argmax, not labels or
       regularization. `brick_router_gumbel.py` implements it; `train_router.py --router gumbel`
       is the default.

**Prior work**: MoE routers / Switch Transformer are known mechanisms; our difference is
frozen backbone + teacher representation as oracle; no mechanism novelty claimed.

**Priority P0** (implemented).

---

## A2. Segment-Level Routing (pragmatic fix for KV-cache consistency)

**Core problem**: token-level routing is fatal — if token t goes through branch0 (layers
16-30 update cache) and token t+1 through branch1 (layers 31-39), branch1's layers did not
write KV at time t, so t+1's attention sees a context missing t -> causality broken.

**Algorithm**: route by **segment**, not by token:
```
choose a branch once every S=16 tokens; all tokens in the segment use that branch.
```
At segment boundaries both branches' cache lengths match (same number of processed tokens);
do one KV alignment at the boundary (lightweight projection fills the inactive branch's cache).

**Why it may work**: Stepped MoE / MoD show segment-level routing is viable; our difference
is branch-segment routing on a frozen backbone. Short context (Amoeba's strength) has few
segments and low alignment overhead.

**Risk**: too coarse loses the fine-grained dynamic-activation advantage; too fine lets
alignment overhead eat the speedup. S=16 is a hyperparameter to tune.

**Prior work**: Stepped MoE (arXiv:2610.07348) already does segment-level routing; no novelty.

**Priority P1** (fallback if A1's cache problem is hard to solve).

---

## A3. Frozen Joint-KV (cross-branch KV alignment, most valuable and hardest)

**Core problem**: MBT's Joint-KV shares cache because its branches are trained from
scratch and representations align naturally. Our branches are frozen pretrained layers
at different depths; representations do not align.

**Algorithm**: add a lightweight K/V projection head (rank-16 LoRA) after **each layer of
each branch**:
```
branch_k layer l output h -> K_l^k = W_K^l @ h, V_l^k = W_V^l @ h  (shared KV space)
all branches write K_l^k into the same cache slot l
```
Train these heads to align branch KV with teacher KV (teacher KV as supervision).

**Why it may work**: directly solves A2's causality problem — all tokens write KV at all
branches, cache is always complete; only FFN/attention compute can be selected by branch.
This is the true "2D topology + dynamic activation".

**Risk**:
- number of heads = layers x branches, params may exceed 10M (rank compression needed)
- projection-head training is unstable (Amoeba's v4 KV-prediction lesson: QA +5pp but
  decode 0.973x) — must measure end-to-end speed, not only QA

**Prior work**: MBT Joint-KV (from-scratch); our difference is frozen backbone + LoRA KV
alignment; no novelty claimed.

**Priority P2** (most scientific value, but wait for A0/A1 results).

---

## A4. Progressive Branch Growing (safest "never collapse" option)

**Core idea**: do not start with 2 branches — grow them progressively from the original
1D model:
```
Stage 0: single branch = original model (100% capability)
Stage 1: duplicate middle layers to form branch1, fusion zero-initialized (still 100%)
Stage 2: train fusion + router so branch1 takes part of the compute
```
Each stage starts from the previous stage's capability — isomorphic to MIDAS/UpLoop depth
growth, but on a frozen backbone.

**Why it may work**: completely avoids "2D rearrangement destroys representation" risk —
the start point is the original model; training only improves. If Amoeba's `all` mode
collapses, A4 is the fallback.

**Prior work**: MIDAS (arXiv:2410.06270), UpLoop (2026) do depth growth; no novelty.

**Priority P1** (fallback for A0).

---

## A5. Branch Specialization

**Core idea**: instead of requiring both branches to handle all tokens, let them
**specialize**:
```
branch0 <- math / symbolic / code tokens (where branch0 is closer in teacher signal)
branch1 <- language / commonsense / Chinese tokens
```
Use a learnable router + load-balancing loss (avoid assigning everything to one branch).

**Why it may work**: Amoeba's 79-question set naturally has 5 categories (math 33 / fact 12 /
reasoning 14 / Chinese 10 / English 19) — we can directly test whether branch selection
aligns with question type, an **interpretability result** valuable even without speedup.

**Risk**: specialization may reduce capability on some question types (zero-regression
criterion broken) -> negative result.

**Prior work**: MoE expert specialization is known; no novelty.

**Priority P2** (exploratory; interpretability > speed value).

---

## A6. Depth-Coordinate Interpolation (continuous depth interpolation)

**Core idea**: treat each branch's "processing depth" as a continuous coordinate
(reference LiFT's continuous depth coordinate):
```
branch0 output means "depth 15", branch1 output means "depth 9"
fusion learns a depth embedding, output = sum_k softmax(w_k) * branch_k
where w_k is determined jointly by token difficulty and target depth budget
```
This gives multiple inference budgets in one checkpoint by adjusting a target depth
(similar to Accordion).

**Why it may work**: LiFT shows continuous depth coordinates let models loop beyond
training depth; our difference is branch-depth interpolation on a frozen backbone.

**Risk**: fusion must be retrained, and the depth-embedding supervision signal is unclear
(what is the "target depth"?).

**Prior work**: LiFT (arXiv:2610.05538), Accordion (2026); no novelty.

**Priority P3** (most exploratory, wait for prior results).

---

## A7. Confidence-Gated Coda (rejected, reason recorded)

**Core idea**: use branch disagreement `||branch0 - branch1||` as confidence: low
disagreement -> skip part of the coda layers.

**Why not recommended**: coda has only 2 layers (including readout), saving <5%; and v2
showed model self-confidence is unreliable. Listed only to **explicitly record that we
considered and rejected it**, to avoid repeating the mistake.

**Priority rejected** (goes into the report's negative-design section).

---

## Supplement 1: Layer/Block Importance Ablation (2026-11-25, real 1B)

**Method**: skip a layer (residual bypass), measure next-token CE increase Delta CE.
6 chat windows, seq 16.

**1D per-layer results**:
- layer 0: **+6.51** (most critical cornerstone layer)
- layer 4: +5.62
- layer 22: +5.02, layer 23: +3.62
- layers 5-21: ~0 (mostly negative) — **middle 17 layers almost entirely redundant**
- **U-shaped**: critical at both ends, redundant in the middle

**Block-level results (1D topology)**:
- skip prelude(0-7): **+7.35** (least skippable)
- skip coda(22-23): +5.74
- skip branch0(8-15): +0.086, branch1(16-21): +0.067 — **almost free**

**Meaning for BRICK**:
1. Our topology `prelude(0-7)+[branch0(8-15),branch1(16-21)]+coda(22-23)`
   **exactly places critical layers in prelude/coda and redundant layers in branches** —
   the partition is empirically justified.
2. Branch skip Delta CE ~0.07 suggested dynamic activation was almost free — **later
   shown to be an over-optimistic estimate** (see Supplement 2).

**Relation to Amoeba (scale-dependence, not contradiction)**:
- Amoeba 2B (42 layers): last layer alone ~1.3 nats CE
- here 1B (24 layers): layer 0 (+6.51) more critical than last layer (+3.62)
- **1B relies more on early cornerstone layers, 2B on the final readout layer** —
  importance distribution changes with scale.
- metrics also differ (chat teacher-forcing CE here vs self-generated reply CE in
  Amoeba), not directly comparable.

**Honest note**: negative Delta CE (layer 1 = -1.73) is 6-window small-sample noise,
not evidence that the layer is harmful.

**Prior work (no novelty claimed)**:
- Investigating Layer Importance in LLMs (BlackboxNLP 2024): cornerstone layers, early
  layers dominate
- The Curse of Depth in LLMs (NeurIPS 2025): Pre-LN deep layers contribute less
- No Free Swap (arXiv:2605.16234): per-layer skip PPL table + "swap-similar != prunable"
- Reassessing Layer Pruning (ICLR 2026): Pre-LN deep layers marginal
- **Our difference**: on top of these 1D analyses, we do **block-level ablation of a 2D
  rewired topology** and use it to guide dynamic-activation design; the 1D part
  reproduces known conclusions.

---

## Supplement 2: BRICK 2D Topology Block Ablation (2026-11-25, real 1B)

Unlike the 1D sequential ablation above (where later layers can compensate for a skipped
layer), here we measure dynamic-activation loss in the actual BRICK 2D topology
(8 + [8,6] + 2 parallel branches + fusion).

| Mode | CE | Delta CE vs both |
|---|---|---|
| both (both branches) | 3.79 | 0 |
| branch1 only (skip branch0) | 4.10 | +0.31 |
| branch0 only (skip branch1) | 4.14 | +0.36 |
| none (skip both) | 4.47 | +0.68 |
| **skip prelude 4-7** | **15.66** | **+11.87 (catastrophic)** |

**Key findings**:
1. **Prelude is a hard bottleneck**: skipping prelude layers 4-7 sends CE from 3.79 to
   15.66. Early layers encode the input into a representation later layers can use;
   breaking it collapses everything — consistent with the 1D layer0/4 cornerstone result.
2. **Branch dynamic activation has a moderate cost**: skipping one branch costs +0.31-0.36,
   both +0.68. Compared with 1D's +0.07, the 2D cost is **4-5x larger**, because in 1D
   later layers compensate but in 2D branches are parallel and the coda has only 2 layers.
3. **branch1 only (skip branch0) is slightly better** than branch0 only — the deeper
   branch (8 layers) is more skippable than the shallow one (6 layers), consistent with
   Amoeba's last-layer dominance and BitSkip's final-layer cliff.

**Design rule (from ablation, not guesswork)**:
- prelude: must execute fully, excluded from dynamic activation
- coda: must be kept (readout interface)
- only the branch pair is the legal dynamic-activation scope — A1 router / A2 segment's domain

---

## Router Improvement: Ablation-Aware Oracle (2026-11-25)

**Motivation**: prior oracles used "which branch representation is closer to teacher" or
"which branch has lower CE" — neither is true early-exit logic. Switched to
**ablation-aware oracle**: for each token compute the CE cost of skipping branch0 vs
branch1, skip the cheaper one (activate the other) — `brick_router.py::ablation_oracle`.

**Tiny result**: all three oracle labels are balanced; end-to-end CE nearly identical
(6.928-6.931) — branch differences too small under random weights; tiny cannot discriminate.

**Real 1B result (key)**:
- all three oracle labels biased to the deep branch ~[0.23, 0.77]
- ablation CE: both=4.219, skip_b0=4.999, skip_b1=4.920
  -> **any "skip one branch" combination costs ~0.7-0.8 above both**
- all three routers converge to the same end-to-end CE (4.9992) because they all degrade
  to "mostly choose b1"; CE dominated by skip_b0.

**Core conclusion (honest negative result)**:
> With an **untrained fusion**, dynamic activation — no matter what the router chooses —
> costs ~0.7-0.8 CE above "both branches". **The benefit of dynamic activation can only
> materialize after fusion is trained** — before that, both is the only reasonable choice.

This corrects the earlier optimistic assumption: the 2D ablation (+0.31) used an h_pre-
placeheld, untrained fusion, overestimating feasibility. Fusion must be trained first
(A0, LR=3e-5), then the router, for a fair evaluation.

**Corrected pipeline**: `build -> train_fusion (A0) -> freeze fusion -> ablation_oracle ->
train_router (Gumbel) -> end-to-end CE`. `train_router.py` now loads fusion before router
training; ablation oracle is integrated as the default (`--oracle ablation`).

---

## Decisive Experiment: BRICK Full Chain (2026-11-25, real 1B)

Pipeline: teacher data -> A0 fusion 50 steps (LR=3e-5) -> freeze -> ablation oracle ->
Gumbel router 50 steps -> end-to-end CE.

| Stage | Result |
|---|---|
| both CE (before fusion) | 4.219 |
| fusion MSE | 1.42 -> 0.03 (converged in 50 steps) |
| **both CE (after fusion)** | **4.041** (fusion works) |
| skip_b0 / skip_b1 CE | 4.44 / 4.83 |
| router distribution | [0.09, 0.91] (still deep-branch biased) |
| **router end-to-end CE** | **4.4435 (Delta vs both = +0.40)** |

**Conclusion (honest negative result)**:
> Training fusion reduces the both-branches CE from 4.219 to 4.041, but dynamic activation
> (running only one branch) still costs +0.40 CE above both — beyond the +0.3 positive
> threshold. Reasons: (1) the router stays biased on the real model (Gumbel's balancing
> works on tiny, fails on real 1B); (2) more fundamentally, **the skipped branch's
> information is genuinely missing and fusion cannot reconstruct it** — fusion training
> fixes the "merging" loss, not the "branch-skipping" information loss.

**This is BRICK's central boundary**: with a frozen backbone and 2D parallel branches,
dynamic activation has an irreducible CE cost.

**Remedies**:
- **Remedy 1 (falsified, 2026-11-25)**: expose fusion to single-branch inputs during
  training (test_cpu_1b_fusion_single.py: both-only / mixed / single-only).
  Results: A(both-only) Delta=+0.89, B(mixed) Delta=+1.02, C(single-only) Delta=+0.99 —
  **exposure makes it worse**, and both CE itself degrades (4.267->4.420). Root cause:
  missing information is truly gone; forcing fusion to "guess" both fails to guess and
  pollutes normal fusion. This path is closed.
- Remedy 2 (unverified): drop branch skipping, use token-weighted fusion with both
  branches evaluated (A6 depth interpolation) — preserves capability but saves no compute.
- Remedy 3 (unverified): finer granularity — skip sub-blocks inside a branch (based on
  2D ablation's b0_early/b1_late), possibly less information loss than a whole branch.

---

## Roadmap (by priority)

```
P0  A0 residual-aligned fusion    -> implemented
P0  A1 Oracle router              -> implemented + Gumbel fix
    + ablation-aware oracle        -> implemented, integrated as default
P1  A2 Segment routing             -> KV-consistency fallback
P1  A4 Progressive growing         -> fallback if A0 collapses
P2  A3 Frozen Joint-KV             -> most valuable acceleration mechanism
P2  A5 Branch specialization       -> interpretability result
P3  A6 Depth interpolation         -> multi-budget inference
Rejected A7 Confidence-gated coda (v2 lesson)
```

## Relation to Amoeba's negative results (must be in the report)

- **v2 confidence layering failure** -> motivated A1 (teacher oracle instead of
  self-confidence)
- **v3 forward-KL collapse** (alpha2 0.469->0.020) -> avoid forward-KL for router training
- **v4 KV-prediction slowness** (decode 0.973x) -> A3 must measure end-to-end speed
- **K=8 low acceptance** -> do not be greedy with branch count; 2 branches is the start
- **last layer dominates (~1.3 nats)** -> coda is untouchable (first of the three constraints)
