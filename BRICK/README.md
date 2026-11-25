# BRICK

**B**ranch-**R**ewired **I**nference with **C**omposable **K**nowledge

> **Paper**: `BRICK_PAPER.pdf` / `.tex` / `.docx` — 5-page honest empirical report
> (2D topology negative result, full ablation chain, no novelty claims). Draft v0.1, 2026-11-25.
>
> This project is now archived inside the Amoeba repository as a follow-up study.
> Amoeba's paper and release package are unchanged.

Frozen pretrained small models, 2D branch topology, learned lightweight fusion —
an empirical study of whether dynamic activation works when the layer chain is rewired
from 1D into parallel heterogeneous branches. Motivated by the FinFET transition
(change the geometry, not the material) and by our earlier finding that many middle
layers of a 1B model have near-zero ablation cost.

## Central finding (negative result)

The 2D rearrangement is **not** capability-destroying — a trained fusion recovers the
"both-branches" CE from 4.22 to 4.04. But **skipping** a branch carries an irreducible
cost of +0.40–0.90 nats: the skipped branch's information is genuinely absent and
cannot be reconstructed. Exposing fusion to single-branch inputs during training makes
results worse, not better. Dynamic activation on a frozen backbone with 2D parallel
branches is therefore structurally bounded.

## Why LayerSkip works and this does not

LayerSkip (ACL 2024) and BitSkip (2026) train with layer dropout and an early-exit loss,
so intermediate representations become directly decodable. We have a frozen pretrained
backbone: redundancy exists under *ablation* (later layers compensate) but not under
*readout* (intermediate hidden states cannot be decoded). BRICK's learned fusion is a
lightweight attempt to bridge that gap — it succeeds for fusion, fails for branch skipping.

## Topology (MiniCPM5-1B, 24 layers, hidden 1536)

```
                  +-- branch0: L8..L15 (8 layers) --+
L0..L7 (prelude) --+                                 +-- Fusion -- L22..L23 (coda) -- LM head
                  +-- branch1: L16..L21 (6 layers) --+
```

Branches consume the same prelude output and run in parallel. Each uses distinct
original layer indices, so the 24 KV-cache slots are each written once per token —
no cache conflict. The coda always keeps the final layer (readout interface).

## Key experimental results (real 1B, CPU)

| Experiment | Result |
|---|---|
| Both-branches CE, fusion before / after 50 steps | 4.219 -> **4.041** (fusion works) |
| Skip-branch CE (ablation) | both 4.04, skip_b0 4.44, skip_b1 4.83 |
| Full chain: router end-to-end CE | **4.4435** (Delta +0.40 vs both) — negative |
| Fusion trained on single-branch inputs | Delta +0.89 -> +1.02 (worse) — remedy falsified |
| Gumbel router distribution | [0.09, 0.91] on real 1B (balances only on tiny) |
| 1D layer ablation | U-shaped: L0 +6.51, L4 +5.62, L22 +5.02, L23 +3.62; L5-21 ~0 |
| 2D block ablation | skip prelude L4-7: CE 3.79 -> 15.66 (catastrophic); skip a branch: +0.31-0.36 |

Design rule from ablation: prelude and coda must be fully executed; only the branch
pair is a legal dynamic-activation scope.

## Code

```
code/
  brick_config.py        config: 42/24-layer topology, fusion mode, all hyperparameters
  layer_utils.py         transformers 4.44~5.24 layer-call compatibility (position_embeddings!)
  brick_model.py         BrickModel 2D forward + BrickFusion (residual_branch_avg)
  brick_router.py        oracle routers + ablation-aware oracle + oracle diagnostics
  brick_router_gumbel.py Gumbel-Softmax router (temperature annealing + straight-through)
  brick_kv.py            A3 Frozen Joint-KV (LoRA K/V projections, not integrated)
  brick_depth.py         A6 depth interpolation (not integrated)
  build_brick_data.py    teacher-supervision generation from Amoeba on-policy chat corpus
  train_fusion.py        A0 fusion training (MSE + CE)
  train_router.py        A1/A2/A5 router training (CE / ablation oracle, Gumbel default)
  train_progressive.py   A4 progressive branch growing (fallback)
  eval_brick.py          79-question capability evaluation + speed benchmark
  run_all.sh             full pipeline (GPU)
  test_cpu_toy.py        tiny-model test, 7/7 PASS on transformers 5.24.1
  test_cpu_*.py          CPU 1B experiments (ablation, router, fusion, full chain)
  toy_test.py            local topology self-check (no torch needed)
```

Full algorithm designs, all negative results, and the positioning relative to LayerSkip /
Curse of Depth / MBT / DAGFormer are in `ALGORITHMS.md`.

## Verification status

- 27 Python files: all pass `py_compile`; toy test 7/7 PASS on real transformers 5.24.1.
- Seven bugs that would have wasted a remote GPU run were found and fixed locally:
  (1) 5.x layer-call signature requires `position_embeddings=(cos,sin)`;
  (2) double zero-init gives zero gradient; (3) representation-distance oracle collapses to
  the deep branch; (4) softmax argmax router collapses (Gumbel fix);
  (5) fp32/bf16 dtype mismatch on the real model; (6) wrong residual anchor
  (h_pre -> branch mean); (7) LR too high (3e-4 -> 3e-5).
- Real 1B CPU verification: layer-call equivalence diff=0, BRICK forward + generate OK.
- Full convergence and 79-question evaluation need a GPU; all reported numbers are trends
  and boundaries, not final accuracy.

## License

All code and reports Apache-2.0. No LayerSkip code copied; ideas implemented from papers.
All published mechanisms cited, no novelty claims. Independent research, not affiliated
with any base-model provider.
