# CROSS

**C**ross **R**outer **O**f **S**elector **S**ystem

> **Paper**: `CROSS_PAPER.pdf` / `.tex` / `.docx` — 4-page honest report
> (positive fusion result + negative selector result). Draft v0.1, 2026-11-26.

A follow-up to Amoeba and BRICK. Program-level MoE fusion over a frozen LLM:
generate several candidate layer-execution programs, execute each independently,
and fuse their outputs with a token-level cross router.

> **Positioning**: PoLar (ICML 2026) already solved *single*-program skip/repeat
> selection. MoE work (MoRE/MoUE/UniPool) shares experts *within* layers. CROSS
> targets the intersection nobody covers: **MoE fusion over multiple program-level
> outputs**. The goal is capability, not acceleration. Dense-to-MoE conversion
> itself is NOT novel (CMoE, ToMoE, ExpertWeaver already do it at the FFN-neuron
> level); CROSS claims only the *program-as-expert* variant — an entire
> skip/repeat layer-execution program is one expert.

All documents in this project are English-only.

## Idea

A pretrained LLM with D layers defines D functions f_0..f_{D-1}. PoLar shows that
many input-specific execution programs π=(i_1..i_K) yield correct predictions, often
shorter than the standard forward pass. Its pass@k reports "one of the k programs is
right" — but it still returns a *single* program's output.

CROSS asks: if several programs are valid and complementary, can we **fuse their
outputs** to get a better answer than any single one?

```
        +-- program 0: skip L4-7,  keep others  --> h_0 --+
prompt -+-- program 1: repeat L8-11, keep others --> h_1 --+-- Cross Router (top-k, token-level) --> h_fused --> LM head
        +-- program 2: skip L12-15, repeat L20   --> h_2 --+
```

The *selector* generates a pool of candidate programs; the *cross router* selects
and weights program outputs per token. Both are lightweight; the D-layer backbone
is frozen.

## Why this is not BRICK again

BRICK failed because skipping a branch removed information that fusion could not
reconstruct (+0.40 nats irreducible loss). CROSS does **not** skip information at
readout time: every candidate program processes the full input and produces a full
hidden state; fusion combines several complete representations. The compute cost is
higher (K program executions), but the objective is capability, not speed.

## Design constraints

1. Backbone frozen; only selector + cross router trained (<10M params).
2. Readout final layer always executed in every program (Amoeba/BRICK finding:
   final layer is the readout interface).
3. Program pool is PoLar-style contiguous segments with {skip, keep, repeat}.
4. Router is Gumbel-Softmax / top-k to avoid branch collapse (BRICK lesson).
5. Evaluate with Amoeba's criterion: capability preservation (79-question accuracy,
   zero regressions), not bit-exactness.

## Results (real 1B CPU) — claim revised after held-out check

**In-sample result REJECTED**: fusion CE 4.755 vs best single 4.840 (Delta -0.085)
was measured on the same 2-4 windows used for training, with 492K params / 32-64
supervised tokens. This is fitting, not evidence of complementarity.

**Held-out gatekeeper (train 0-2, test 3-5)**:

| Split | P0 identity | oracle-min (labelled) | hidden fusion | prob fusion |
|---|---|---|---|---|
| train | 4.249 | 3.969 | 4.382 | 3.856 |
| **held-out** | **3.847** | 3.056 | **3.428** | **3.061** |

Both fusion levels beat identity on held-out (Delta -0.42 / -0.79). BUT held-out
weights collapse to a single program ([0,0,0,1] hidden, [0,1,0,0] prob): the gain is
**selecting a program better than identity**, NOT complementary fusion. Program
ranking flips between splits (P0 best on train, P1/P3 best on held-out).
Probability fusion (3.061) ≈ oracle (3.056), consistent with Jensen's guarantee.

**COMPLEMENTARITY CLAIM WITHDRAWN.** Supported: frozen models admit programs better
than identity on unseen inputs, and a router learns to select them. Cost is 4x
program executions — capability study, not acceleration. Tiny tests 5/5 PASS.

## Selector result (2026-11-26, negative)

Execution top-k saves ~50-55% layer-executions but CE rises +0.28 to +0.93 across
two runs (6- vs 10-step teacher). The selector perfectly distills the teacher
(agree=1.00), but the under-trained teacher's program ranking is unstable: one run
selected [1,2] (dropped identity), another [0,1]. Forcing P0 in improves CE by 0.09
but does not close the gap. The 3x swing in CE delta with a 6-vs-10-step budget is
itself a finding: execution top-k is not viable with a lightly trained teacher.
See `docs/RESULTS.md`.

## Status

- 2026-11-25: project proposed; program-level MoE intersection appears uncovered.
  H1 supported on real 1B (fusion beats best single program, -0.085).
- 2026-11-26: selector (execution top-k) implemented + first 1B test (negative above).
- Next: more training for the teacher router, then re-test selector; 79-question
  capability eval on GPU.
