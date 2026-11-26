# CROSS — Experimental Results Log (all English)

## 2026-11-25: H1 — multi-program fusion beats the best single program

**Setup**: real MiniCPM5-1B (24 layers, hidden 1536), CPU bf16. Four fixed diverse
programs (identity / skip middle segment / repeat middle segment / skip+repeat),
K=4, top_k=2 fusion. 2 windows, seq 16, 12 training steps.

| Metric | Value |
|---|---|
| 4 single-program CEs | [5.2605, 5.6062, 5.4077, 5.4858] |
| best single (per-window oracle) | 4.8400 |
| untrained fusion CE | 5.4506 (worse than every single program) |
| **trained fusion CE (12 steps)** | **4.7548** |
| **Delta vs best single** | **-0.0853** |
| fusion weight distribution | [0.166, 0.054, 0.249, 0.531] (no collapse) |
| trainable parameters | 492,172 |

**Conclusion (REVISED — in-sample only)**: fusing multiple program outputs beats
any single program on the \emph{training windows}. \textbf{This does NOT establish
complementarity}: 492K trainable parameters were fit on 32--64 supervised tokens,
and both the fusion and the per-window oracle used labels from the same windows.
The in-sample delta of -0.085 is consistent with fitting.

**Held-out gatekeeper (2026-11-26, COMPLETED — claim revised)**:
train windows 0--2, held-out 3--5, 6 steps, hidden + probability fusion.

| Split | P0 identity | oracle-min (labelled) | hidden fusion | prob fusion |
|---|---|---|---|---|
| train | 4.249 | 3.969 | 4.382 | 3.856 |
| **held-out** | **3.847** | 3.056 | **3.428** | **3.061** |

Verdict: both fusion levels beat identity on held-out (−0.42 / −0.79). BUT held-out
fusion weights collapse to a single program ([0,0,0,1] hidden, [0,1,0,0] prob).
The gain is **selecting a program better than identity**, NOT fusing complementary
programs. Program ranking flips between splits: P0 best on train, P1/P3 best on
held-out. Probability fusion (3.061) ≈ oracle (3.056), consistent with Jensen.

**COMPLEMENTARITY CLAIM WITHDRAWN.** Supported statement: frozen models admit
programs beating identity on unseen inputs, and a lightweight router selects them.
This is what the AI reviewer's gatekeeper check was designed to catch.

Other caveats: labelled oracle upper bound; cost K program executions — capability
study, not acceleration.

## Final gatekeeper tests (2026-11-26, pending)
An external review identified three confounds that must be resolved before the
conclusions are final. `test_final_gates.py` runs all three with cached program
hidden states (backbone executed once, then all tests are cheap):

1. **Fixed-weight fusion comparison — COMPLETED (decisive)**: prior held-out
   weights were one-hot, so the hidden-vs-prob gap was selection, not fusion.
   Clean test with fixed weights on 20 windows:

   | weights | hidden CE | prob CE | Jensen bound | best single |
   |---|---|---|---|---|
   | uniform [.25]^4 | 3.827 | **3.719** | 3.966 | 3.880 |
   | in-sample [.17,.05,.25,.53] | 3.879 | **3.774** | 3.956 | 3.880 |

   prob <= Jensen bound in both; prob 3.719 < best single 3.880 — a real fusion
   gain with theoretical backing. The "fusion is empty" worry is resolved:
   fusion helps when weights are fixed; the earlier one-hot runs only selected.
2. **Held-out n — COMPLETED, claim withdrawn**: 14 held-out windows (210 tokens),
   paired per-token bootstrap. fusion 3.775, P0 3.850, oracle 3.240; paired
   fusion-P0 = -0.075, 95% CI [-0.156, +0.012] — **crosses zero**. The earlier
   3-window gap of -0.786 was sampling noise. "Programs beat identity" is withdrawn.
3. **Seed stability**: 5 seeds, check whether program selection is consistent
   (signal) or random (luck).

Results to be inserted. Conclusions (a) and (b) are held pending these tests.

## Selector force-P0 experiment (2026-11-26)

Re-running the selector with a different (10-step) teacher router:

| Config | selected | CE | layers | Delta vs full |
|---|---|---|---|---|
| no force | [0, 1] | 4.4881 | 42 | +0.3764 |
| force-P0 | [0, 1] | 4.3947 | 42 | +0.2830 |

Both still above full CROSS (4.112). force-P0 improves by 0.09 but does not flip
the negative result: execution top-k saves ~55% compute yet costs +0.28--0.38 CE.
The earlier selector run selected [1,2] with +0.93 CE; this one [0,1] with +0.28.
**The selector outcome is highly sensitive to the under-trained teacher's state** —
a 6-vs-10-step difference changed the selected programs and the CE delta by 3x.
This sensitivity is itself a finding: execution top-k is not viable with a lightly
trained teacher, and trend claims require reporting the teacher's training budget.

## 2026-11-26: Selector — from fusion top-k to execution top-k

**Question**: the fusion router's top_k only sparsifies weights; all K programs still
run. Can a selector choose programs *before* execution to save compute?

**Implementation**: see `IMPLEMENTATION.md`. Selector distilled from the trained
full-CROSS router weights (hard top-k is non-differentiable; distillation supplies
gradients).

**Tiny verification (3/3 PASS)**: forward runs, selector receives distill gradients,
hard selection saves 58% of layer-executions.

**Real 1B experiment (2026-11-26, completed)**:

| Metric | Value |
|---|---|
| Stage 2 distill agree | 1.00 (selector perfectly learns the router's program choice) |
| selected programs | [1, 2] |
| full CROSS CE | 4.5933 (93 layer-executions) |
| select-before-execute CE | 5.5278 (46 layer-executions) |
| CE delta (select - full) | **+0.9345** |
| compute saving | **50.5% fewer layer executions** |
| verdict | selector loses too much CE |

**Conclusion (negative)**: the selector perfectly learns which programs the full
router uses (agree=1.00) and cuts compute in half (50.5%), but CE rises by +0.93 nats.
Reason: programs [1,2] (skip-middle, repeat-middle) individually have higher CE than
program 0 (identity). The router's low weights on program 0 came from a 6-step,
2-window training signal — the router had not learned to value the identity program.
Selecting by an under-trained router therefore drops the best program.

Key lesson: execution top-k is only viable if the selector's teacher (the full router)
is well trained. With 6 steps / 2 windows the teacher itself is unreliable. This is a
trend/boundary result (not a final verdict) — more training may make the selector
choose program 0 and close the gap. Honest record kept, Amoeba/BRICK tradition.

## Project status
- Code: `cross_config.py`, `cross_router.py`, `cross_selector.py`, `cross_model.py`,
  `layer_utils.py`, `test_tiny.py`, `test_1b_h1.py`, `test_selector_tiny.py`,
  `test_1b_selector.py`.
- Documents: `README.md`, `docs/DESIGN.md`, `docs/IMPLEMENTATION.md`, this file.
- All Python files compile; all documents English-only.
