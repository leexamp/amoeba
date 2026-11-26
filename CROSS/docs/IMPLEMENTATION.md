# CROSS — Implementation Notes (2026-11-26)

## The selector question

A user asked: "Does the selector choose different layers itself? If so, efficiency
should improve." This led to separating two very different notions of "top-k" that
were previously conflated.

### Two kinds of top-k

1. **Fusion top-k** (what CROSS had): the router sets weights of k of K programs
   non-zero, but **all K programs are still executed**. This only cleans up the
   fusion — it saves no compute.
2. **Execution top-k** (the new mechanism): a selector predicts which k programs to
   run **before** executing any, so only k programs are executed. This is a real
   compute saving.

PoLar selects exactly 1 program to execute. CROSS with execution top-k=2 selects 2
programs and fuses them — one more than PoLar (buying fusion benefits), half of
full CROSS's 4 (saving compute).

## Code changes

### `cross_selector.py` (new)
- `ProgramSelector`: lightweight MLP from a pooled prelude hidden state `[B,H]` to
  K program logits `[B,K]`. `choose()` returns hard top-k indices for inference.
- `distill_loss()`: the selector is **not** trained by backprop through hard top-k
  (non-differentiable). Instead it is distilled from the trained full-CROSS router:
  target = normalized mean fusion weight per program; loss = cross-entropy on the
  argmax program. This is the same "teacher signal, not self-confidence" principle
  from Amoeba/BRICK.

### `cross_model.py` (modified)
- `CROSSModel(..., select_before_execute=True, selector_top_k=2)` enables execution
  top-k.
- New forward path when `select_before_execute`:
  1. Run only the first `min(4, n_layers)` layers to get a cheap prelude hidden.
  2. `selector(sel_h.mean(dim=1))` -> program logits -> `topk(selector_top_k)`.
  3. Execute **only the selected programs** (`idxs`).
  4. Pad program outputs with zero tensors to keep the router's expected K inputs
     (zero padding gets ~0 weight after top-k sparsification, so it does not affect
     fusion).
  5. Same cross-router fusion and LM head as before.
- The default path (all K programs) is unchanged.

### Why hard top-k in forward is fine
- At inference we intentionally use non-differentiable hard selection (we want to
  actually skip computation).
- Trainability of the selector comes from `distill_loss`, not from the forward path.
- Tiny test verifies: (a) select-before-execute forward runs and produces logits;
  (b) selector gets gradients via distill loss; (c) hard selection saves 58% of
  layer-executions on the tiny model.

### dtype consistency (three bugs fixed on real 1B)
Real 1B is bf16. Freshly constructed nn.Modules default to fp32, so dtype mismatches
appeared in three places, each causing "mat1 and mat2 must have the same dtype":
1. `ProgramSelector.forward` had a hardcoded `h_pre.float()` — removed; dtype now
   follows module weights.
2. `CrossRouter.forward` had `self.net(x.float())` — removed for the same reason.
3. `CROSSModel.__init__` now casts `self.router`/`self.selector` to `base.dtype`, and
   `test_1b_selector.py` re-casts after `load_state_dict` (loading can restore fp32).
This is the same class of bug found in BRICK's fusion on the real 1B model: tiny
models (fp32) never reveal it; real bf16 models crash immediately. All tests now
pass on both tiny (fp32) and real 1B (bf16) paths.

## Test scripts
- `test_selector_tiny.py`: forward, distill gradients, compute saving. 3/3 PASS.
- `test_1b_selector.py`: three stages —
  Stage 1: train full CROSS router (6 steps, tiny sample);
  Stage 2: distill selector (4 steps);
  Stage 3: compare full vs select-before-execute CE and layer-execution counts.
  Running on CPU (slow: 4 programs x 24 layers per step).

## Expected trade-off
- Compute: execution top-k=2 of K=4 should save roughly 50-60% of layer executions.
- Capability: CE may rise because 2 programs are dropped. The experiment measures
  the delta. If delta < 0.3 nats, the selector is a viable efficiency knob; if not,
  report the boundary honestly (Amoeba/BRICK tradition).
