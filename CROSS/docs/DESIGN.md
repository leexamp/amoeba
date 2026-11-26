# CROSS — Technical Design (2026-11-25)

## Full name
**C**ross **R**outer **O**f **S**elector **S**ystem

## Research question
Can a token-level MoE router fuse the outputs of *multiple* layer-execution programs
to improve capability beyond any single program, on a frozen LLM?

## Prior work and the gap
- **PoLar** (Li et al., ICML 2026, arXiv:2606.06574): frozen layers as a function
  library; a lightweight predictor generates one skip/repeat program per input;
  pass@k over k predicted programs; 0.83x–0.95x latency with +5.7–9.4 accuracy.
  *Gap*: returns a single program's output; no fusion of program outputs.
- **MoS / Mixture of States** (arXiv:2511.12207): mixes hidden states from different
  layers for multimodal generation. *Gap*: not LLM reasoning, not program-level.
- **MoRE / ReXMoE / MoUE / UniPool** (2025–2026): expert sharing across layers,
  *within* the MoE layer. *Gap*: not program-level outputs.
- **Attentive Multi-Layer Fusion** (ICML 2026): layer fusion in vision transformers.
- **Amoeba / BRICK** (our prior work): capability criterion, 2D topology negative
  result, Gumbel router, oracle lessons.

The uncovered intersection: **token-level MoE fusion over multiple full program
executions**. No novelty claimed for programs, MoE, routers, or fusion individually;
the contribution is the intersection and its empirical capability boundary.

## Architecture

### Selector (program pool)
PoLar-style representation: contiguous layer segments with per-segment operations
{skip, keep, repeat}, Kmax=4. Generate K candidate programs per input either by
(a) beam search over a lightweight predictor, or (b) a fixed diverse pool
(skip different segments / repeat different segments) for the first PoC.

### Executors
Each program π_j is executed independently on the same input tokens:
```
h_j = f_{π_j}(embed(x))       # full hidden state after the program
```
All programs keep the final layer (readout interface). Because programs may skip
different layers, each execution uses its own KV cache; no cross-program cache sharing
in v1.

### Cross Router (the CROSS contribution)
Token-level router over K program outputs:
```
w_j(t) = softmax( router(h_1(t),...,h_K(t)) )[j]      # top-k sparsified
h_fused(t) = sum_j w_j(t) * proj_j(h_j(t))
logits = lm_head(norm(h_fused))
```
Trained end-to-end with next-token CE on frozen backbone. Gumbel-Softmax or
entropy regularization to prevent collapse to one program (BRICK lesson).

### Training signal
- Primary: next-token CE of fused output.
- Auxiliary: per-program CE (encourages diverse competent programs).
- Oracle: programs that individually solve the token get higher weight — inspired by
  PoLar's "valid programs" and BRICK's ablation-aware oracle.

## Hypotheses
H1: fusing K programs improves CE over the best single program (complementarity).
H2: gains concentrate on hard tokens where programs disagree.
H3: top-k fusion (k=2 of K=4) captures most of the gain with bounded cost.

## PoC plan (CPU first, then GPU)
1. Tiny Llama (12 layers) with synthetic programs: verify router gradients, top-k,
   no collapse.
2. Real MiniCPM5-1B (24 layers) with a fixed diverse program pool: measure CE of
   each program and of fused output on chat corpus windows.
3. If H1 holds, train selector + router and evaluate 79-question capability with
   zero-regression criterion.
4. GPU needed for full training; CPU used for boundary/trend (as in BRICK).

## First result: H1 SUPPORTED (2026-11-25, real 1B, CPU)

| Metric | Value |
|---|---|
| 4 program single CEs | [5.26, 5.61, 5.41, 5.49] |
| best single (per-window oracle) | 4.840 |
| untrained fusion | 5.451 (worse than every single program) |
| **trained fusion (12 steps)** | **4.755** |
| **Delta vs best single** | **-0.085** |
| weight distribution | [0.17, 0.05, 0.25, 0.53] (no collapse) |
| trainable params | 492K |

**H1 holds**: fusing multiple program outputs beats any single program. This is what
PoLar's pass@k cannot do (it selects one program, CROSS fuses several). It also
contrasts sharply with BRICK: there, skipping a branch lost information fusion could
not recover (+0.40 nats); here, every program executes fully so information is complete
and fusion adds complementary signal.

Honest caveats: 2 windows, 12 steps — a trend, not final accuracy; "best single" is a
per-window oracle the router cannot know; the gain is small (-0.085) and the cost is
4x program executions. This is a capability study, not acceleration.

## Risks / honest boundaries
- Cost: K program executions => Kx compute. This is a capability study, not acceleration.
- If programs are highly correlated, fusion reduces to the single program (no gain).
- KV cache and position handling must remain causal per program.
- If fusion cannot beat best single program, report negative result (Amoeba tradition).
