# -*- coding: utf-8 -*-
"""CROSS PoC configuration."""
import os

# Real model (remote GPU path) / local fallback
MODEL_REMOTE = "/data/work/ckpt_1b_fixed/full_model"
MODEL_LOCAL  = r"E:\AMOEBA_MODELS\MiniCPM5-1B"
TOKENIZER_REMOTE = "/data/models/MiniCPM5-1B"
CORPUS_REMOTE = "/data/work/chat_corpus.pt"
CORPUS_LOCAL = r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
WORK = "/data/work/cross"
CKPT = os.path.join(WORK, "cross_router.pt")

# Architecture
N_LAYERS = 24          # MiniCPM5-1B
K_PROGRAMS = 4         # candidate programs per token
TOP_K = 2              # top-k program fusion
KMAX_SEG = 4           # max segment length (PoLar style)
KEEP_FINAL = True      # final layer always executed (readout interface)

# Router
ROUTER_HIDDEN = 256
LR = 3e-4
STEPS = 1500
BATCH = 2

# Evaluation
SEQ = 16
NWIN = 4
GEN_TEMP = 0.8
GEN_TOP_P = 0.95
GEN_REP_PEN = 1.05
MAX_NEW = 160

def default_programs(n_layers=N_LAYERS, k=K_PROGRAMS):
    """Fixed diverse program pool (v1): each is a list of layer indices to execute.

    PoLar-style operations are simplified to 'skip a contiguous segment' or
    'repeat a contiguous segment'. Every program keeps the final layer.
    """
    full = list(range(n_layers))
    mid = n_layers // 2
    pool = []
    # P0: identity (standard forward)
    pool.append(full[:])
    # P1: skip a middle segment (quarter length)
    seg = mid // 2
    pool.append([i for i in full if not (seg <= i < seg + max(2, seg))])
    # P2: repeat a middle segment once
    r0, r1 = mid - 2, mid + 2
    pool.append(full[:r1] + list(range(r0, r1)) + list(range(r1, n_layers)))
    # P3: skip a late segment + repeat an early segment
    s0, s1 = int(n_layers * 0.7), int(n_layers * 0.8)
    kept = [i for i in full if not (s0 <= i < s1)]
    cut = n_layers // 4
    pool.append(kept[:cut] + list(range(1, 3)) + kept[cut:])
    assert len(pool) >= k
    return [p for p in pool[:k]]

if __name__ == "__main__":
    for i, p in enumerate(default_programs()):
        print(f"P{i}: {len(p)} executions, layers {p[:6]}...{p[-3:]}")
