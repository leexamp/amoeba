# -*- coding: utf-8 -*-
"""CROSS Cross Router: token-level top-k MoE fusion over K program outputs."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRouter(nn.Module):
    """Router over K program hidden states, token-level.

    Compresses K*H -> bottleneck -> K logits (keeps params small: O(H*B) not O(H*K*H)).
    Fuses with per-program scalar scaling (no big projection matrices).
    """
    def __init__(self, hidden, k, bottleneck=64, top_k=2, gumbel=False, tau=1.0, fusion_level="hidden"):
        super().__init__()
        self.k, self.top_k = k, top_k
        self.gumbel = gumbel
        self.tau = tau
        # "hidden": fuse hidden states then apply lm_head (no Jensen guarantee)
        # "prob": fuse per-program next-token distributions (guarantees CE <= weighted member CE)
        assert fusion_level in ("hidden", "prob")
        self.fusion_level = fusion_level
        self.net = nn.Sequential(
            nn.Linear(hidden * k, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, k),
        )
        # scalar per-program scale, init 1 — no large proj matrices
        self.scale = nn.Parameter(torch.ones(k))

    def forward(self, program_outputs, temperature=None):
        x = torch.cat(program_outputs, dim=-1)
        logits = self.net(x)   # dtype follows module weights (bf16 on real models)
        tau = temperature if temperature is not None else self.tau
        if self.training and self.gumbel:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            w = F.softmax((logits + g) / tau, dim=-1)
        else:
            w = F.softmax(logits, dim=-1)

        if self.top_k < self.k:
            top = torch.topk(w, self.top_k, dim=-1)
            mask = torch.zeros_like(w).scatter_(-1, top.indices, 1.0)
            w = w * mask
            w = w / (w.sum(-1, keepdim=True) + 1e-10)

        # fuse: weighted sum of raw program outputs, with learned scalar scales
        fused = sum((w[..., j:j+1] * self.scale[j]).float() * program_outputs[j].float()
                    for j in range(self.k))
        return fused.to(program_outputs[0].dtype), w, logits

    def entropy(self, w):
        p = w.clamp(min=1e-10)
        return -(p * p.log()).sum(-1).mean()


class ProgramSelector(nn.Module):
    """Lightweight selector over prelude hidden state (v1: program weights)."""
    def __init__(self, hidden, k, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(), nn.Linear(bottleneck, k)
        )

    def forward(self, h_pre):
        return self.net(h_pre.float())
