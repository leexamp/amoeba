# -*- coding: utf-8 -*-
"""CROSS Selector: predict which programs to execute BEFORE running them.

Key idea (user's insight): top-k in the router only sparsifies fusion weights;
it does NOT save compute because all K programs are still executed. A selector
that predicts the top-k program indices from the input lets us execute only k
programs, cutting compute from Kx to kx while keeping multi-program fusion.

Training: distill from the full CROSS router's fusion weights. The selector
learns to predict which programs the trained router considers important.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgramSelector(nn.Module):
    """Predict per-token program importance from a cheap input representation.

    Uses only the embedding/prelude hidden (no full program execution), so its
    cost is negligible compared with running K programs.
    """
    def __init__(self, hidden, k, bottleneck=64):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, k),
        )

    def forward(self, h_pre):
        return self.net(h_pre)                # [B,T,K] logits; dtype follows module weights

    def choose(self, h_pre, top_k=2):
        """Hard top-k program indices [B,T,top_k] (for inference)."""
        logits = self.forward(h_pre)
        return torch.topk(logits, top_k, dim=-1).indices

    def distill_loss(self, h_pre, router_weights, top_k=2):
        """Learn to predict which programs the full router actually uses.

        h_pre: [B,1,H] pooled prelude hidden (or [B,T,H]); router_weights: [B,K].
        Target = normalized router weights; the selector learns the argmax program.
        """
        if h_pre.dim() == 3:
            h_pre = h_pre.mean(dim=1)                 # [B,H]
        if router_weights.dim() == 3:
            router_weights = router_weights.mean(dim=1)  # [B,K]
        soft = router_weights.detach()
        soft = soft / (soft.sum(-1, keepdim=True) + 1e-10)
        logits = self.forward(h_pre)                   # [B,K]
        return F.cross_entropy(logits, soft.argmax(-1)), \
               (logits.argmax(-1) == soft.argmax(-1)).float().mean().item()
