# -*- coding: utf-8 -*-
"""A6: Depth-Coordinate Interpolation —— 连续深度插值融合。

核心思想：把分支的"有效深度"当作连续坐标，用一个目标深度预算控制
分支权重，实现单 checkpoint 多档推理（类似 LiFT / Accordion）。

分支0 = 15层（深），分支1 = 9层（浅）→ 深度坐标 d0=15, d1=9。
目标深度 β ∈ [9,15] 时，权重 = softmax(-|d_k - β| / τ)。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthInterpolationFusion(nn.Module):
    """按目标深度对分支输出加权（A0 残差对齐 + 深度条件）。

    h_out = h_pre + alpha * Σ_k w_k(β) · proj_k(branch_k)
    """

    def __init__(self, hidden: int, branch_depths: list, temperature: float = 2.0):
        super().__init__()
        self.k = len(branch_depths)
        self.depths = torch.tensor(branch_depths, dtype=torch.float32)
        self.tau = temperature
        # 每个分支独立投影（比 A0 的单一 proj 更 expressive）
        self.projs = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(self.k)])
        self.alpha = nn.Parameter(torch.zeros(1))       # 只 alpha 零初始化
        # projs 保持正常初始化（否则 alpha 梯度为 0，训练无法启动）

    def weights(self, beta):
        """给定目标深度 β（标量或 [B]），返回 [B,K] 权重。"""
        if not torch.is_tensor(beta):
            beta = torch.tensor(float(beta))
        if beta.dim() == 0:
            beta = beta.reshape(1)
        beta = beta.to(self.depths.device).float()
        d = self.depths.to(beta.device)
        # 深度越接近 β 权重越高；[B,1] - [K] -> [B,K]
        logits = -(beta.unsqueeze(-1) - d).abs() / self.tau
        return F.softmax(logits, dim=-1)                # [B,K]

    def forward(self, h_pre, branch_outs, beta):
        w = self.weights(beta)                          # [B,K]
        fused = sum(w[:, k:k+1] * self.projs[k](bo.float()) for k, bo in enumerate(branch_outs))
        return h_pre + self.alpha * fused.to(h_pre.dtype)


class BudgetConditionedRouter(nn.Module):
    """A6 配套：token 级路由器，根据请求预算 β 决定有效深度。

    与 A1 router 的区别：额外输入 β（目标深度预算），实现"一个 checkpoint 多档"。
    """

    def __init__(self, hidden: int, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 1, bottleneck),           # +1 for beta
            nn.ReLU(),
            nn.Linear(bottleneck, 1),                    # 输出连续深度偏好
        )

    def forward(self, h, beta):
        B, T, H = h.shape
        b = torch.full((B, T, 1), float(beta), device=h.device)
        return self.net(torch.cat([h.float(), b], dim=-1)).squeeze(-1)  # [B,T]
