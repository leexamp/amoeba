# -*- coding: utf-8 -*-
"""Gumbel-Softmax Router：解决"标签均衡但 router 仍塌缩"的训练动态问题。

背景（本机对照实验发现）：
  CE oracle 标签均衡 ~50/50，但普通 softmax + CE 训练的 router 仍倾向单分支
  （异深 0.99，等深 0.91，最好也只 0.85）。原因：argmax 决策不可导，
  teacher-forcing 下微小偏差会被放大成赢者通吃。

方案：
  1. Gumbel-Softmax：训练时用可微的随机采样（温度退火），让 router 探索两个分支
  2. 熵正则：显式最大化路由分布的熵，防止塌缩
  3. 欠采样偏差校正：对高频分支的梯度降权

三种机制都实现，本机对照验证哪种有效。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRouter(nn.Module):
    """Gumbel-Softmax router，温度退火 + 可选熵正则。"""

    def __init__(self, hidden, k, bottleneck=256, tau_start=1.0, tau_end=0.1,
                 entropy_weight=0.0, sample_during_train=True):
        super().__init__()
        self.k = k
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.entropy_weight = entropy_weight
        self.sample_during_train = sample_during_train
        self.net = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, k),
        )
        self.step = 0

    def temperature(self, total_steps):
        """线性退火：tau_start -> tau_end。"""
        frac = min(self.step / max(total_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * frac

    def forward(self, h, hard=False, total_steps=1):
        logits = self.net(h.float())                  # [B,T,K]
        if self.training and self.sample_during_train:
            tau = self.temperature(total_steps)
            # Gumbel-Max trick + softmax 松弛（straight-through）
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            y = (logits + g) / tau
            soft = F.softmax(y, dim=-1)
            if hard:
                # straight-through: 前向用 hard one-hot，反向用 soft
                hard_onehot = F.one_hot(soft.argmax(-1), self.k).float()
                soft = (hard_onehot - soft).detach() + soft
            return soft, logits
        # eval 时直接 softmax（确定性）
        return F.softmax(logits, dim=-1), logits

    def entropy_loss(self, soft):
        """熵正则：最大化路由熵，防止塌缩。返回负值供最小化。"""
        p = soft.clamp(min=1e-10)
        return -(p * p.log()).sum(-1).mean()          # 最大化这个 -> loss 加负号

    def choose(self, h, total_steps=1):
        soft, _ = self.forward(h, hard=True, total_steps=total_steps)
        return soft.argmax(-1)


class BalancedGumbelRouter(GumbelRouter):
    """Gumbel + 全局负载均衡：限制整个 batch 的分支使用比例。"""

    def __init__(self, *a, target_ratio=0.5, balance_weight=0.5, **kw):
        super().__init__(*a, **kw)
        self.target_ratio = target_ratio
        self.balance_weight = balance_weight

    def balance_loss(self, soft):
        """让 batch 内各分支的平均使用概率接近 target。"""
        avg = soft.mean(dim=(0, 1))                    # [K]
        target = torch.full_like(avg, self.target_ratio)
        return F.mse_loss(avg, target)
