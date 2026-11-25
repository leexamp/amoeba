# -*- coding: utf-8 -*-
"""BRICK Router 模块族：A1 Oracle / A2 Segment / A5 Specialization。

设计原则（来自 Amoeba 失败教训）：
- 不用模型自置信度（v2 失败）→ 用教师表示当 oracle 监督（A1）
- 不用 forward-KL（v3 崩溃）→ 用交叉熵拟合离散 oracle 标签
- token-level 路由有 cache 因果性问题 → A2 段级路由作为务实退路
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrickRouter(nn.Module):
    """A1: token-level router. h_pre -> 各分支概率 [B,T,K]."""

    def __init__(self, hidden: int, k: int, bottleneck: int = 256):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, k),
        )

    def forward(self, h):
        return self.net(h.float())                    # [B,T,K]

    def choose(self, h):
        """返回 [B,T] 的分支索引（argmax）。"""
        return self.forward(h).argmax(-1)


class SegmentRouter(BrickRouter):
    """A2: 段级路由。每 seg_len 个 token 决策一次，段内共享选择。

    解决的问题：token-level 切换分支会破坏 KV cache 因果性
    （不同分支用不同原始层索引，cache 槽写入不一致）。段级让每个段内
    所有 token 走同一分支，只在段边界做一次 KV 对齐。
    """

    def __init__(self, hidden, k, seg_len=16, bottleneck=256):
        super().__init__(hidden, k, bottleneck)
        self.seg_len = seg_len

    def forward(self, h):
        B, T, H = h.shape
        # 每段用首 token 的隐藏状态作为决策依据
        idx = torch.arange(0, T, self.seg_len, device=h.device)
        seg_h = h[:, idx, :]                          # [B, nseg, H]
        seg_logits = super().forward(seg_h)           # [B, nseg, K]
        # 广播回每个 token
        expand = torch.arange(T, device=h.device) // self.seg_len
        return seg_logits[:, expand, :]               # [B,T,K]


class SpecializedRouter(BrickRouter):
    """A5: 专业化路由 = A1 router + 可解释的负载均衡损失。

    目标：让分支按题型分化（数学/中文/事实），同时避免全塌缩到单分支。
    """

    def balance_loss(self, h, target_entropy=0.9):
        """鼓励路由分布接近均匀（防止塌缩）。target_entropy 为相对熵比例。"""
        probs = F.softmax(self.forward(h), dim=-1)    # [B,T,K]
        avg = probs.mean(dim=(0, 1))                  # [K]
        uniform = torch.ones_like(avg) / self.k
        # KL(uniform || avg) —— 均匀时最小
        kl = F.kl_div(uniform.log(), avg, reduction="batchmean")
        return kl

    def specialization_report(self, h, labels):
        """按 label（题型）统计分支选择分布，返回 [n_types, K] 矩阵。"""
        choose = self.choose(h).detach().cpu()        # [B,T]
        labels = labels.detach().cpu()
        out = []
        for t in labels.unique():
            mask = labels == t
            hist = torch.bincount(choose[mask], minlength=self.k).float()
            out.append((t.item(), (hist / hist.sum()).tolist()))
        return out


# ---------------- A1 的 oracle 标签生成 ----------------
def oracle_labels(branch_outs, teacher_h, reduction="mse"):
    """用教师表示生成"哪个分支更接近真相"的离散标签。

    branch_outs: list of [B,T,H]（各分支输出）
    teacher_h:   [B,T,H]（教师 = 原始 1D 拓扑 coda 前的隐藏状态）
    返回: [B,T] 的 long 张量，值 ∈ {0..K-1}
    """
    assert len(branch_outs) >= 2
    dists = []
    for bo in branch_outs:
        d = (bo.float() - teacher_h.float()).pow(2).mean(-1)   # [B,T]
        dists.append(d)
    return torch.stack(dists, dim=-1).argmin(-1)                # [B,T]


def oracle_accuracy(pred_choose, label):
    """router 预测与 oracle 的一致率（诊断 router 是否学到教师信号）。"""
    return (pred_choose == label).float().mean().item()


def ce_oracle(base, layers, branch_outs, tokens, position_ids, position_embeddings, coda):
    """CE-based oracle（修正版，替代表示距离 oracle）。

    对每个分支输出，经 coda+lm_head 算 next-token CE，选 CE 最低的分支
    （"这个分支已经够好" → early-exit 逻辑）。

    本机对照实验证明：repr oracle 标签 100% 偏深分支，CE oracle 标签均衡 ~50/50。
    """
    ces = []
    for hb in branch_outs:
        h = hb
        for li in coda:
            out = layers[li](hidden_states=h, attention_mask=None,
                             position_ids=position_ids,
                             position_embeddings=position_embeddings,
                             use_cache=False)
            h = out[0] if isinstance(out, (tuple, list)) else out
        logits = base.lm_head(base.model.norm(h))
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             tokens[:, 1:].reshape(-1), reduction="none")
        ce = ce.view(tokens.shape[0], -1)
        ces.append(F.pad(ce, (0, 1), value=ce.mean()))    # 对齐长度
    return torch.stack(ces, dim=-1).argmin(dim=-1)        # [B,T]


def ablation_oracle(base, layers, h_pre, branches, tokens, position_ids,
                    position_embeddings, coda, fusion, use_gumbel=False):
    """★ 改进版 oracle：用 per-token "跳过代价" 当标签。

    来自 BRICK 2D ablation 的洞察：真正该问的不是"哪个分支表示更接近教师"，
    而是"跳过哪个分支的 CE 代价更小" —— 这才是 early-exit 的动态激活逻辑。

    对每个 token 计算：
      ce_both      : 两分支都跑 + fusion（基线）
      ce_skip_b0   : 只跑 branch1（跳 branch0），fusion 输入 [h_pre, b1]
      ce_skip_b1   : 只跑 branch0（跳 branch1），fusion 输入 [b0, h_pre]
    标签 = argmin(ce_skip_*)：跳哪个代价小就跳哪个（即激活另一个）。

    返回 (label, stats)：
      label: [B,T] long，值为"要激活的分支索引"
      stats: dict，含平均 CE 与代价差，用于诊断
    """
    def to_logits(h):
        for li in coda:
            out = layers[li](hidden_states=h, attention_mask=None,
                             position_ids=position_ids,
                             position_embeddings=position_embeddings,
                             use_cache=False)
            h = out[0] if isinstance(out, (tuple, list)) else out
        return base.lm_head(base.model.norm(h))

    def ce_of(h):
        logits = to_logits(h)
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             tokens[:, 1:].reshape(-1), reduction="none")
        return F.pad(ce.view(tokens.shape[0], -1), (0, 1), value=ce.mean())

    k = len(branches)
    assert k == 2, "ablation oracle 当前为 2 分支设计"

    # 各分支输出
    outs = []
    for br in branches:
        hb = h_pre
        for li in br:
            out = layers[li](hidden_states=hb, attention_mask=None,
                             position_ids=position_ids,
                             position_embeddings=position_embeddings,
                             use_cache=False)
            hb = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(hb)

    # 三种激活组合的 CE
    ce_both = ce_of(fusion(h_pre, outs))
    ce_skip_b0 = ce_of(fusion(h_pre, [h_pre, outs[1]]))   # 只跑 b1 → 激活分支1
    ce_skip_b1 = ce_of(fusion(h_pre, [outs[0], h_pre]))   # 只跑 b0 → 激活分支0

    # 标签 = 跳过代价更小的那个分支（即"要激活的分支"）
    # skip_b0 代价小 → 激活分支1（label=1）；skip_b1 代价小 → 激活分支0（label=0）
    cost_skip_b0 = ce_skip_b0 - ce_both
    cost_skip_b1 = ce_skip_b1 - ce_both
    label = (cost_skip_b0 < cost_skip_b1).long()          # True=1(激活b1), False=0(激活b0)

    stats = {
        "ce_both": ce_both.mean().item(),
        "ce_skip_b0": ce_skip_b0.mean().item(),
        "ce_skip_b1": ce_skip_b1.mean().item(),
        "label_dist": torch.bincount(label.flatten(), minlength=k).float().div(label.numel()).tolist(),
    }
    return label, stats
