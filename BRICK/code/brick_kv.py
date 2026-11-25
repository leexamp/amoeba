# -*- coding: utf-8 -*-
"""A3: Frozen Joint-KV —— 跨分支 KV 对齐（最有价值也最难）。

核心问题：MBT 的 Joint-KV 能共享缓存是因为分支从头训练、表示天然对齐。
BRICK 的分支是冻结的不同预训练层，表示不对齐。

算法：给每个分支的每一层加 rank-r LoRA K/V 投影，把分支 hidden 投影到
共享 KV 空间，写入同一个 cache 槽。用教师的 K/V 当监督训练投影头。

注意（v4 教训）：KV 预测在 Amoeba 上 QA +5pp 但 decode 0.973× 更慢 ——
本模块必须配合端到端速度测试，不能只看准确率。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAKVProjector(nn.Module):
    """单个分支单层的 K/V LoRA 投影：h -> (K, V) 共享空间。

    用低秩分解 W = B@A 控制参数量（rank 默认 16）。
    """

    def __init__(self, hidden: int, kv_dim: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.kv_dim = kv_dim
        # K 投影
        self.kA = nn.Parameter(torch.randn(rank, hidden) * 0.02)
        self.kB = nn.Parameter(torch.zeros(kv_dim, rank))
        # V 投影
        self.vA = nn.Parameter(torch.randn(rank, hidden) * 0.02)
        self.vB = nn.Parameter(torch.zeros(kv_dim, rank))

    def forward(self, h):
        # h: [B,T,H]
        k = F.linear(F.linear(h.float(), self.kA), self.kB)     # [B,T,kv_dim]
        v = F.linear(F.linear(h.float(), self.vA), self.vB)
        return k, v

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class BranchKVAligner(nn.Module):
    """管理 K 个分支 × L 层的投影头，提供对齐损失。"""

    def __init__(self, hidden: int, kv_dim: int, branch_layers: list, rank: int = 16):
        """
        branch_layers: [[layer_idx, ...], ...] 每个分支的原始层索引列表
        """
        super().__init__()
        self.branch_layers = branch_layers
        self.k = len(branch_layers)
        self.projectors = nn.ModuleList([
            nn.ModuleList([LoRAKVProjector(hidden, kv_dim, rank) for _ in br])
            for br in branch_layers
        ])
        total = sum(p.num_params() for br in self.projectors for p in br)
        print(f"[Joint-KV] {self.k} branches, {sum(len(b) for b in branch_layers)} projectors, "
              f"{total:,} params ({total/1e6:.2f}M)", flush=True)

    def project_branch(self, b, hidden_list):
        """把分支 b 的各层 hidden 投影成共享 KV。

        hidden_list: list of [B,T,H]，长度 = 该分支层数
        返回: (k_list, v_list)
        """
        k_out, v_out = [], []
        for p, h in zip(self.projectors[b], hidden_list):
            k, v = p(h)
            k_out.append(k)
            v_out.append(v)
        return k_out, v_out

    def align_loss(self, branch_kv, teacher_kv):
        """让各分支的 K/V 对齐教师的 K/V（MSE）。

        branch_kv:   list over branches of (k_list, v_list)
        teacher_kv:  (k_list, v_list) from teacher（同层索引）
        """
        tk, tv = teacher_kv
        losses = []
        for k_list, v_list in branch_kv:
            for k, v, kt, vt in zip(k_list, v_list, tk, tv):
                losses.append(F.mse_loss(k.float(), kt.float()))
                losses.append(F.mse_loss(v.float(), vt.float()))
        return torch.stack(losses).mean()


@torch.no_grad()
def extract_teacher_kv(base, layer_indices, input_ids, position_ids):
    """从教师模型提取指定层的 K/V（用于对齐监督）。

    注意：这是诊断/数据生成用，训练时不反传。
    实际 K/V 在 Llama 里是 q_proj/k_proj/v_proj 的输出，这里近似用
    layer 的 input hidden 经 k_proj/v_proj 的结果。
    """
    kv = []
    h = base.model.embed_tokens(input_ids)
    for li in layer_indices:
        layer = base.model.layers[li]
        # 取 RMSNorm 后的 hidden（近似 attention 输入）
        h_norm = layer.input_layernorm(h)
        k = layer.self_attn.k_proj(h_norm)
        v = layer.self_attn.v_proj(h_norm)
        kv.append((k, v))
        # 继续前向以得到下一层输入（简化：忽略 attention/MLP 更新）
        out = layer(hidden_states=h, position_ids=position_ids, use_cache=False)
        h = out[0] if isinstance(out, (tuple, list)) else out
    return kv
