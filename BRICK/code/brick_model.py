# -*- coding: utf-8 -*-
"""BRICK 核心：把冻结的 1D 层链重排成 2D 分支拓扑。

关键洞察：每个分支使用**不同的原始层索引**，因此各分支的 KV cache 在
HF 的 DynamicCache 中天然不冲突（原模型的 42 个 cache 槽各用一次）。
第一阶段用 mode="all"（所有分支都算 + 融合），回答核心科学问题：
"2D 重排 + 轻量 fusion 能否保持冻结模型的能力？"
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from layer_utils import call_layer, precompute_rope
import brick_config as cfg


class BrickFusion(nn.Module):
    """融合模块，支持三种模式（本机真实 1B 对照后扩展）：

    mode="residual_prelude" (A0 原始)：
        h_out = h_pre + alpha * proj(branches)     —— 起点=h_pre，但真实模型上 MSE=19.6 收敛慢
    mode="residual_branch_avg" (A0-fix 推荐)：
        h_base = mean(branches); h_out = h_base + alpha * proj(branches)
        —— 起点=分支平均，更接近教师（分支经过 14 层，h_pre 只经过 8 层）
    mode="direct" (A0-alt)：
        h_out = proj(branches)，proj 初始化为等权平均 —— 无残差锚点，直接学融合

    alpha=0 起点保证：前两种模式的起点都是恒等/近恒等映射（不破坏表示）。
    proj 保持正常初始化（否则 alpha 梯度恒为 0，训练无法启动 —— 本机实测）。
    """

    def __init__(self, hidden: int, k: int, mode: str = "residual_branch_avg"):
        super().__init__()
        assert mode in ("residual_prelude", "residual_branch_avg")
        self.k = k
        self.mode = mode
        self.proj = nn.Linear(k * hidden, hidden, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))    # alpha=1：proj 有完整梯度
        with torch.no_grad():
            self.proj.weight.zero_()                 # proj 零初始化：起点 = 残差锚点（恒等）

        # 注意：必须只零初始化 proj、alpha=1。
        # 之前 alpha=0 + proj 正常初始化导致 alpha 梯度≈0、训练几乎不动（本机实测）。
        # direct 模式的等权平均初始化已证伪（训练爆炸到 MSE 19），故移除。

    def set_dtype(self, dtype):
        """把 proj/alpha 转为与冻结主干相同的 dtype（真实模型是 bf16）。"""
        self.proj.to(dtype=dtype)
        self.alpha.data = self.alpha.data.to(dtype=dtype)

    def forward(self, h_pre, branch_outs):
        x = torch.cat(branch_outs, dim=-1)
        if self.proj.weight.dtype != h_pre.dtype:
            self.set_dtype(h_pre.dtype)
        if self.mode == "residual_prelude":
            return h_pre + self.alpha * self.proj(x)
        h_base = torch.stack(branch_outs, dim=0).mean(0)   # 分支平均作为残差锚点
        return h_base + self.alpha * self.proj(x)

    def extra_repr(self):
        return f"k={self.k}, mode={self.mode}, alpha={self.alpha.item():.4f}"


class BrickModel(nn.Module):
    """冻结主干 + 2D 分支重排。仅 fusion（及可选 gate）可训练。"""

    def __init__(self, base, config=cfg):
        super().__init__()
        self.base = base
        self.cfg = config
        self.layers = base.model.layers
        self.norm = base.model.norm
        self.lm_head = base.lm_head
        self.embed = base.model.embed_tokens

        self.prelude = list(range(config.PRELUDE))
        self.branches = config.branches()
        self.coda = list(range(config.N_LAYERS - config.CODA, config.N_LAYERS))
        self._validate()

        fusion_mode = getattr(config, "FUSION_MODE", "residual_branch_avg")
        self.fusion = BrickFusion(base.config.hidden_size, len(self.branches), mode=fusion_mode)

        # 冻结全部主干
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.mode = "all"   # "all" | "routed"（routed 为第二阶段接口）

    def _validate(self):
        used = self.prelude + sum(self.branches, []) + self.coda
        assert sorted(used) == list(range(self.cfg.N_LAYERS)), "层索引必须完整覆盖且不重叠"
        assert len(set(used)) == len(used), "层索引不可重复"
        assert self.coda[-1] == self.cfg.N_LAYERS - 1, "最后一层必须在 coda（readout interface）"

    def trainable_parameters(self):
        return list(self.fusion.parameters())

    def save_fusion(self, path):
        torch.save({"fusion": self.fusion.state_dict(), "cfg": {
            "prelude": self.prelude, "branches": self.branches, "coda": self.coda}}, path)

    def load_fusion(self, path):
        ck = torch.load(path, map_location="cpu")
        self.fusion.load_state_dict(ck["fusion"])

    def generate(self, **kwargs):
        """委托给 base 的 generate，但临时把 forward 换成 BRICK 的 2D 拓扑。

        依赖 transformers>=4.45 的 DynamicCache（可变对象，层调用时原地写入），
        因此 42 个层的 KV cache 槽在 all 模式下各用一次、互不冲突。
        远端是 transformers 5.24 —— 满足该假设。
        """
        old = self.base.forward
        self.base.forward = self.forward
        try:
            return self.base.generate(**kwargs)
        finally:
            self.base.forward = old

    # ---------- 层调用（兼容 HF 多版本签名）----------
    def _call_layer(self, idx, h, past, mask, pos_ids, cp, use_cache, rope=None):
        out, _ = call_layer(self.layers[idx], h, position_ids=pos_ids, past=past,
                            cache_position=cp, use_cache=use_cache, attention_mask=mask,
                            position_embeddings=rope)
        return out

    def _run_branch(self, indices, h, past, mask, pos_ids, cp, use_cache, rope=None):
        for i in indices:
            h = self._call_layer(i, h, past, mask, pos_ids, cp, use_cache, rope)
        return h

    def _infer_positions(self, h, position_ids, past):
        """generate 时 position_ids 可能为 None；从 past 长度推断。"""
        if position_ids is not None:
            return position_ids
        seqlen = h.shape[1]
        if past is not None:
            try:
                offset = past.get_seq_length()
            except Exception:
                offset = 0
        else:
            offset = 0
        return torch.arange(offset, offset + seqlen, device=h.device).unsqueeze(0)

    # ---------- 教师前向：原始 1D 拓扑，用于生成蒸馏监督 ----------
    @torch.no_grad()
    def teacher_hidden(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                       position_ids=None, past_key_values=None, cache_position=None):
        """返回 coda 之前的隐藏状态（即原模型层 0..N-CODA-1 顺序执行的结果）。"""
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        pos_ids = self._infer_positions(h, position_ids, past_key_values)
        rope = precompute_rope(self.base, pos_ids)
        for i in range(self.cfg.N_LAYERS - self.cfg.CODA):
            h = self._call_layer(i, h, past_key_values, attention_mask,
                                 pos_ids, cache_position, False, rope)
        return h

    # ---------- BRICK 前向 ----------
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                position_ids=None, cache_position=None, inputs_embeds=None,
                use_cache=True, labels=None, **kwargs):
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        pos_ids = self._infer_positions(h, position_ids, past_key_values)
        rope = precompute_rope(self.base, pos_ids)

        for i in self.prelude:
            h = self._call_layer(i, h, past_key_values, attention_mask,
                                 pos_ids, cache_position, use_cache, rope)

        if self.mode == "all":
            h_pre = h   # 残差起点：prelude 输出
            bo = [self._run_branch(br, h, past_key_values, attention_mask,
                                   pos_ids, cache_position, use_cache, rope)
                  for br in self.branches]
            h = self.fusion(h_pre, bo)
        else:
            raise NotImplementedError(
                "token-level routed 模式（每 token 只走一个分支）是第二阶段："
                "需解决未激活分支的 KV cache 长度一致性问题。第一阶段先用 all 模式。")

        for i in self.coda:
            h = self._call_layer(i, h, past_key_values, attention_mask,
                                 pos_ids, cache_position, use_cache, rope)

        logits = self.lm_head(self.norm(h))
        out = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out.loss = F.cross_entropy(shift_logits.float().reshape(-1, shift_logits.size(-1)),
                                       shift_labels.reshape(-1))
        return out


def load_brick(model_path=cfg.BASE_MODEL, tokenizer_path=cfg.TOKENIZER, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device)
    base.config.use_cache = True
    model = BrickModel(base).to(device)
    return model, tok


if __name__ == "__main__":
    # 无 GPU 时只做拓扑校验（不加载模型）
    print("prelude :", list(range(cfg.PRELUDE)))
    for i, b in enumerate(cfg.branches()):
        print(f"branch{i}:", b[0], "..", b[-1], f"({len(b)} layers)")
    print("coda    :", list(range(cfg.N_LAYERS - cfg.CODA, cfg.N_LAYERS)))
    b = cfg.branches()
    print("branch param ratio:", [len(x) for x in b], "-> 异形积木 ✓")
