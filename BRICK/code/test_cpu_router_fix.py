# -*- coding: utf-8 -*-
"""本机对照实验：定位 router 塌缩根因 + 验证修正方案。

对照：
  E1  表示距离 oracle + 异深分支 (4,2)   —— 已验证：塌缩 [1,0]
  E2  CE-based oracle  + 异深分支 (4,2)   —— 修正1：用 CE 而非表示距离
  E3  表示距离 oracle + A5 负载均衡       —— 修正2：balance_loss
  E4  表示距离 oracle + 等深分支 (3,3)    —— 验证"深度差异导致塌缩"假设
  E5  CE-based oracle + 等深分支 (3,3)    —— 双重修正

输出每个实验的分支选择分布 + oracle 一致率，判断哪个修正真正解决塌缩。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_router import BrickRouter, SpecializedRouter, oracle_labels, oracle_accuracy
import brick_config as cfg

torch.manual_seed(42)
CONF = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=12,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
                   max_position_embeddings=128, use_cache=True)
SEQ, BATCH, NWIN = 32, 4, 24

class Topology:
    def __init__(self, sizes):
        self.N_LAYERS=12; self.PRELUDE=4; self.CODA=2; self.BRANCH_SIZES=sizes
    def branches(self):
        s=self.PRELUDE; out=[]
        for n in self.BRANCH_SIZES: out.append(list(range(s,s+n))); s+=n
        assert s == self.N_LAYERS - self.CODA
        return out

def make_data(base):
    ph, th, tk = [], [], []
    pos = torch.arange(SEQ).unsqueeze(0)
    rope = precompute_rope(base, pos)
    with torch.no_grad():
        for _ in range(NWIN):
            ids = torch.randint(0, 1000, (1, SEQ))
            h = base.model.embed_tokens(ids)
            for li in range(4):
                h,_ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
            ph.append(h[0].clone())
            for li in range(4, 10):
                h,_ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
            th.append(h[0].clone())
            tk.append(ids[0].clone())
    return torch.stack(ph), torch.stack(th), torch.stack(tk), pos, rope

def branch_outputs(layers, brs, h, pos, rope):
    out = []
    for br in brs:
        hb = h
        for li in br:
            hb,_ = call_layer(layers[li], hb, position_ids=pos, position_embeddings=rope)
        out.append(hb)
    return out

def ce_oracle(base, layers, bo, tk, pos, rope):
    """CE-based oracle：对每个分支输出，经 coda+lm_head 算 next-token CE，
    选 CE 最低的分支（'这个分支已经够好' → early exit 逻辑）。"""
    ces = []
    for hb in bo:
        h = hb
        for li in [10, 11]:
            h,_ = call_layer(layers[li], h, position_ids=pos, position_embeddings=rope)
        logits = base.lm_head(base.model.norm(h))
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             tk[:, 1:].reshape(-1), reduction='none').view(tk.shape[0], -1)
        # 对齐到分支输出长度
        ces.append(F.pad(ce, (0, 1), value=ce.mean()))  # [B,T]
    return torch.stack(ces, -1).argmin(-1)   # [B,T]

def run(name, topo, oracle_kind, use_balance=False):
    torch.manual_seed(7)
    base = LlamaForCausalLM(CONF).eval()
    ph, th, tk, pos, rope = make_data(base)
    bo = branch_outputs(base.model.layers, topo.branches(), ph, pos, rope)

    if oracle_kind == "repr":
        label = oracle_labels(bo, th)
    else:
        with torch.no_grad():
            label = ce_oracle(base, base.model.layers, bo, tk, pos, rope)
    k = len(topo.branches())

    if use_balance:
        router = SpecializedRouter(128, k)
    else:
        router = BrickRouter(128, k)
    opt = torch.optim.AdamW(router.parameters(), lr=3e-4)
    for step in range(80):
        sel = torch.randint(0, NWIN, (BATCH,))
        logits = router(ph[sel])
        loss = F.cross_entropy(logits.reshape(-1, k), label[sel].reshape(-1))
        if use_balance:
            loss = loss + 0.3 * router.balance_loss(ph[sel])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        choose = router(ph).argmax(-1)
        hist = torch.bincount(choose.flatten(), minlength=k).float()
        hist = (hist/hist.sum()).tolist()
        agree = oracle_accuracy(choose, label)
    # oracle 自身的分布（标签是否本身就偏）
    lab_hist = torch.bincount(label.flatten(), minlength=k).float()
    lab_hist = (lab_hist/lab_hist.sum()).tolist()
    print(f"  {name:42s} router={[round(x,2) for x in hist]}  oracle标签={[round(x,2) for x in lab_hist]}  agree={agree:.2f}", flush=True)
    return hist

print("=== router 塌缩根因对照实验（tiny CPU）===", flush=True)
print("格式: router选择分布 / oracle标签自身分布 / 一致率\n", flush=True)

run("E1 repr oracle + 异深(4,2)", Topology((4,2)), "repr")
run("E2 CE oracle  + 异深(4,2)", Topology((4,2)), "ce")
run("E3 repr oracle + 异深(4,2) + balance", Topology((4,2)), "repr", use_balance=True)
run("E4 repr oracle + 等深(3,3)", Topology((3,3)), "repr")
run("E5 CE oracle  + 等深(3,3)", Topology((3,3)), "ce")
run("E6 CE oracle  + 等深(3,3) + balance", Topology((3,3)), "ce", use_balance=True)

print("\n=== 判读 ===", flush=True)
print("若 oracle 标签自身分布已偏 → 问题在监督信号定义，不在 router 训练", flush=True)
print("若 oracle 均衡但 router 仍塌缩 → 问题在训练动态，需加强 balance", flush=True)
print("若 CE oracle 比 repr oracle 更均衡 → 修正方案有效，写入 ALGORITHMS.md", flush=True)
