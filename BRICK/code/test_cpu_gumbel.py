# -*- coding: utf-8 -*-
"""本机对照：Gumbel-Softmax 能否解决 router 塌缩（承接 test_cpu_router_fix.py 的 E6=0.85/0.15）。

配置对照：
  G1  普通 router (softmax+CE)                     —— 基线（已知 0.85~0.99）
  G2  Gumbel (soft, tau退火)                        —— 可微随机探索
  G3  Gumbel (hard straight-through, tau退火)      —— 前向离散、反向可微
  G4  Gumbel hard + 熵正则                          —— 显式防塌缩
  G5  Gumbel hard + 熵正则 + 全局负载均衡           —— 三重机制
  G6  Gumbel hard + 熵正则 + load balance（等深拓扑）

用 CE oracle 标签（已验证均衡），只改 router 训练方式，看选择分布是否接近 0.5/0.5。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_router import BrickRouter, ce_oracle
from brick_router_gumbel import GumbelRouter, BalancedGumbelRouter

torch.manual_seed(42)
CONF = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=12,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
                   max_position_embeddings=128, use_cache=True)
SEQ, BATCH, NWIN, STEPS = 32, 4, 24, 120

class Topo:
    def __init__(self, sizes): self.sizes=sizes
    def branches(self):
        s=4; out=[]
        for n in self.sizes: out.append(list(range(s,s+n))); s+=n
        return out

def make_data(base):
    ph, th, tk = [], [], []
    pos = torch.arange(SEQ).unsqueeze(0); rope = precompute_rope(base, pos)
    with torch.no_grad():
        for _ in range(NWIN):
            ids = torch.randint(0,1000,(1,SEQ))
            h = base.model.embed_tokens(ids)
            for li in range(4): h,_ = call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
            ph.append(h[0].clone())
            for li in range(4,10): h,_ = call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
            th.append(h[0].clone()); tk.append(ids[0].clone())
    return torch.stack(ph), torch.stack(th), torch.stack(tk), pos, rope

def branch_outputs(layers, brs, h, pos, rope):
    out=[]
    for br in brs:
        hb=h
        for li in br: hb,_ = call_layer(layers[li],hb,position_ids=pos,position_embeddings=rope)
        out.append(hb)
    return out

base = LlamaForCausalLM(CONF).eval()
ph, th, tk, pos, rope = make_data(base)
TOPS = {"异深(4,2)": Topo((4,2)), "等深(3,3)": Topo((3,3))}
cache = {}
for name, topo in TOPS.items():
    bo = branch_outputs(base.model.layers, topo.branches(), ph, pos, rope)
    with torch.no_grad():
        cache[name] = ce_oracle(base, base.model.layers, bo, tk, pos, rope, [10,11])

def run(gname, topo_name, build_router, entropy=0.0, balance=0.0):
    torch.manual_seed(13)
    label = cache[topo_name]
    r = build_router()
    opt = torch.optim.AdamW(r.parameters(), lr=3e-4)
    is_gumbel = isinstance(r, (GumbelRouter, BalancedGumbelRouter))
    for step in range(STEPS):
        sel = torch.randint(0, NWIN, (BATCH,))
        h = ph[sel]
        if is_gumbel:
            soft, logits = r(h, hard=True, total_steps=STEPS)
            loss = F.cross_entropy(logits.reshape(-1, 2), label[sel].reshape(-1))
            if entropy > 0: loss = loss - entropy * r.entropy_loss(soft)
            if balance > 0 and hasattr(r, 'balance_loss'):
                loss = loss + balance * r.balance_loss(soft)
        else:
            logits = r(h)
            loss = F.cross_entropy(logits.reshape(-1, 2), label[sel].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if is_gumbel: r.step += 1
    with torch.no_grad():
        if is_gumbel:
            choose = r.choose(ph, total_steps=STEPS)
        else:
            choose = r(ph).argmax(-1)
        hist = torch.bincount(choose.flatten(), minlength=2).float()
        hist = (hist/hist.sum()).tolist()
        agree = (choose == label).float().mean().item()
    print(f"  {gname:44s} {topo_name:10s} 分布={[round(x,2) for x in hist]}  agree={agree:.2f}", flush=True)

print("=== Gumbel-Softmax 防塌缩对照（CE oracle 标签，已均衡）===", flush=True)
run("G1 普通 router", "异深(4,2)", lambda: BrickRouter(128,2))
run("G2 Gumbel soft tau退火", "异深(4,2)", lambda: GumbelRouter(128,2,tau_start=1.0,tau_end=0.1))
run("G3 Gumbel hard ST tau退火", "异深(4,2)", lambda: GumbelRouter(128,2,tau_start=1.0,tau_end=0.1))
run("G4 Gumbel hard + 熵正则0.1", "异深(4,2)", lambda: GumbelRouter(128,2,tau_start=1.0,tau_end=0.1), entropy=0.1)
run("G5 Gumbel hard+熵+负载均衡0.5", "异深(4,2)", lambda: BalancedGumbelRouter(128,2,tau_start=1.0,tau_end=0.1), entropy=0.1, balance=0.5)
run("G6 Gumbel hard+熵+负载均衡0.5", "等深(3,3)", lambda: BalancedGumbelRouter(128,2,tau_start=1.0,tau_end=0.1), entropy=0.1, balance=0.5)
run("G7 Gumbel hard+强熵0.5+强均衡1.0", "等深(3,3)", lambda: BalancedGumbelRouter(128,2,tau_start=1.0,tau_end=0.1), entropy=0.5, balance=1.0)

print("\n判读: 分布越接近 [0.5,0.5] 越好；agree 不应太低（否则路由无意义）", flush=True)
