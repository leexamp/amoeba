# -*- coding: utf-8 -*-
"""本机 CPU 小规模完整训练 loop（tiny Llama，验证训练动态，非真实能力）。

流程：
  1. tiny 模型 + 随机 token 语料，生成教师监督 (prelude_h, teacher_h)
  2. A0 fusion 训练 60 步：MSE + CE，看 loss / alpha 是否下降/上升
  3. A1 router 训练 60 步：oracle CE，看 oracle 一致率是否上升

规模：12层 hidden128, seq32, batch4 —— CPU 数分钟内完成。
"""
import sys, time, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel
from brick_router import BrickRouter, oracle_labels, oracle_accuracy
import brick_config as cfg

class ToyConfig:
    N_LAYERS=12; PRELUDE=4; CODA=2; BRANCH_SIZES=(4,2)
    def branches(self):
        s=self.PRELUDE; out=[]
        for n in self.BRANCH_SIZES: out.append(list(range(s,s+n))); s+=n
        return out

T = ToyConfig()
torch.manual_seed(42)
CONF = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=12,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
                   max_position_embeddings=128, use_cache=True)
SEQ, BATCH = 32, 4

def make_data(base, nwin=24):
    """随机 token 语料 -> (prelude_h, teacher_h, tokens)。"""
    ph, th, tk = [], [], []
    with torch.no_grad():
        for _ in range(nwin):
            ids = torch.randint(0, 1000, (1, SEQ))
            pos = torch.arange(SEQ).unsqueeze(0)
            rope = precompute_rope(base, pos)
            h = base.model.embed_tokens(ids)
            for li in range(T.PRELUDE):
                h, _ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
            ph.append(h[0].clone())
            for li in range(T.PRELUDE, T.N_LAYERS - T.CODA):
                h, _ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
            th.append(h[0].clone())
            tk.append(ids[0].clone())
    return torch.stack(ph), torch.stack(th), torch.stack(tk)

def branch_outputs(layers, branches, h, pos, rope):
    bo = []
    for br in branches:
        hb = h
        for li in br:
            hb, _ = call_layer(layers[li], hb, position_ids=pos, position_embeddings=rope)
        bo.append(hb)
    return bo

t0 = time.time()
print("=== 生成教师监督数据 ===", flush=True)
base = LlamaForCausalLM(CONF).eval()
ph, th, tk = make_data(base)
print(f"  {ph.shape[0]} windows, prelude_h {tuple(ph.shape)}, teacher_h {tuple(th.shape)} "
      f"({time.time()-t0:.1f}s)", flush=True)

pos = torch.arange(SEQ).unsqueeze(0)
rope = precompute_rope(base, pos)
model = BrickModel(base, config=T).eval()
# 冻结主干（本来 BrickModel 就冻结了）
model.fusion.train()
for p in model.fusion.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW(model.fusion.parameters(), lr=3e-4)

print("\n=== A0 fusion 训练（60 步）===", flush=True)
print("  step |  MSE  |  CE   | alpha", flush=True)
for step in range(60):
    sel = torch.randint(0, ph.shape[0], (BATCH,))
    h, t, y = ph[sel], th[sel], tk[sel]
    bo = branch_outputs(model.layers, model.branches, h, pos, rope)
    fused = model.fusion(h, bo)
    mse = F.mse_loss(fused.float(), t.float())
    hc = fused
    for li in model.coda:
        hc, _ = call_layer(model.layers[li], hc, position_ids=pos, position_embeddings=rope)
    logits = model.lm_head(model.norm(hc))
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
    loss = mse + 0.5 * ce
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0 or step == 59:
        print(f"  {step:4d} | {mse.item():.4f} | {ce.item():.4f} | {model.fusion.alpha.item():.4f}", flush=True)

# 冻结 fusion，训练 router
model.fusion.eval()
for p in model.fusion.parameters(): p.requires_grad_(False)
router = BrickRouter(128, 2)
ropt = torch.optim.AdamW(router.parameters(), lr=3e-4)

print("\n=== A1 router 训练（60 步，oracle 监督）===", flush=True)
print("  step |  loss  | oracle-agree（全数据）", flush=True)
with torch.no_grad():
    all_bo = branch_outputs(model.layers, model.branches, ph, pos, rope)
    all_label = oracle_labels(all_bo, th)

for step in range(60):
    sel = torch.randint(0, ph.shape[0], (BATCH,))
    h = ph[sel]
    logits = router(h)
    label = all_label[sel]
    loss = F.cross_entropy(logits.reshape(-1, 2), label.reshape(-1))
    ropt.zero_grad(); loss.backward(); ropt.step()
    if step % 10 == 0 or step == 59:
        with torch.no_grad():
            agree = oracle_accuracy(router(ph).argmax(-1), all_label)
        print(f"  {step:4d} | {loss.item():.4f} | {agree:.3f}", flush=True)

# 分支塌缩诊断
with torch.no_grad():
    choose = router(ph).argmax(-1)
    hist = torch.bincount(choose.flatten(), minlength=2).float()
    hist = (hist / hist.sum()).tolist()
print(f"\n分支选择分布: {hist}" + ("  ← 塌缩！" if max(hist) > 0.95 else "  ← 有动态性"), flush=True)
print(f"\n总耗时 {time.time()-t0:.1f}s", flush=True)
print("\n=== 结论 ===", flush=True)
print("A0: MSE/CE 应下降、alpha 应从 0 上升 → 训练动态正常")
print("A1: oracle-agree 应上升 → 教师信号可被 router 学到")
print("（tiny 随机语料，非真实能力；仅验证训练机制，数字不代表 2B 结果）")
