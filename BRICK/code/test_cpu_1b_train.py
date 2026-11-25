# -*- coding: utf-8 -*-
"""真实 MiniCPM5-1B 上的 A0 fusion 微训练（CPU，小规模）。

目的：验证在真实模型上 fusion 能否把 2D 重排造成的损失拉回来 ——
这是 BRICK 项目最关键的实证问题，tiny 模型回答不了。

规模：chat_corpus 取 6 个窗口(seq64)，训练 30 步，bf16 CPU，约几分钟。
指标：MSE(fused, teacher_h) 和 CE 是否下降。
"""
import sys, time, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel, load_brick

MODEL = r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS = r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ, NWIN, STEPS = 16, 3, 4   # CPU bf16 极慢：每步约 4-5s，4步约20s

class OneBConfig:
    N_LAYERS=24; PRELUDE=8; CODA=2; BRANCH_SIZES=(8,6)
    def branches(self):
        s=self.PRELUDE; out=[]
        for n in self.BRANCH_SIZES: out.append(list(range(s,s+n))); s+=n
        return out

t0 = time.time()
print("=== 加载 1B + BRICK ===", flush=True)
model, tok = load_brick(MODEL, MODEL, device="cpu")
base = model.base
model.config = OneBConfig()
# 重建为 24 层拓扑（load_brick 默认用 42 层 cfg）
from brick_model import BrickModel
model = BrickModel(base, config=OneBConfig()).to("cpu")
model.eval()
print(f"  加载耗时 {time.time()-t0:.1f}s", flush=True)

print("\n=== 从 chat_corpus_1b.pt 取真实 token 窗口 ===", flush=True)
if not os.path.exists(CORPUS):
    print(f"  语料不存在 {CORPUS}，用随机 token 代替", flush=True)
    c = {"arg": torch.randint(0, 1000, (NWIN, 512))}
else:
    c = torch.load(CORPUS, map_location="cpu")
arg = c["arg"]
print(f"  arg shape {tuple(arg.shape)}", flush=True)

pos = torch.arange(SEQ).unsqueeze(0)
rope = precompute_rope(base, pos)
ph, th, tk = [], [], []
t = time.time()
with torch.no_grad():
    for i in range(NWIN):
        ids = arg[i, :SEQ].unsqueeze(0).long()
        h = base.model.embed_tokens(ids)
        for li in range(8):
            h,_ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
        ph.append(h[0].clone())
        for li in range(8, 22):
            h,_ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
        th.append(h[0].clone())
        tk.append(ids[0].clone())
ph, th, tk = torch.stack(ph), torch.stack(th), torch.stack(tk)
print(f"  教师数据 {tuple(ph.shape)}, 耗时 {time.time()-t:.1f}s", flush=True)

# 训练 fusion
model.fusion.train()
for p in model.fusion.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW(model.fusion.parameters(), lr=3e-4)
print(f"  fusion 可训练参数 {sum(p.numel() for p in model.fusion.parameters()):,}", flush=True)

def branches_out(h):
    bo = []
    for br in model.branches:
        hb = h
        for li in br:
            hb,_ = call_layer(model.layers[li], hb, position_ids=pos, position_embeddings=rope)
        bo.append(hb)
    return bo

print("\n=== A0 fusion 训练（真实 1B，30 步）===", flush=True)
print("  step |   MSE   |   CE    | alpha |  vs 原始(随机窗口)", flush=True)
with torch.no_grad():
    # baseline: 2D 重排 fusion=恒等 时与教师的 MSE
    bo0 = branches_out(ph[:1])
    f0 = model.fusion(ph[:1], bo0)
    base_mse = F.mse_loss(f0.float(), th[:1].float()).item()
    print(f"  起点 MSE(2D重排 vs 教师) = {base_mse:.4f}", flush=True)

for step in range(STEPS):
    sel = torch.randint(0, NWIN, (1,))
    h, t, y = ph[sel], th[sel], tk[sel]
    bo = branches_out(h)
    fused = model.fusion(h, bo)
    mse = F.mse_loss(fused.float(), t.float())
    hc = fused
    for li in model.coda:
        hc,_ = call_layer(model.layers[li], hc, position_ids=pos, position_embeddings=rope)
    logits = model.lm_head(model.norm(hc))
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
    loss = mse + 0.5 * ce
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0 or step == STEPS-1:
        print(f"  {step:4d} | {mse.item():.4f} | {ce.item():.4f} | "
              f"{model.fusion.alpha.item():.4f}", flush=True)

with torch.no_grad():
    bo_end = branches_out(ph)
    f_end = model.fusion(ph, bo_end)
    end_mse = F.mse_loss(f_end.float(), th.float()).item()
print(f"\n全数据 MSE: {base_mse:.4f} -> {end_mse:.4f}", flush=True)
print(f"总耗时 {time.time()-t0:.1f}s", flush=True)
print("\n判读: MSE 下降 = fusion 学会把 2D 重排拉回教师表示；不降 = 负面结果", flush=True)
