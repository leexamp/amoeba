# -*- coding: utf-8 -*-
"""真实 MiniCPM5-1B：三种 fusion 模式对照（本机 CPU）。

核心问题（上轮实验暴露）：residual_prelude 起点 MSE=19.6，alpha≈0 收敛慢。
对照：
  residual_prelude   : h_pre + α·proj(branches)        （A0 原始）
  residual_branch_avg: mean(branches) + α·proj(branches)（推荐修正）
  direct             : proj(branches) 等权平均初始化   （无残差锚点）

指标：起点 MSE（fusion 输入 vs 教师）→ 4步训练后 MSE，看哪个起点更接近教师、
哪个下降更快。seq16, 3窗口, 每模式4步，CPU 约 3×20s = 1分钟。
"""
import sys, time, copy
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel

MODEL = r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS = r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ, NWIN, STEPS = 16, 3, 4

class OneBConfig:
    N_LAYERS=24; PRELUDE=8; CODA=2; BRANCH_SIZES=(8,6)
    FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=self.PRELUDE; out=[]
        for n in self.BRANCH_SIZES: out.append(list(range(s,s+n))); s+=n
        return out

t0 = time.time()
print("=== 加载 1B ===", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True).eval()
print(f"  耗时 {time.time()-t0:.1f}s", flush=True)

c = torch.load(CORPUS, map_location="cpu")
arg = c["arg"]
pos = torch.arange(SEQ).unsqueeze(0)
rope = precompute_rope(base, pos)
ph, th = [], []
with torch.no_grad():
    for i in range(NWIN):
        ids = arg[i, :SEQ].unsqueeze(0).long()
        h = base.model.embed_tokens(ids)
        for li in range(8): h,_ = call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        ph.append(h[0].clone())
        for li in range(8,22): h,_ = call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        th.append(h[0].clone())
ph, th = torch.stack(ph), torch.stack(th)
print(f"  教师数据 {tuple(ph.shape)}", flush=True)

def branch_out(layers, brs, h):
    bo=[]
    for br in brs:
        hb=h
        for li in br: hb,_ = call_layer(layers[li],hb,position_ids=pos,position_embeddings=rope)
        bo.append(hb)
    return bo

print("\n=== 三模式对照（真实 1B）===", flush=True)
results = {}
for mode in ["residual_prelude", "residual_branch_avg"]:
    torch.manual_seed(7)
    C = OneBConfig(); C.FUSION_MODE = mode
    model = BrickModel(base, config=C).eval()
    model.fusion.train()
    for p in model.fusion.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(model.fusion.parameters(), lr=3e-4)

    with torch.no_grad():
        bo = branch_out(model.layers, model.branches, ph)
        h_base = torch.stack(bo,0).mean(0)
        start_prelude = F.mse_loss(ph.float(), th.float()).item()
        start_branchavg = F.mse_loss(h_base.float(), th.float()).item()
        f0 = model.fusion(ph, bo)
        start_fused = F.mse_loss(f0.float(), th.float()).item()

    for step in range(STEPS):
        s = torch.randint(0, NWIN, (1,))
        h, t = ph[s], th[s]
        bo_s = branch_out(model.layers, model.branches, h)
        fused = model.fusion(h, bo_s)
        loss = F.mse_loss(fused.float(), t.float())
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        bo_e = branch_out(model.layers, model.branches, ph)
        f_e = model.fusion(ph, bo_e)
        end_mse = F.mse_loss(f_e.float(), th.float()).item()
    results[mode] = (start_fused, end_mse, model.fusion.alpha.item())
    print(f"  {mode:22s} 起点(h_pre)={start_prelude:6.2f}  起点(分支均值)={start_branchavg:6.2f}  "
          f"起点(fused)={start_fused:6.2f} → 4步后={end_mse:6.2f}  alpha={model.fusion.alpha.item():+.4f}", flush=True)
    del model, opt

print("\n=== 判读 ===", flush=True)
print("起点(fused) 越小 → 融合起点越接近教师（越好）", flush=True)
print("4步后 MSE 下降越多 → 收敛越快", flush=True)
best = min(results, key=lambda m: results[m][1])
print(f"最优模式（4步后 MSE 最低）: {best}", flush=True)
print(f"总耗时 {time.time()-t0:.1f}s", flush=True)
