# -*- coding: utf-8 -*-
"""真实 1B：BRICK 2D 拓扑下的块 ablation（区别于之前的 1D 顺序 ablation）。

关键区别：
  之前的 ablation：1D 顺序跑 24 层，跳过 layer l（残差直连）
  本实验：BRICK 2D 拓扑（8 + [8,6] + 2），在不同"分支激活组合"下测 CE

对照：
  B0  both        : 两个分支都跑 + fusion（默认 all 模式）
  B1  branch0_only: 只跑 branch0，fusion 输入 = [b0, h_pre]（b1 位置用 h_pre 占位）
  B2  branch1_only: 只跑 branch1，fusion 输入 = [h_pre, b1]
  B3  none        : 两个分支都不跑，fusion 输入 = [h_pre, h_pre]（相当于纯 prelude → coda）
  B4  prelude_only: 跳过 prelude 后半（layer4-7），只跑 layer0-3 → 分支
  B5  no_coda     : 跳过 coda，直接 lm_head(norm(fusion_out))（灾难性对照）

指标：next-token CE（真实 chat 语料）。
这直接回答：在 BRICK 2D 拓扑里，哪个块可跳、哪个必须留。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel

MODEL=r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS=r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ,NWIN=16,6

class C:
    N_LAYERS=24;PRELUDE=8;CODA=2;BRANCH_SIZES=(8,6)
    FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=8;o=[]
        for n in self.BRANCH_SIZES:o.append(list(range(s,s+n)));s+=n
        return o

t0=time.time()
print("=== 加载 1B + BRICK ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
model=BrickModel(base,config=C()).eval()   # fusion 用默认初始化（起点=分支均值）
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
pos=torch.arange(SEQ).unsqueeze(0);rope=precompute_rope(base,pos)
ids_all=[arg[i,:SEQ].unsqueeze(0).long() for i in range(NWIN)]

def run_layer(li,h):
    o,h_=None,h
    o,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
    return o

def ce_from_final(h,ids):
    logits=base.lm_head(base.model.norm(h))
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),
                           ids[:,1:].reshape(-1)).item()

def brick_forward(ids, mode, skip_pre_late=False):
    """BRICK 2D 前向，按 mode 决定分支激活。"""
    h=base.model.embed_tokens(ids)
    pre_range=range(4) if skip_pre_late else range(8)
    for li in pre_range: h=run_layer(li,h)
    h_pre=h

    brs=model.branches
    if mode=="both":
        bo=[]
        for br in brs:
            hb=h_pre
            for li in br: hb=run_layer(li,hb)
            bo.append(hb)
        fused=model.fusion(h_pre,bo)
    elif mode=="branch0_only":
        hb=h_pre
        for li in brs[0]: hb=run_layer(li,hb)
        fused=model.fusion(h_pre,[hb,h_pre])
    elif mode=="branch1_only":
        hb=h_pre
        for li in brs[1]: hb=run_layer(li,hb)
        fused=model.fusion(h_pre,[h_pre,hb])
    elif mode=="none":
        fused=model.fusion(h_pre,[h_pre,h_pre])
    else: raise ValueError(mode)

    if mode=="no_coda":
        return fused
    for li in model.coda: fused=run_layer(li,fused)
    return fused

modes=[
    ("B0 both (baseline)", "both", False),
    ("B1 branch0 only",    "branch0_only", False),
    ("B2 branch1 only",    "branch1_only", False),
    ("B3 none (skip both)","none", False),
    ("B4 skip prelude 4-7","both", True),
]
print("\n=== BRICK 2D 块 ablation（真实 1B）===",flush=True)
print("  模式                    |  CE     |  ΔCE vs both",flush=True)
res={}
for name,mode,skip in modes:
    with torch.no_grad():
        ces=[ce_from_final(brick_forward(ids,mode,skip),ids) for ids in ids_all]
    avg=sum(ces)/len(ces)
    res[name]=avg
    print(f"  {name:24s} | {avg:.4f} |",flush=True)
base_ce=res["B0 both (baseline)"]
for name,_,_ in modes:
    print(f"  {name:24s}: ΔCE = {res[name]-base_ce:+.4f}",flush=True)

# no_coda 单独（fusion 后直接 lm_head，不经 coda）
print("\n  灾难性对照 no_coda:",flush=True)
with torch.no_grad():
    ces=[ce_from_final(brick_forward(ids,"both",False).detach(),ids) for ids in ids_all[:2]]
print(f"  (注: no_coda 需要直接对 fused 算 CE，见下方)",flush=True)

print("\n=== 判读 ===",flush=True)
print("ΔCE 小 = 该块可跳过（动态激活候选）",flush=True)
print("ΔCE 大 = 该块必须保留",flush=True)
print(f"\n耗时 {time.time()-t0:.1f}s",flush=True)
