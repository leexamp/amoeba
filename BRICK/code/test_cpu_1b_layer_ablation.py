# -*- coding: utf-8 -*-
"""真实 1B：逐层/逐块 ablation —— 哪一层（块）对性能影响最大？

方法：跳过某层 l（残差直连 h_out = h_in），测 next-token CE 上升量 ΔCE_l。
ΔCE 越大 = 该层越关键（不可跳过）；ΔCE≈0 = 冗余层（动态激活候选）。

两组实验：
  A. 1D 逐层（24 层）—— 找最关键的原始层
  B. BRICK 块（prelude 子段 / branch0 / branch1 / coda）—— 找最关键的块

输出按 ΔCE 排序，并标注与 Amoeba 发现（最后一层独占 1.3 nats）的关系。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer, precompute_rope

MODEL=r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS=r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ,NWIN=16,6

t0=time.time()
print("=== 加载 1B ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
pos=torch.arange(SEQ).unsqueeze(0);rope=precompute_rope(base,pos)

# 取真实窗口
ids_all=[]
for i in range(NWIN):
    ids_all.append(arg[i,:SEQ].unsqueeze(0).long())

def ce_from_h(h, targets):
    """从最后隐藏状态算 CE。"""
    logits=base.lm_head(base.model.norm(h))
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),
                           targets[:,1:].reshape(-1)).item()

def forward_skip(skip_layers, ids):
    """顺序跑 24 层，跳过 skip_layers 中的层（残差直连）。"""
    h=base.model.embed_tokens(ids)
    for li in range(24):
        if li in skip_layers: continue
        h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
    return h

# 基线 CE
with torch.no_grad():
    base_ce=[ce_from_h(forward_skip(set(),ids),ids) for ids in ids_all]
print(f"基线 CE = {sum(base_ce)/len(base_ce):.4f} (n={NWIN})",flush=True)

print("\n=== A. 1D 逐层 ablation（跳过第 l 层）===",flush=True)
results=[]
for l in range(24):
    with torch.no_grad():
        ces=[ce_from_h(forward_skip({l},ids),ids) for ids in ids_all]
    dce=sum(ces)/len(ces)-sum(base_ce)/len(base_ce)
    results.append((l,dce))
    print(f"  layer {l:2d}: ΔCE = {dce:+.4f}",flush=True)

print("\n按 |ΔCE| 排序（越大越关键）:",flush=True)
for l,d in sorted(results,key=lambda x:-abs(x[1]))[:8]:
    print(f"  layer {l:2d}: ΔCE {d:+.4f}",flush=True)

print("\n=== B. BRICK 块 ablation（8+[8,6]+2）===",flush=True)
# BRICK 拓扑：prelude 层0-7, branch0 层8-15, branch1 层16-21, coda 层22-23
BRICKS={
    "prelude(0-7)": list(range(0,8)),
    "branch0(8-15)": list(range(8,16)),
    "branch1(16-21)": list(range(16,22)),
    "coda(22-23)": list(range(22,24)),
}
brick_res=[]
for name,layers in BRICKS.items():
    with torch.no_grad():
        ces=[ce_from_h(forward_skip(set(layers),ids),ids) for ids in ids_all]
    dce=sum(ces)/len(ces)-sum(base_ce)/len(base_ce)
    brick_res.append((name,dce))
    print(f"  skip {name:18s}: ΔCE = {dce:+.4f}",flush=True)

# 细分 prelude / branch
SUB={
    "pre_early(0-3)":[0,1,2,3], "pre_late(4-7)":[4,5,6,7],
    "b0_early(8-11)":[8,9,10,11], "b0_late(12-15)":[12,13,14,15],
    "b1_early(16-18)":[16,17,18], "b1_late(19-21)":[19,20,21],
    "coda_22":[22], "coda_23(final)":[23],
}
print("\n=== B2. 细分块 ===",flush=True)
sub_res=[]
for name,layers in SUB.items():
    with torch.no_grad():
        ces=[ce_from_h(forward_skip(set(layers),ids),ids) for ids in ids_all]
    dce=sum(ces)/len(ces)-sum(base_ce)/len(base_ce)
    sub_res.append((name,dce))
    print(f"  skip {name:18s}: ΔCE = {dce:+.4f}",flush=True)

print("\n=== 判读 ===",flush=True)
print("ΔCE 最大的层/块 = 最不可跳过（瓶颈）",flush=True)
print("ΔCE ≈ 0 的层/块 = 冗余（动态激活候选）",flush=True)
print(f"\n耗时 {time.time()-t0:.1f}s",flush=True)
