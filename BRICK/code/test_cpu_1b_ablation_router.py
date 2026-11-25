# -*- coding: utf-8 -*-
"""真实 1B：三种 oracle 的端到端 CE 对照（tiny 差异太小，必须真实模型）。

流程：
  1. 生成 repr / CE / ablation 三种 oracle 标签
  2. 各训一个 Gumbel router（80步）
  3. 端到端 CE：router 选分支 → 只跑该分支 → 测 CE
判据：端到端 CE 越低越好。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel
from brick_router import oracle_labels, ce_oracle, ablation_oracle
from brick_router_gumbel import GumbelRouter

MODEL=r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS=r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ,NWIN,STEPS=16,4,50

class C:
    N_LAYERS=24;PRELUDE=8;CODA=2;BRANCH_SIZES=(8,6);FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=8;o=[]
        for n in self.BRANCH_SIZES:o.append(list(range(s,s+n)));s+=n
        return o

t0=time.time()
print("=== 加载 1B + BRICK ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
model=BrickModel(base,config=C()).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
pos=torch.arange(SEQ).unsqueeze(0);rope=precompute_rope(base,pos)
ph,th,tk=[],[],[]
with torch.no_grad():
    for i in range(NWIN):
        ids=arg[i,:SEQ].unsqueeze(0).long()
        h=base.model.embed_tokens(ids)
        for li in range(8):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        ph.append(h[0].clone())
        for li in range(8,22):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        th.append(h[0].clone());tk.append(ids[0].clone())
ph,th,tk=torch.stack(ph),torch.stack(th),torch.stack(tk)
print(f"  数据 {tuple(ph.shape)} 耗时 {time.time()-t0:.1f}s",flush=True)

def branch_outs(h):
    out=[]
    for br in model.branches:
        hb=h
        for li in br:hb,_=call_layer(model.layers[li],hb,position_ids=pos,position_embeddings=rope)
        out.append(hb)
    return out

print("\n=== 生成三种 oracle（真实 1B）===",flush=True)
with torch.no_grad():
    bo=branch_outs(ph)
    lab_repr=oracle_labels(bo,th)
    lab_ce=ce_oracle(base,model.layers,bo,tk,pos,rope,model.coda)
    lab_abl,stats=ablation_oracle(base,model.layers,ph,model.branches,tk,pos,rope,model.coda,model.fusion)
print(f"  repr: {torch.bincount(lab_repr.flatten(),minlength=2).float().div(lab_repr.numel()).tolist()}",flush=True)
print(f"  CE  : {torch.bincount(lab_ce.flatten(),minlength=2).float().div(lab_ce.numel()).tolist()}",flush=True)
print(f"  abl : {stats['label_dist']}",flush=True)
print(f"  abl CE: both={stats['ce_both']:.4f} skip_b0={stats['ce_skip_b0']:.4f} skip_b1={stats['ce_skip_b1']:.4f}",flush=True)

def e2e_ce(router):
    ces=[]
    with torch.no_grad():
        for i in range(NWIN):
            h=ph[i:i+1];ids=tk[i:i+1]
            choose=router.choose(h,total_steps=STEPS)
            c=int(choose[0,0].item())
            bo=branch_outs(h)
            fused=model.fusion(h,[bo[c] if c==k else h for k in range(2)])
            hc=fused
            for li in model.coda:hc,_=call_layer(model.layers[li],hc,position_ids=pos,position_embeddings=rope)
            logits=base.lm_head(base.model.norm(hc))
            ces.append(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),ids[:,1:].reshape(-1)).item())
    return sum(ces)/len(ces)

# both 基线 CE（融合两分支，不跳）
with torch.no_grad():
    bo=branch_outs(ph); fused=model.fusion(ph,bo); hc=fused
    for li in model.coda:hc,_=call_layer(model.layers[li],hc,position_ids=pos,position_embeddings=rope)
    lg=base.lm_head(base.model.norm(hc))
    both_ce=F.cross_entropy(lg[:,:-1].float().reshape(-1,lg.size(-1)),tk[:,1:].reshape(-1)).item()
print(f"\n  both 基线 CE = {both_ce:.4f}",flush=True)

print("\n=== 三种 router（Gumbel）===",flush=True)
for name,label in [("repr",lab_repr),("ce",lab_ce),("ablation",lab_abl)]:
    torch.manual_seed(13)
    router=GumbelRouter(base.config.hidden_size,2,tau_start=1.0,tau_end=0.1)
    opt=torch.optim.AdamW(router.parameters(),lr=3e-4)
    for step in range(STEPS):
        sel=torch.randint(0,NWIN,(2,))
        soft,logits=router(ph[sel],hard=True,total_steps=STEPS)
        loss=F.cross_entropy(logits.reshape(-1,2),label[sel].reshape(-1))
        opt.zero_grad();loss.backward();opt.step();router.step+=1
    dist=torch.bincount(router.choose(ph,total_steps=STEPS).flatten(),minlength=2).float()
    dist=(dist/dist.sum()).tolist()
    ce=e2e_ce(router)
    print(f"  {name:10s}: 分布={[round(x,2) for x in dist]}  端到端CE={ce:.4f}  vs both {ce-both_ce:+.4f}",flush=True)

print(f"\n耗时 {time.time()-t0:.1f}s",flush=True)
