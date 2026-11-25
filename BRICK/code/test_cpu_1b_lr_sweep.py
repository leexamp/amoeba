# -*- coding: utf-8 -*-
"""真实 1B：fusion 训练的 LR 扫描（本机 CPU）。

背景：alpha=1 + proj零初始化，lr=3e-4 会爆炸（MSE 5.72→19.18）。
扫 LR ∈ {1e-5, 3e-5, 1e-4}，8步，看哪个稳定下降。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel

MODEL=r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS=r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ,NWIN,STEPS=16,3,8

class C:
    N_LAYERS=24;PRELUDE=8;CODA=2;BRANCH_SIZES=(8,6);FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=8;o=[]
        for n in self.BRANCH_SIZES:o.append(list(range(s,s+n)));s+=n
        return o

t0=time.time()
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
pos=torch.arange(SEQ).unsqueeze(0);rope=precompute_rope(base,pos)
ph,th=[],[]
with torch.no_grad():
    for i in range(NWIN):
        ids=arg[i,:SEQ].unsqueeze(0).long();h=base.model.embed_tokens(ids)
        for li in range(8):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        ph.append(h[0].clone())
        for li in range(8,22):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        th.append(h[0].clone())
ph,th=torch.stack(ph),torch.stack(th)

def bo_of(model,h):
    out=[]
    for br in model.branches:
        hb=h
        for li in br:hb,_=call_layer(model.layers[li],hb,position_ids=pos,position_embeddings=rope)
        out.append(hb)
    return out

print("=== LR sweep (real 1B, alpha=1 + proj零初始化) ===",flush=True)
print("  lr      | start  |  step1  |  step4  |  step8  | alpha",flush=True)
for lr in [1e-5,3e-5,1e-4]:
    torch.manual_seed(7)
    model=BrickModel(base,config=C()).eval();model.fusion.train()
    for p in model.fusion.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.fusion.parameters(),lr=lr)
    with torch.no_grad():
        bo=bo_of(model,ph);f0=model.fusion(ph,bo)
        start=F.mse_loss(f0.float(),th.float()).item()
    mids={}
    for step in range(STEPS):
        s=torch.randint(0,NWIN,(1,));h,t=ph[s],th[s]
        fused=model.fusion(h,bo_of(model,h))
        loss=F.mse_loss(fused.float(),t.float())
        opt.zero_grad();loss.backward();opt.step()
        if step in (0,3,7):
            with torch.no_grad():
                mids[step]=F.mse_loss(model.fusion(ph,bo_of(model,ph)).float(),th.float()).item()
    print(f"  {lr:.0e}  | {start:6.2f} | {mids[0]:7.2f} | {mids[3]:7.2f} | {mids[7]:7.2f} | {model.fusion.alpha.item():+.3f}",flush=True)
    del model,opt

print(f"\n耗时 {time.time()-t0:.1f}s",flush=True)
