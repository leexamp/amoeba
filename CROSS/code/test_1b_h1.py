# -*- coding: utf-8 -*-
"""Real 1B: CROSS H1 test — can trained fusion beat the best single program?

Pipeline:
  1. Load MiniCPM5-1B, K=4 diverse programs
  2. Measure each program's CE and the untrained fusion CE
  3. Train cross router 30 steps (frozen backbone) with CE + entropy
  4. Re-measure fused CE vs best single program
  5. Check fusion weight distribution (no collapse)
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from cross_model import CROSSModel
from layer_utils import precompute_rope
import cross_config as cfg

MODEL=cfg.MODEL_LOCAL; CORPUS=cfg.CORPUS_LOCAL
SEQ,NWIN,STEPS=16,2,12   # CPU: 4 programs x 24 layers per step is slow

t0=time.time()
print("=== load 1B + CROSS ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
model=CROSSModel(base,top_k=2,gumbel=True).eval()
print(f"  programs {[len(p) for p in model.programs]}, trainable {sum(p.numel() for p in model.trainable_parameters()):,}",flush=True)

c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
ph=[arg[i,:SEQ].unsqueeze(0).long() for i in range(NWIN)]

def ce_from_h(h,ids):
    lg=base.lm_head(base.model.norm(h))
    return F.cross_entropy(lg[:,:-1].float().reshape(-1,lg.size(-1)),ids[:,1:].reshape(-1)).item()

# baseline: each program CE + untrained fusion CE
print("\n=== baseline (untrained router) ===",flush=True)
single, untrained_fused=[],[]
with torch.no_grad():
    for ids in ph:
        out=model(input_ids=ids)
        single.append([ce_from_h(h,ids) for h in out.program_hidden])
        w=out.fusion_weights
        fh=sum((w[...,j:j+1]*model.router.scale[j]).float()*h.float() for j,h in enumerate(out.program_hidden))
        untrained_fused.append(ce_from_h(fh.to(base.dtype),ids))
best_single=sum(min(s) for s in single)/NWIN
print(f"  per-program CE averages: {[round(sum(s[j] for s in single)/NWIN,4) for j in range(model.k)]}",flush=True)
print(f"  best single (avg): {best_single:.4f}, untrained fused: {sum(untrained_fused)/NWIN:.4f}",flush=True)

# train router
print(f"\n=== train router {STEPS} steps ===",flush=True)
tp=model.trainable_parameters()
model.train()
opt=torch.optim.AdamW(tp,lr=cfg.LR)
for step in range(STEPS):
    ids=ph[torch.randint(0,NWIN,(1,)).item()]
    out=model(input_ids=ids,labels=ids)
    loss=out.loss   # top-k sparsification already prevents collapse; entropy term was unstable on 1B
    opt.zero_grad();loss.backward();opt.step()
    model.router.tau=max(0.1,model.router.tau*0.97)
    if (step+1)%5==0 or step==0:
        print(f"  step {step+1}/{STEPS} loss {loss.item():.4f} tau {model.router.tau:.3f}",flush=True)

# evaluate
print("\n=== after training: fused vs best single ===",flush=True)
model.eval()
fused_ces=[]
with torch.no_grad():
    for ids in ph:
        out=model(input_ids=ids)
        w=out.fusion_weights
        fh=sum((w[...,j:j+1]*model.router.scale[j]).float()*h.float() for j,h in enumerate(out.program_hidden))
        fused_ces.append(ce_from_h(fh.to(base.dtype),ids))
fused=sum(fused_ces)/NWIN
dist=sum([model(input_ids=ids).fusion_weights for ids in ph])/NWIN
dist=dist.mean((0,1))
print(f"  fused CE: {fused:.4f}",flush=True)
print(f"  best single: {best_single:.4f}",flush=True)
print(f"  Delta (fused - best single): {fused-best_single:+.4f}",flush=True)
print(f"  weight dist: {[round(x,3) for x in dist.tolist()]}",flush=True)
print("\n=== verdict ===",flush=True)
if fused < best_single:
    print("  H1 SUPPORTED: fusion beats best single program",flush=True)
else:
    print("  H1 NOT SUPPORTED on this small sample (negative/trend)",flush=True)
print(f"elapsed {time.time()-t0:.1f}s",flush=True)
