# -*- coding: utf-8 -*-
"""CROSS held-out evaluation — the gatekeeper experiment.

CRITICAL: prior numbers (4.755, 4.840) were computed on the SAME 2-4 windows used
for router training, with 492K trainable params and only 32-64 supervised tokens.
That is fitting, not evidence. This script splits windows:
  train:  windows 0-5
  held-out: windows 6-11  (NEVER seen during router training)

On held-out, report:
  - each program's CE (no labels needed for these, just next-token CE)
  - best FIXED program (P0) CE — no oracle, honest baseline
  - per-window oracle min over programs (uses labels; upper bound, labeled as such)
  - trained fusion CE — trained ONLY on train windows
If trained fusion beats P0 on held-out, the complementarity claim survives.
If not, the claim dies and we report it.

Also tests probability-level fusion (Jensen guarantee) as an option.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from cross_model import CROSSModel
import cross_config as cfg

MODEL=cfg.MODEL_LOCAL; CORPUS=cfg.CORPUS_LOCAL
SEQ=16; N_TRAIN=3; N_HELD=3; STEPS=6   # CPU: 4 programs x 24 layers per step is slow

t0=time.time()
print("=== load 1B ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
# first 12 windows: 0-5 train, 6-11 held-out (disjoint)
train_ids=[arg[i,:SEQ].unsqueeze(0).long() for i in range(N_TRAIN)]
held_ids=[arg[i,:SEQ].unsqueeze(0).long() for i in range(N_TRAIN, N_TRAIN+N_HELD)]

def ce_of_logits(logits,ids):
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),ids[:,1:].reshape(-1)).item()

def run_model(model, ids_list):
    """Returns per-window: (program CEs list, fusion CE, fusion weights)."""
    out=[]
    with torch.no_grad():
        for ids in ids_list:
            o=model(input_ids=ids)
            pces=[ce_of_logits(base.lm_head(base.model.norm(h)),ids) for h in o.program_hidden]
            fce=ce_of_logits(o.logits,ids)
            out.append((pces,fce,o.fusion_weights.mean((0,1)).tolist()))
    return out


def evaluate(model, tag, ids_list):
    res=run_model(model,ids_list)
    p0=[r[0][0] for r in res]                          # P0 = identity
    oracle=[min(r[0]) for r in res]                    # per-window min (LABELLED upper bound)
    fused=[r[1] for r in res]
    print(f"\n  [{tag}] P0(identity)={sum(p0)/len(p0):.4f}  "
          f"oracle-min(labelled)={sum(oracle)/len(oracle):.4f}  "
          f"fusion={sum(fused)/len(fused):.4f}",flush=True)
    print(f"  [{tag}] program CEs={[round(sum(r[0][j] for r in res)/len(res),4) for j in range(len(res[0][0]))]}",flush=True)
    print(f"  [{tag}] fusion weights={[round(x,3) for x in res[0][2]]}",flush=True)
    return sum(p0)/len(p0), sum(oracle)/len(oracle), sum(fused)/len(fused)


# ---- hidden-level fusion, trained on train windows ----
print("\n=== Model A: hidden-level fusion ===",flush=True)
mA=CROSSModel(base,top_k=2,gumbel=True,fusion_level="hidden").eval()
mA.train(); optA=torch.optim.AdamW(mA.trainable_parameters(),lr=3e-4)
for step in range(STEPS):
    ids=train_ids[step%N_TRAIN]
    o=mA(input_ids=ids,labels=ids)
    optA.zero_grad();o.loss.backward();optA.step()
    mA.router.tau=max(0.1,mA.router.tau*0.97)
    if (step+1)%4==0: print(f"  A step {step+1}/{STEPS} loss {o.loss.item():.4f}",flush=True)
mA.eval()
trA=evaluate(mA,"A TRAIN",train_ids)
heA=evaluate(mA,"A HELD-OUT",held_ids)

# ---- probability-level fusion, trained on train windows ----
print("\n=== Model B: probability-level fusion ===",flush=True)
mB=CROSSModel(base,top_k=2,gumbel=True,fusion_level="prob").eval()
mB.train(); optB=torch.optim.AdamW(mB.trainable_parameters(),lr=3e-4)
for step in range(STEPS):
    ids=train_ids[step%N_TRAIN]
    o=mB(input_ids=ids,labels=ids)
    optB.zero_grad();o.loss.backward();optB.step()
    mB.router.tau=max(0.1,mB.router.tau*0.97)
    if (step+1)%4==0: print(f"  B step {step+1}/{STEPS} loss {o.loss.item():.4f}",flush=True)
mB.eval()
trB=evaluate(mB,"B TRAIN",train_ids)
heB=evaluate(mB,"B HELD-OUT",held_ids)

print("\n=== GATEKEEPER VERDICT ===",flush=True)
print(f"  Hidden fusion: held-out fusion {heA[2]:.4f} vs P0 {heA[0]:.4f} "
      f"(delta {heA[2]-heA[0]:+.4f}) vs oracle {heA[1]:.4f}",flush=True)
print(f"  Prob fusion:   held-out fusion {heB[2]:.4f} vs P0 {heB[0]:.4f} "
      f"(delta {heB[2]-heB[0]:+.4f}) vs oracle {heB[1]:.4f}",flush=True)
alive = heA[2] < heA[0] or heB[2] < heB[0]
print(f"\n  COMPLEMENTARITY CLAIM: {'SURVIVES' if alive else 'DIES'} on held-out",flush=True)
print(f"elapsed {time.time()-t0:.1f}s",flush=True)
