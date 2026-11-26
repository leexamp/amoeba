# -*- coding: utf-8 -*-
"""Real 1B: does select-before-execute save compute without losing CE?

Two-stage:
  Stage 1: train full CROSS (all K programs), get router weights.
  Stage 2: distill selector from router weights.
  Stage 3: evaluate select-before-execute mode (only top-k programs run).

Compare:
  - full CROSS CE (all 4 programs)
  - select-before-execute CE (only 2 programs)
  - single best program CE
  - actual layers executed (compute proxy)
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from cross_model import CROSSModel
from cross_selector import ProgramSelector
import cross_config as cfg

MODEL=cfg.MODEL_LOCAL; CORPUS=cfg.CORPUS_LOCAL
SEQ,NWIN,STEPS,DSTEPS=16,2,6,4   # CPU: 4 programs x 24 layers per step is slow

t0=time.time()
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
ph=[arg[i,:SEQ].unsqueeze(0).long() for i in range(NWIN)]

def ce_of_logits(logits,ids):
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),ids[:,1:].reshape(-1)).item()

# ---- Stage 1: train full CROSS ----
print("=== Stage 1: full CROSS router ===",flush=True)
full=CROSSModel(base,top_k=2,gumbel=True).eval()
full.train(); opt=torch.optim.AdamW(full.trainable_parameters(),lr=3e-4)
for step in range(STEPS):
    ids=ph[step%NWIN]
    out=full(input_ids=ids,labels=ids)
    opt.zero_grad();out.loss.backward();opt.step()
    full.router.tau=max(0.1,full.router.tau*0.97)
    if (step+1)%4==0: print(f"  step {step+1}/{STEPS} loss {out.loss.item():.4f}",flush=True)
full.eval()

# router weights on training windows -> distillation targets
router_w=[]
with torch.no_grad():
    for ids in ph:
        router_w.append(full(input_ids=ids).fusion_weights)
router_w=torch.cat(router_w,dim=1)   # [B*NWIN,T,K]

# ---- Stage 2: distill selector ----
print("\n=== Stage 2: distill selector from router weights ===",flush=True)
H=base.config.hidden_size
selector=ProgramSelector(H,full.k).to(base.dtype)
sopt=torch.optim.AdamW(selector.parameters(),lr=1e-3)
# prelude hidden as selector input
from layer_utils import call_layer,precompute_rope
pos=torch.arange(SEQ).unsqueeze(0)
rope=precompute_rope(base,pos)
sel_inputs=[]
with torch.no_grad():
    for ids in ph:
        h=base.model.embed_tokens(ids)
        for li in range(4):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        sel_inputs.append(h.mean(dim=1))
sel_inputs=torch.cat(sel_inputs,dim=0).to(base.dtype)  # [NWIN,H], dtype matched to selector

for step in range(DSTEPS):
    loss,agree=selector.distill_loss(sel_inputs.unsqueeze(1),router_w.reshape(NWIN,SEQ,full.k)[:,0,:],top_k=2)
    sopt.zero_grad();loss.backward();sopt.step()
    if (step+1)%2==0: print(f"  step {step+1}/{DSTEPS} loss {loss.item():.4f} agree {agree:.2f}",flush=True)

# ---- Stage 3: compare ----
print("\n=== Stage 3: full vs select-before-execute ===",flush=True)
# cache full forward once per window (avoid re-running 4 programs repeatedly)
with torch.no_grad():
    full_logits=[full(input_ids=ids).logits for ids in ph]
full_ce=sum(ce_of_logits(lg,ids) for lg,ids in zip(full_logits,ph))/NWIN

sel_model=CROSSModel(base,top_k=2,gumbel=True,select_before_execute=True,selector_top_k=2).eval()
sel_model.selector.load_state_dict(selector.state_dict())
sel_model.selector = sel_model.selector.to(base.dtype)   # ensure dtype after load
sel_model.router = sel_model.router.to(base.dtype)
sel_model.selector.eval()
with torch.no_grad():
    sel_logits=[sel_model(input_ids=ids).logits for ids in ph]
sel_ce=sum(ce_of_logits(lg,ids) for lg,ids in zip(sel_logits,ph))/NWIN

full_layers=sum(len(p) for p in full.programs)
selected_idx=torch.topk(selector(sel_inputs[0:1]),2,dim=-1).indices[0].tolist()
sel_layers=sum(len(full.programs[j]) for j in selected_idx)
print(f"  selected programs: {selected_idx}",flush=True)
print(f"  full CROSS CE: {full_ce:.4f}  ({full_layers} layer-executions)",flush=True)
print(f"  select-before-execute CE: {sel_ce:.4f}  ({sel_layers} layer-executions)",flush=True)
print(f"  CE delta (select - full): {sel_ce-full_ce:+.4f}",flush=True)
print(f"  compute saving: {1-sel_layers/full_layers:.1%} fewer layer executions",flush=True)
print(f"\nverdict: ", "selector saves compute" if sel_ce-full_ce<0.3 else "selector loses too much CE",flush=True)
print(f"elapsed {time.time()-t0:.1f}s",flush=True)
