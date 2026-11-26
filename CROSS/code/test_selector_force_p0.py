# -*- coding: utf-8 -*-
"""Selector with P0 forced — does keeping the best single program fix the +0.93 CE loss?

The previous selector chose [1,2] and dropped P0 (identity, the best single program),
causing CE +0.93. Fix: force P0 in, choose top-(k-1) from the remaining programs.
Same compute saving (~50%), potentially much better CE.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from cross_model import CROSSModel
from cross_selector import ProgramSelector
from layer_utils import call_layer, precompute_rope
import cross_config as cfg

MODEL=cfg.MODEL_LOCAL; CORPUS=cfg.CORPUS_LOCAL
SEQ,NWIN,STEPS,DSTEPS=16,3,10,6

def ce(lg,ids):
    return F.cross_entropy(lg[:,:-1].float().reshape(-1,lg.size(-1)),ids[:,1:].reshape(-1)).item()

t0=time.time()
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
c=torch.load(CORPUS,map_location="cpu");arg=c["arg"]
ph=[arg[i,:SEQ].unsqueeze(0).long() for i in range(NWIN)]

# train full CROSS (teacher router)
full=CROSSModel(base,top_k=2,gumbel=True).eval()
full.train(); opt=torch.optim.AdamW(full.trainable_parameters(),lr=3e-4)
for s in range(STEPS):
    o=full(input_ids=ph[s%NWIN],labels=ph[s%NWIN])
    opt.zero_grad();o.loss.backward();opt.step();full.router.tau=max(0.1,full.router.tau*0.97)
full.eval()

with torch.no_grad():
    rw=torch.cat([full(input_ids=ids).fusion_weights for ids in ph],dim=1)
pos=torch.arange(SEQ).unsqueeze(0); rope=precompute_rope(base,pos)
sel_in=[]
with torch.no_grad():
    for ids in ph:
        h=base.model.embed_tokens(ids)
        for li in range(4):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        sel_in.append(h.mean(dim=1))
sel_in=torch.cat(sel_in,dim=0).to(base.dtype)
H=base.config.hidden_size
sel=ProgramSelector(H,full.k).to(base.dtype)
so=torch.optim.AdamW(sel.parameters(),lr=1e-3)
for s in range(DSTEPS):
    loss,agree=sel.distill_loss(sel_in.unsqueeze(1),rw.reshape(NWIN,SEQ,full.k)[:,0,:])
    so.zero_grad();loss.backward();so.step()
print(f"distill agree {agree:.2f}",flush=True)

# baseline: full CROSS CE
with torch.no_grad():
    full_ce=sum(ce(full(input_ids=ids).logits,ids) for ids in ph)/NWIN

# selectors
for force in [False, True]:
    sm=CROSSModel(base,top_k=2,gumbel=True,select_before_execute=True,
                  selector_top_k=2,force_program_0=force).eval()
    sm.selector.load_state_dict(sel.state_dict())
    sm.selector=sm.selector.to(base.dtype); sm.router=sm.router.to(base.dtype)
    with torch.no_grad():
        sce=sum(ce(sm(input_ids=ids).logits,ids) for ids in ph)/NWIN
    idx=torch.topk(sel(sel_in[0:1]),2,dim=-1).indices[0].tolist()
    if force: idx=[0]+[j for j in idx if j!=0][:1]
    layers=sum(len(full.programs[j]) for j in idx)
    print(f"force_P0={force}: selected {idx}, CE {sce:.4f}, layers {layers}, "
          f"delta vs full {sce-full_ce:+.4f}",flush=True)
print(f"elapsed {time.time()-t0:.1f}s",flush=True)
