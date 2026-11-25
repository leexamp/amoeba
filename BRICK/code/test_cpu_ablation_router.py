# -*- coding: utf-8 -*-
"""对照：repr oracle / CE oracle / ablation oracle 三种 router 监督信号。

关键区别：
  repr oracle  : 哪个分支表示更接近教师（已证伪：100% 偏深分支）
  ce oracle    : 哪个分支的 CE 更低（标签均衡，但不是 early-exit 逻辑）
  ablation oracle: 跳过哪个分支的 CE 代价更小（★真正的动态激活逻辑）

判据不只是标签分布/一致率，而是**端到端 CE**：
  用 router 选分支 → 只跑选中的分支 → 测 CE。哪个 router 选得好，CE 就低。

本机 tiny 模型验证（快），然后真实 1B 小规模验证。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel
from brick_router import BrickRouter, oracle_labels, ce_oracle, ablation_oracle
from brick_router_gumbel import GumbelRouter

torch.manual_seed(42)
CONF=LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=12,
                 num_attention_heads=4,num_key_value_heads=2,vocab_size=1000,
                 max_position_embeddings=128,use_cache=True)
SEQ,NWIN,STEPS=32,16,80

class T:
    N_LAYERS=12;PRELUDE=4;CODA=2;BRANCH_SIZES=(4,2);FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=4;o=[]
        for n in self.BRANCH_SIZES:o.append(list(range(s,s+n)));s+=n
        return o

base=LlamaForCausalLM(CONF).eval()
model=BrickModel(base,config=T()).eval()

# 语料
pos=torch.arange(SEQ).unsqueeze(0);rope=precompute_rope(base,pos)
ph,th,tk=[],[],[]
with torch.no_grad():
    for _ in range(NWIN):
        ids=torch.randint(0,1000,(1,SEQ))
        h=base.model.embed_tokens(ids)
        for li in range(4):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        ph.append(h[0].clone())
        for li in range(4,10):h,_=call_layer(base.model.layers[li],h,position_ids=pos,position_embeddings=rope)
        th.append(h[0].clone());tk.append(ids[0].clone())
ph,th,tk=torch.stack(ph),torch.stack(th),torch.stack(tk)

def branch_outs(h):
    out=[]
    for br in model.branches:
        hb=h
        for li in br:hb,_=call_layer(model.layers[li],hb,position_ids=pos,position_embeddings=rope)
        out.append(hb)
    return out

def end_to_end_ce(router, is_gumbel=False):
    """用 router 选分支 → 只跑该分支 → 测 CE（真正的判据）。"""
    ces=[]
    with torch.no_grad():
        for i in range(NWIN):
            h=ph[i:i+1];ids=tk[i:i+1]
            choose=router.choose(h,total_steps=STEPS) if is_gumbel else router(h).argmax(-1)
            bo=branch_outs(h)
            # 用选中的分支，另一个用 h_pre 占位
            sel=[bo[k] if choose[0,j]==k else h for k in range(2) for j in range(1)][0:2]
            # 简化：整段用一个选择（token-level 在 tiny 上用首 token 决策）
            c=int(choose[0,0].item())
            fused=model.fusion(h,[bo[c] if c==k else h for k in range(2)])
            hc=fused
            for li in model.coda:hc,_=call_layer(model.layers[li],hc,position_ids=pos,position_embeddings=rope)
            logits=base.lm_head(base.model.norm(hc))
            ce=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),ids[:,1:].reshape(-1)).item()
            ces.append(ce)
    return sum(ces)/len(ces)

# 生成三种 oracle 标签
print("=== 生成三种 oracle 标签 ===",flush=True)
with torch.no_grad():
    bo_all=branch_outs(ph)
    lab_repr=oracle_labels(bo_all,th)
    lab_ce=ce_oracle(base,model.layers,bo_all,tk,pos,rope,model.coda)
    lab_abl,stats=ablation_oracle(base,model.layers,ph,model.branches,tk,pos,rope,model.coda,model.fusion)
print(f"  repr 标签分布: {torch.bincount(lab_repr.flatten(),minlength=2).float().div(lab_repr.numel()).tolist()}",flush=True)
print(f"  CE   标签分布: {torch.bincount(lab_ce.flatten(),minlength=2).float().div(lab_ce.numel()).tolist()}",flush=True)
print(f"  abl  标签分布: {stats['label_dist']}",flush=True)
print(f"  abl CE: both={stats['ce_both']:.3f}, skip_b0={stats['ce_skip_b0']:.3f}, skip_b1={stats['ce_skip_b1']:.3f}",flush=True)

print("\n=== 三种 router 训练（Gumbel，防塌缩）===",flush=True)
results={}
for name,label in [("repr",lab_repr),("ce",lab_ce),("ablation",lab_abl)]:
    torch.manual_seed(13)
    router=GumbelRouter(128,2,tau_start=1.0,tau_end=0.1)
    opt=torch.optim.AdamW(router.parameters(),lr=3e-4)
    for step in range(STEPS):
        sel=torch.randint(0,NWIN,(4,))
        soft,logits=router(ph[sel],hard=True,total_steps=STEPS)
        loss=F.cross_entropy(logits.reshape(-1,2),label[sel].reshape(-1))
        opt.zero_grad();loss.backward();opt.step();router.step+=1
    dist=torch.bincount(router.choose(ph,total_steps=STEPS).flatten(),minlength=2).float()
    dist=(dist/dist.sum()).tolist()
    e2e=end_to_end_ce(router,is_gumbel=True)
    results[name]=(dist,e2e)
    print(f"  {name:10s}: router分布={[round(x,2) for x in dist]}  端到端CE={e2e:.4f}",flush=True)

print("\n=== 判据：端到端 CE 越低越好 ===",flush=True)
best=min(results,key=lambda k:results[k][1])
print(f"最优 router 监督: {best} (CE={results[best][1]:.4f})",flush=True)
