# -*- coding: utf-8 -*-
"""出路 1：让 fusion 训练时暴露"单分支输入" —— 决定性对照实验。

假设：之前 fusion 只在 both 上训练，所以没见过"只跑一个分支"的分布，
      导致动态激活 CE 高 0.40。如果训练时混入单分支样本 + CE 监督，
      fusion 可能学会补偿缺失分支。

对照：
  A. fusion 只在 both 上训（已做：端到端 CE 4.4435）
  B. fusion 在 both/skip_b0/skip_b1 三种组合上训（本实验）
     训练时随机采样组合，CE 监督（经 coda+lm_head）
  C. 混合：50% both + 50% 单分支

然后冻结 fusion，用同样的 ablation oracle + Gumbel router，测端到端 CE。
判据：B/C 的 router 端到端 CE 是否比 A（4.4435）低，能否接近 both（4.041）。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel
from brick_router import ablation_oracle
from brick_router_gumbel import GumbelRouter

MODEL=r"E:\AMOEBA_MODELS\MiniCPM5-1B"
CORPUS=r"E:\AMOEBA_RESEARCH\chat_corpus_1b.pt"
SEQ,NWIN,FSTEPS,RSTEPS=16,3,12,20   # CPU bf16 约15s/步，12步够看趋势

class C:
    N_LAYERS=24;PRELUDE=8;CODA=2;BRANCH_SIZES=(8,6);FUSION_MODE="residual_branch_avg"
    def branches(self):
        s=8;o=[]
        for n in self.BRANCH_SIZES:o.append(list(range(s,s+n)));s+=n
        return o

t0=time.time()
print("=== 加载 1B + BRICK ===",flush=True)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).eval()
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

def branch_outs(model,h):
    out=[]
    for br in model.branches:
        hb=h
        for li in br:hb,_=call_layer(model.layers[li],hb,position_ids=pos,position_embeddings=rope)
        out.append(hb)
    return out

def ce_of_h(h,ids):
    hc=h
    for li in C().branches()[:0]+list(range(22,24)):
        hc,_=call_layer(base.model.layers[li],hc,position_ids=pos,position_embeddings=rope)
    lg=base.lm_head(base.model.norm(hc))
    return F.cross_entropy(lg[:,:-1].float().reshape(-1,lg.size(-1)),ids[:,1:].reshape(-1)).item()

def train_and_eval(strategy, name):
    """strategy: 'both' | 'mixed' | 'single_only'"""
    torch.manual_seed(7)
    model=BrickModel(base,config=C()).eval()
    model.fusion.train()
    for p in model.fusion.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.fusion.parameters(),lr=3e-5)

    print(f"\n=== [{name}] fusion 训练 {FSTEPS}步 strategy={strategy} ===",flush=True)
    for step in range(FSTEPS):
        s=torch.randint(0,NWIN,(1,))
        h,t,y=ph[s],th[s],tk[s]
        bo=branch_outs(model,h)
        # 采样组合
        if strategy=="both": combo=0
        elif strategy=="single_only": combo=1+torch.randint(0,2,(1,)).item()
        else: combo=torch.randint(0,3,(1,)).item()   # mixed: 0/1/2 均匀
        if combo==0: inp=[bo[0],bo[1]]
        elif combo==1: inp=[h,bo[1]]      # 跳 b0
        else: inp=[bo[0],h]               # 跳 b1
        fused=model.fusion(h,inp)
        # 损失 = MSE(对教师)。CE 评估在训练后做（CPU 上 lm_head 130k 词汇太慢）
        loss=F.mse_loss(fused.float(),t.float())
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%15==0 or step==0:
            print(f"  step {step+1}/{FSTEPS} loss {loss.item():.4f} combo{combo}",flush=True)
    model.fusion.eval()
    for p in model.fusion.parameters():p.requires_grad_(False)

    # both CE
    with torch.no_grad():
        bo=branch_outs(model,ph)
        both_ce=sum(ce_of_h(model.fusion(ph,bo)[i:i+1],tk[i:i+1]) for i in range(NWIN))/NWIN

    # ablation oracle + router
    with torch.no_grad():
        label,stats=ablation_oracle(base,model.layers,ph,model.branches,tk,pos,rope,model.coda,model.fusion)
    torch.manual_seed(13)
    router=GumbelRouter(base.config.hidden_size,2,tau_start=1.0,tau_end=0.1)
    ropt=torch.optim.AdamW(router.parameters(),lr=3e-4)
    for step in range(RSTEPS):
        s=torch.randint(0,NWIN,(2,))
        soft,logits=router(ph[s],hard=True,total_steps=RSTEPS)
        loss=F.cross_entropy(logits.reshape(-1,2),label[s].reshape(-1))
        ropt.zero_grad();loss.backward();ropt.step();router.step+=1
    ces=[]
    with torch.no_grad():
        for i in range(NWIN):
            h=ph[i:i+1];ids=tk[i:i+1]
            cc=int(router.choose(h,total_steps=RSTEPS)[0,0].item())
            bo=branch_outs(model,h)
            fused=model.fusion(h,[bo[cc] if cc==k else h for k in range(2)])
            ces.append(ce_of_h(fused,ids))
    e2e=sum(ces)/len(ces)
    dist=torch.bincount(router.choose(ph,total_steps=RSTEPS).flatten(),minlength=2).float()
    dist=(dist/dist.sum()).tolist()
    print(f"  [{name}] both CE={both_ce:.4f}  router CE={e2e:.4f} Δ={e2e-both_ce:+.4f} 分布={[round(x,2) for x in dist]}",flush=True)
    print(f"  [{name}] ablation CE: both={stats['ce_both']:.3f} skip_b0={stats['ce_skip_b0']:.3f} skip_b1={stats['ce_skip_b1']:.3f}",flush=True)
    return both_ce,e2e

print("=== 出路1 对照：fusion 训练时暴露单分支输入 ===",flush=True)
r_both=train_and_eval("both","A_both_only")
r_mixed=train_and_eval("mixed","B_mixed")
r_single=train_and_eval("single_only","C_single_only")

print("\n=== 判读 ===",flush=True)
print(f"  A(both-only):    router CE {r_both[1]:.4f}, Δ {r_both[1]-r_both[0]:+.4f}",flush=True)
print(f"  B(mixed):        router CE {r_mixed[1]:.4f}, Δ {r_mixed[1]-r_mixed[0]:+.4f}",flush=True)
print(f"  C(single-only):  router CE {r_single[1]:.4f}, Δ {r_single[1]-r_single[0]:+.4f}",flush=True)
print("  Δ 越小（越接近0）= 动态激活越接近 both = 出路1 越有效",flush=True)
print(f"\n耗时 {time.time()-t0:.1f}s",flush=True)
