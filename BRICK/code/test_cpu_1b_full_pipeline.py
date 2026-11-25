# -*- coding: utf-8 -*-
"""真实 1B：BRICK 完整链条（本机 CPU，决定性实验）。

流程：
  1. 教师数据（prelude_h / teacher_h / tokens）
  2. A0 fusion 训练 50 步（LR=3e-5, residual_branch_avg）
  3. 冻结 fusion
  4. ablation oracle（用训练过的 fusion 算"跳过代价"）
  5. Gumbel router 训练 50 步
  6. 端到端 CE：router 选分支 → 只跑该分支 → 测 CE
  对比 both 基线（4.219 附近）—— 决定 BRICK 动态激活是否成立。

判据：
  - router 端到端 CE 接近 both（差 <0.3）且省一个分支计算 → 正面
  - router CE 仍高（>0.5）→ 负面：动态激活在冻结主干上不可行
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
SEQ,NWIN,FSTEPS,RSTEPS=16,4,50,50

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

def ce_of_h(h,ids):
    hc=h
    for li in model.coda:hc,_=call_layer(model.layers[li],hc,position_ids=pos,position_embeddings=rope)
    lg=base.lm_head(base.model.norm(hc))
    return F.cross_entropy(lg[:,:-1].float().reshape(-1,lg.size(-1)),ids[:,1:].reshape(-1)).item()

# ---------- 1. both 基线 ----------
with torch.no_grad():
    bo=branch_outs(ph)
    both_ce=sum(ce_of_h(model.fusion(ph,bo)[i:i+1],tk[i:i+1]) for i in range(NWIN))/NWIN
print(f"\n=== both 基线 CE = {both_ce:.4f} ===",flush=True)

# ---------- 2. fusion 训练 ----------
print(f"\n=== A0 fusion 训练 {FSTEPS} 步（LR=3e-5）===",flush=True)
model.fusion.train()
for p in model.fusion.parameters():p.requires_grad_(True)
opt=torch.optim.AdamW(model.fusion.parameters(),lr=3e-5)
for step in range(FSTEPS):
    s=torch.randint(0,NWIN,(1,))
    h,t=ph[s],th[s]
    fused=model.fusion(h,branch_outs(h))
    loss=F.mse_loss(fused.float(),t.float())
    opt.zero_grad();loss.backward();opt.step()
    if (step+1)%10==0 or step==0:
        print(f"  step {step+1}/{FSTEPS} MSE {loss.item():.4f} alpha {model.fusion.alpha.item():.4f}",flush=True)
model.fusion.eval()
for p in model.fusion.parameters():p.requires_grad_(False)
with torch.no_grad():
    bo=branch_outs(ph)
    both_ce_trained=sum(ce_of_h(model.fusion(ph,bo)[i:i+1],tk[i:i+1]) for i in range(NWIN))/NWIN
print(f"  训练后 both CE = {both_ce_trained:.4f} (起点 {both_ce:.4f})",flush=True)

# ---------- 3. ablation oracle（用训练过的 fusion）----------
print("\n=== ablation oracle（训练后的 fusion）===",flush=True)
with torch.no_grad():
    label,stats=ablation_oracle(base,model.layers,ph,model.branches,tk,pos,rope,model.coda,model.fusion)
print(f"  CE: both={stats['ce_both']:.4f} skip_b0={stats['ce_skip_b0']:.4f} skip_b1={stats['ce_skip_b1']:.4f}",flush=True)
print(f"  标签分布 {stats['label_dist']}",flush=True)

# ---------- 4. router 训练 ----------
print(f"\n=== Gumbel router 训练 {RSTEPS} 步 ===",flush=True)
torch.manual_seed(13)
router=GumbelRouter(base.config.hidden_size,2,tau_start=1.0,tau_end=0.1)
ropt=torch.optim.AdamW(router.parameters(),lr=3e-4)
for step in range(RSTEPS):
    s=torch.randint(0,NWIN,(2,))
    soft,logits=router(ph[s],hard=True,total_steps=RSTEPS)
    loss=F.cross_entropy(logits.reshape(-1,2),label[s].reshape(-1))
    ropt.zero_grad();loss.backward();ropt.step();router.step+=1
    if (step+1)%10==0 or step==0:
        print(f"  step {step+1}/{RSTEPS} loss {loss.item():.4f}",flush=True)

# ---------- 5. 端到端 CE ----------
print("\n=== 端到端 CE（router 选分支，只跑该分支）===",flush=True)
ces=[]
with torch.no_grad():
    for i in range(NWIN):
        h=ph[i:i+1];ids=tk[i:i+1]
        c=int(router.choose(h,total_steps=RSTEPS)[0,0].item())
        bo=branch_outs(h)
        fused=model.fusion(h,[bo[c] if c==k else h for k in range(2)])
        ces.append(ce_of_h(fused,ids))
router_ce=sum(ces)/len(ces)
dist=torch.bincount(router.choose(ph,total_steps=RSTEPS).flatten(),minlength=2).float()
dist=(dist/dist.sum()).tolist()
print(f"  router 分布 {[round(x,2) for x in dist]}",flush=True)
print(f"  router 端到端 CE = {router_ce:.4f}",flush=True)
print(f"  vs both(训练后) {both_ce_trained:.4f}: Δ = {router_ce-both_ce_trained:+.4f}",flush=True)

print("\n=== 判读 ===",flush=True)
if router_ce - both_ce_trained < 0.3:
    print("  ✅ 正面：动态激活 CE 接近 both，且省一个分支计算",flush=True)
else:
    print("  ❌ 负面：动态激活 CE 显著高于 both —— 冻结主干上不可行",flush=True)
print(f"\n总耗时 {time.time()-t0:.1f}s",flush=True)
