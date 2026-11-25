# -*- coding: utf-8 -*-
"""训练 BRICK fusion：冻结主干，只优化 fusion 权重。

损失 = MSE(fusion 输出, 教师中间表示) + λ CE(经 coda+lm_head 的 next-token)。
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer
from brick_model import BrickModel
import brick_config as cfg

def main():
    torch.manual_seed(cfg.SEED)
    device = "cuda"
    data = torch.load(cfg.TEACHER_DATA, map_location="cpu")
    ph, th, tk = data["prelude_h"], data["teacher_h"], data["tokens"]
    n = ph.shape[0]
    nsplit = data["n_train"]
    print(f"data: {n} windows (train {nsplit}, val {n-nsplit})", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL, dtype=torch.bfloat16, device_map=device).eval()
    base.config.use_cache = True
    model = BrickModel(base).to(device)
    # 冻结主干，只放开 fusion
    model.fusion.train()
    for p in model.fusion.parameters():
        p.requires_grad_(True)

    ntr = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {ntr:,} (frozen backbone: {sum(p.numel() for p in base.parameters()):,})", flush=True)

    opt = torch.optim.AdamW(model.fusion.parameters(), lr=cfg.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.STEPS, eta_min=cfg.LR*0.1)

    def batch(sel):
        h = ph[sel].to(device)          # prelude 输出作为分支输入
        t = th[sel].to(device)          # 教师目标
        y = tk[sel].to(device)          # token（CE 用）
        pos = torch.arange(h.shape[1], device=device).unsqueeze(0).expand(h.shape[0], -1)
        return h, t, y, pos

    print("\n[训练]", flush=True)
    for step in range(cfg.STEPS):
        sel = torch.randint(0, nsplit, (cfg.BATCH,))
        h, t, y, pos = batch(sel)

        # 分支前向（输入是 prelude_h，跳过 prelude）
        bo = []
        for br in model.branches:
            hb = h
            for li in br:
                hb, _ = call_layer(model.layers[li], hb, position_ids=pos)
            bo.append(hb)
        fused = model.fusion(h, bo)   # A0: h_pre=prelude_h, 残差融合

        loss_mse = F.mse_loss(fused.float(), t.float())

        # CE: 经 coda + lm_head
        hc = fused
        for li in model.coda:
            hc, _ = call_layer(model.layers[li], hc, position_ids=pos)
        logits = model.lm_head(model.norm(hc))
        loss_ce = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
                                  y[:, 1:].reshape(-1))

        loss = cfg.LAMBDA_MSE * loss_mse + cfg.LAMBDA_CE * loss_ce
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step+1}/{cfg.STEPS} loss {loss.item():.4f} "
                  f"(mse {loss_mse.item():.4f} ce {loss_ce.item():.4f})", flush=True)

    # 验证集：teacher forcing 下重建误差
    model.fusion.eval()
    with torch.no_grad():
        sel = torch.arange(nsplit, n)
        h, t, y, pos = batch(sel)
        bo = []
        for br in model.branches:
            hb = h
            for li in br:
                hb, _ = call_layer(model.layers[li], hb, position_ids=pos)
            bo.append(hb)
        fused = model.fusion(h, bo)
        mse = F.mse_loss(fused.float(), t.float()).item()
        cos = F.cosine_similarity(fused.float().reshape(-1, fused.size(-1)),
                                  t.float().reshape(-1, t.size(-1)), dim=-1).mean().item()
    print(f"\n[验证] hidden MSE {mse:.4f}, cosine {cos:.4f}", flush=True)

    os.makedirs(os.path.dirname(cfg.FUSION_CKPT), exist_ok=True)
    model.save_fusion(cfg.FUSION_CKPT)
    print(f"saved {cfg.FUSION_CKPT}", flush=True)
    print(f"fusion params: {model.fusion.extra_repr()}", flush=True)

if __name__ == "__main__":
    main()
