# -*- coding: utf-8 -*-
"""A4: Progressive Branch Growing —— 保底方案。

从原始 1D 模型出发，渐进长出分支：
  Stage 0: 单分支（=原始模型），fusion 恒等
  Stage 1: 加第二分支，fusion 零初始化（能力仍 100%）
  Stage 2: 训练 fusion + router

每个 Stage 起点 = 上一 Stage 的能力，保证不崩（MIDAS/UpLoop 思路的 frozen 版）。
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer
from brick_model import BrickModel
import brick_config as cfg


def stage_forward(model, h, pos, n_active_branches):
    """只跑前 n_active_branches 个分支（渐进增长）。"""
    bo = []
    for br in model.branches[:n_active_branches]:
        hb = h
        for li in br:
            hb, _ = call_layer(model.layers[li], hb, position_ids=pos)
        bo.append(hb)
    # 未激活分支用 h_pre 占位（保持 shape 一致）
    while len(bo) < len(model.branches):
        bo.append(h)
    return model.fusion(h, bo)


def main():
    torch.manual_seed(cfg.SEED)
    dev = "cuda"
    data = torch.load(cfg.TEACHER_DATA, map_location="cpu")
    ph, th = data["prelude_h"], data["teacher_h"]
    nsplit = data["n_train"]

    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL, dtype=torch.bfloat16, device_map=dev).eval()
    model = BrickModel(base).to(dev)
    model.fusion.train()
    for p in model.fusion.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.fusion.parameters(), lr=cfg.LR)

    for stage, nactive in enumerate(cfg.GROW_STAGES):
        print(f"\n=== Stage {stage}: {nactive} active branch(es) ===", flush=True)
        steps = cfg.STEPS // len(cfg.GROW_STAGES)
        for step in range(steps):
            sel = torch.randint(0, nsplit, (cfg.BATCH,))
            h, t = ph[sel].to(dev), th[sel].to(dev)
            pos = torch.arange(h.shape[1], device=dev).unsqueeze(0).expand(h.shape[0], -1)

            fused = stage_forward(model, h, pos, nactive)
            loss = F.mse_loss(fused.float(), t.float())
            opt.zero_grad(); loss.backward(); opt.step()
            if (step+1) % 150 == 0 or step == 0:
                print(f"  stage{stage} step {step+1}/{steps} loss {loss.item():.4f} "
                      f"alpha {model.fusion.alpha.item():.4f}", flush=True)

    out = os.path.join(cfg.WORK, "fusion_progressive.pt")
    os.makedirs(cfg.WORK, exist_ok=True)
    model.save_fusion(out)
    print(f"\nsaved {out}  (alpha={model.fusion.alpha.item():.4f})", flush=True)
    print("诊断：alpha 接近 0 说明分支未提供有用信息 —— 写进报告", flush=True)

if __name__ == "__main__":
    main()
