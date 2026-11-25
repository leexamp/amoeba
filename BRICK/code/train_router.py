# -*- coding: utf-8 -*-
"""A1+A2+A5: 训练 router（oracle 监督 + 可选段级 + 可选负载均衡）。

与 train_fusion.py 分开：先训好 fusion（A0），再训 router。
router 用教师的表示当 oracle 标签，不用模型自置信度（v2 教训）。
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from layer_utils import call_layer
from brick_model import BrickModel
from brick_router import BrickRouter, SegmentRouter, SpecializedRouter, oracle_labels, oracle_accuracy, ce_oracle, ablation_oracle
from brick_router_gumbel import GumbelRouter, BalancedGumbelRouter
from layer_utils import precompute_rope
import brick_config as cfg


def build_router(kind, hidden, k, seg_len=cfg.SEG_LEN):
    if kind == "token":
        return BrickRouter(hidden, k, cfg.ROUTER_BOTTLENECK)
    if kind == "segment":
        return SegmentRouter(hidden, k, seg_len, cfg.ROUTER_BOTTLENECK)
    if kind == "specialized":
        return SpecializedRouter(hidden, k, cfg.ROUTER_BOTTLENECK)
    raise ValueError(kind)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["token", "segment", "specialized"], default="token")
    ap.add_argument("--oracle", choices=["repr", "ce", "ablation"], default="ablation",
                    help="repr=表示距离(证伪), ce=CE-based, ablation=跳过代价(默认,需先训fusion)")
    ap.add_argument("--router", choices=["plain", "gumbel", "gumbel_balanced"], default="gumbel",
                    help="plain=softmax(已证伪会塌缩), gumbel=Gumbel-Softmax(默认,已验证均衡)")
    a = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    dev = "cuda"
    data = torch.load(cfg.TEACHER_DATA, map_location="cpu")
    ph, th = data["prelude_h"], data["teacher_h"]
    n, nsplit = ph.shape[0], data["n_train"]

    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL, dtype=torch.bfloat16, device_map=dev).eval()
    model = BrickModel(base).to(dev)
    if os.path.exists(cfg.FUSION_CKPT):
        model.load_fusion(cfg.FUSION_CKPT)
        print("loaded A0 fusion", flush=True)
    model.eval()

    if a.router == "gumbel":
        router = GumbelRouter(base.config.hidden_size, len(model.branches),
                              tau_start=1.0, tau_end=0.1).to(dev)
    elif a.router == "gumbel_balanced":
        router = BalancedGumbelRouter(base.config.hidden_size, len(model.branches),
                                      tau_start=1.0, tau_end=0.1,
                                      balance_weight=0.5).to(dev)
    else:
        router = build_router(a.kind, base.config.hidden_size, len(model.branches)).to(dev)
    is_gumbel = isinstance(router, (GumbelRouter, BalancedGumbelRouter))
    opt = torch.optim.AdamW(router.parameters(), lr=cfg.LR)

    print(f"[训练 router: {a.kind}] {sum(p.numel() for p in router.parameters()):,} params", flush=True)

    def branch_outputs(h, pos):
        rope = precompute_rope(base, pos)
        bo = []
        for br in model.branches:
            hb = h
            for li in br:
                hb, _ = call_layer(model.layers[li], hb, position_ids=pos,
                                   position_embeddings=rope)
            bo.append(hb)
        return bo

    for step in range(cfg.ROUTER_STEPS):
        sel = torch.randint(0, nsplit, (cfg.BATCH,))
        h = ph[sel].to(dev)                      # [B,T,H] prelude 输出
        t = th[sel].to(dev)                      # 教师 hidden
        pos = torch.arange(h.shape[1], device=dev).unsqueeze(0).expand(h.shape[0], -1)

        with torch.no_grad():
            bo = branch_outputs(h, pos)
            rope = precompute_rope(base, pos)
            if a.oracle == "ablation":
                # ★ 跳过代价 oracle：需先训练 fusion（train_fusion.py）
                label, _ = ablation_oracle(base, model.layers, h, model.branches,
                                           data["tokens"][sel].to(dev), pos, rope,
                                           model.coda, model.fusion)
            elif a.oracle == "ce":
                label = ce_oracle(base, model.layers, bo, data["tokens"][sel].to(dev),
                                  pos, rope, model.coda)
            else:
                label = oracle_labels(bo, t)     # 已证伪：标签 100% 偏深分支

        if is_gumbel:
            soft, logits = router(h, hard=True, total_steps=cfg.ROUTER_STEPS)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), label.reshape(-1))
            if a.router == "gumbel_balanced":
                loss = loss + cfg.LAMBDA_BALANCE * router.balance_loss(soft)
            router.step += 1
        else:
            logits = router(h)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), label.reshape(-1))
            if a.kind == "specialized":
                loss = loss + cfg.LAMBDA_BALANCE * router.balance_loss(h)

        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 200 == 0 or step == 0:
            if is_gumbel:
                choose = router.choose(h, total_steps=cfg.ROUTER_STEPS)
            else:
                choose = router(h).argmax(-1)
            agree = oracle_accuracy(choose, label)
            print(f"  step {step+1}/{cfg.ROUTER_STEPS} loss {loss.item():.4f} "
                  f"oracle-agree {agree:.3f}", flush=True)

    out = os.path.join(cfg.WORK, f"router_{a.kind}_{a.router}.pt")
    os.makedirs(cfg.WORK, exist_ok=True)
    torch.save({"router": router.state_dict(), "kind": a.kind}, out)
    print(f"saved {out}", flush=True)

    # 诊断：路由分布（是否塌缩）
    sel = torch.arange(nsplit, n)
    with torch.no_grad():
        h, t = ph[sel].to(dev), th[sel].to(dev)
        pos = torch.arange(h.shape[1], device=dev).unsqueeze(0).expand(h.shape[0], -1)
        bo = branch_outputs(h, pos)
        rope = precompute_rope(base, pos)
        if a.oracle == "ablation":
            label, _ = ablation_oracle(base, model.layers, h, model.branches,
                                       data["tokens"][sel].to(dev), pos, rope,
                                       model.coda, model.fusion)
        elif a.oracle == "ce":
            label = ce_oracle(base, model.layers, bo, data["tokens"][sel].to(dev),
                              pos, rope, model.coda)
        else:
            label = oracle_labels(bo, t)
        choose = router.choose(h, total_steps=cfg.ROUTER_STEPS) if is_gumbel else router(h).argmax(-1)
        agree = oracle_accuracy(choose, label)
        hist = torch.bincount(choose.flatten(), minlength=len(model.branches)).float()
        hist = (hist / hist.sum()).tolist()
    print(f"\n[诊断] oracle 一致率 {agree:.3f}, 分支选择分布 {hist}", flush=True)
    if max(hist) > 0.95:
        print("  WARNING: 路由塌缩到单分支 —— 动态性失败，写进报告", flush=True)

if __name__ == "__main__":
    main()
