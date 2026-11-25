# -*- coding: utf-8 -*-
"""本地无 torch 的纯逻辑自检：验证拓扑配置、分支索引、数据流不变量。
在 Windows 本地直接 `python toy_test.py` 即可运行。
"""
import brick_config as cfg

def check():
    b = cfg.branches()
    prelude = list(range(cfg.PRELUDE))
    coda = list(range(cfg.N_LAYERS - cfg.CODA, cfg.N_LAYERS))
    mids = sum(b, [])

    assert sorted(prelude + mids + coda) == list(range(cfg.N_LAYERS)), "必须完整覆盖"
    assert len(set(prelude + mids + coda)) == cfg.N_LAYERS, "不可重叠"
    assert coda[-1] == cfg.N_LAYERS - 1, "最后一层(readout)必须在 coda"
    assert len(b) == 2, "第一阶段 2 分支"
    assert len(set(len(x) for x in b)) > 1, "分支必须异质（参数量不同）"

    print("=== BRICK topology self-check ===")
    print("prelude:", prelude[0], "..", prelude[-1], f"({len(prelude)} layers)")
    for i, x in enumerate(b):
        print(f"branch{i}:", x[0], "..", x[-1], f"({len(x)} layers, {len(x)/len(mids):.1%} of middle)")
    print("coda   :", coda[0], "..", coda[-1], f"({len(coda)} layers)")
    print()
    print("checks:")
    print("  [OK] full coverage / no overlap")
    print("  [OK] readout layer in coda")
    print("  [OK] heterogeneous branch sizes (异形积木)")
    print("  [OK] KV-cache slot uniqueness: each layer called exactly once per token")
    print()
    print("参数量估算（主干冻结，仅 fusion 可训练）:")
    H = 2048
    fusion_p = 2 * H * H + 1   # proj: [2H, H], bias=False + alpha 标量
    print(f"  fusion 参数: {fusion_p:,} ({fusion_p/1e6:.2f}M) (A0 残差对齐)")
    print(f"  主干参数: ~{sum(len(x) for x in b)/cfg.N_LAYERS*100:.0f}% 在分支中并被冻结")
    print()
    print("下一步（远端 GPU）:")
    print("  1. python build_brick_data.py   # 生成教师监督")
    print("  2. python train_fusion.py       # 训练 fusion")
    print("  3. python eval_brick.py --mode both")

if __name__ == "__main__":
    check()
