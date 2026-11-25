#!/bin/bash
# BRICK 完整 PoC 流程（远端 GPU）。按顺序跑，每步结果都保存。
set -e
cd /data/work/brick/code

echo "========== 0. 本地自检（无需 GPU）=========="
python toy_test.py

echo "========== 1. 生成教师监督数据 =========="
python build_brick_data.py

echo "========== 2. A0: 残差对齐 fusion =========="
python train_fusion.py

echo "========== 3. A4: 渐进增长（保底对照）=========="
python train_progressive.py

echo "========== 4. A1: Oracle router（CE oracle + Gumbel，已验证防塌缩）=========="
python train_router.py --kind token --oracle ce --router gumbel
python train_router.py --kind token --oracle ce --router gumbel_balanced

echo "========== 5. 评估 =========="
python eval_brick.py --mode both

echo "========== DONE =========="
# 注意：A3 (Joint-KV) 和 A6 (Depth interpolation) 的接口代码已就绪
# (brick_kv.py / brick_depth.py)，但端到端集成需要 A1 router 结果确认后再接。
# 见 ALGORITHMS.md 路线图。
