# -*- coding: utf-8 -*-
"""BRICK PoC 配置：Branch-Rewired Inference with Composable Knowledge.

所有路径与 Amoeba 既有脚本保持一致（远端 /data/work/...）。
"""
import os

# ---------- 路径 ----------
BASE_MODEL   = "/data/work/ckpt_2b_fixed/full_model"   # 冻结的 MiniCPM5-2B (Amoeba 基座)
TOKENIZER    = "/data/models/MiniCPM5-2B"
CORPUS       = "/data/work/chat_corpus.pt"             # Amoeba 自生成 chat 语料
WORK         = "/data/work/brick"
TEACHER_DATA = os.path.join(WORK, "teacher_data.pt")
FUSION_CKPT  = os.path.join(WORK, "fusion.pt")
QUESTIONS    = "/data/work/cap_score_2b.json"          # 79 题（能力评估）
EOS_IDS      = [1, 130073]

# ---------- 2D 拓扑 ----------
N_LAYERS   = 42
PRELUDE    = 16        # 层 0..15       共享前奏
CODA       = 2         # 层 40..41      共享尾部（含 readout 层 41）
# 中间 24 层 (16..39) 切成 2 个异质分支：
#   branch 0: 层 16..30 (15 层)
#   branch 1: 层 31..39 ( 9 层)
# 分支层数不同 -> 参数量不同（"异形积木"）
BRANCH_SIZES = (15, 9)

# ---------- 训练 ----------
BATCH       = 2
STEPS       = 1500
LR          = 3e-5        # 真实 1B 扫描确定：3e-4 会爆炸，1e-4 不稳定，3e-5 稳定下降
FUSION_MODE = "residual_branch_avg"   # 分支均值残差锚点（起点 MSE 5.72 vs h_pre 9.92）
LAMBDA_MSE  = 1.0      # fusion 输出 vs 教师中间表示
LAMBDA_CE   = 0.5      # 经 coda+lm_head 的 next-token CE
BLOCK       = 512
SEQ_LEN     = 256
TRAIN_SPLIT = 0.9
SEED        = 0

# ---------- A1/A2/A5 Router ----------
ROUTER_BOTTLENECK = 256
SEG_LEN           = 16        # A2 段级路由的段长
LAMBDA_ROUTE      = 1.0       # oracle CE 损失权重
LAMBDA_BALANCE    = 0.1       # A5 负载均衡损失权重
ROUTER_STEPS      = 1000

# ---------- A3 Joint-KV ----------
KV_RANK      = 16
KV_DIM       = 128            # 共享 KV 空间维度
LAMBDA_KVALIGN = 1.0

# ---------- A4 Progressive growing ----------
GROW_STAGES  = [1, 2]         # 从 1 分支渐进到 2 分支

# ---------- A6 Depth interpolation ----------
DEPTH_TAU    = 2.0
BETA_VALUES  = [9, 12, 15]    # 多档目标深度

# ---------- 评估 ----------
GEN_TEMP    = 0.8
GEN_TOP_P   = 0.95
GEN_REP_PEN = 1.05
MAX_NEW     = 160
BENCH_TOKS  = 128
BENCH_REPS  = 3

def branches():
    """返回分支层索引列表（已校验无重叠、覆盖连续、最后一层在 coda）。"""
    start = PRELUDE
    out = []
    for n in BRANCH_SIZES:
        out.append(list(range(start, start + n)))
        start += n
    assert start == N_LAYERS - CODA, (start, N_LAYERS - CODA)
    assert CODA >= 1 and (N_LAYERS - 1) in list(range(N_LAYERS - CODA, N_LAYERS)), \
        "readout 层必须在 coda 中（我们已知最后一层独占 ~1.3 nats）"
    return out

if __name__ == "__main__":
    b = branches()
    print("prelude", list(range(PRELUDE)))
    for i, x in enumerate(b):
        print(f"branch{i}", x[0], "..", x[-1], f"({len(x)} layers)")
    print("coda", list(range(N_LAYERS - CODA, N_LAYERS)))
