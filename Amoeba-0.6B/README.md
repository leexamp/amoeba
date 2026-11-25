> **Updated (Amoeba-DA).** This directory documents the original token-level early-exit research line, which was investigated and **rejected**. The shipped technique is **Amoeba-DA** (parallel draft heads + full-model verification): 2B 2.02×, 1B 1.30×, statistically lossless (not bit-exact). See the root `README.md` / `README_zh.md` and `docs/动态激活_完整实验矩阵.md`.

---
license: apache-2.0
base_model:
  - Qwen/Qwen3-0.6B
library_name: transformers
pipeline_tag: text-generation
tags:
  - dynamic-activation
  - early-exit
  - halt-head
  - qwen3
  - training-in-progress
---

# Amoeba-0.6B

## 中文

**一句话简介**：Amoeba-0.6B 是 `Qwen/Qwen3-0.6B` 的独立研究衍生模型，计划在 Qwen3 架构（28 层、含 q_norm/k_norm）上挂载 token 级 halt head 实现动态深度推理，用于验证 Amoeba 方法跨模型族的普适性。**当前状态：训练中 / 待发布——GPU 上仅有 step-3 冒烟 checkpoint，无可用正式权重。**

### 1. 基座模型（Base model）

| 项目 | 值 |
|---|---|
| 名称 | `Qwen/Qwen3-0.6B` |
| 架构 | `Qwen3ForCausalLM`（`model_type: qwen3`），transformers ≥4.51 原生支持，**不需要** `trust_remote_code`；注意力内含 Qwen3 特有的 **q_norm / k_norm**（per-head RMSNorm, dim=128），wrapper 逐层调用时自动执行，不可绕过或修改 modeling 代码 |
| 层数 | 28 |
| hidden_size / intermediate_size | 1024 / 3072 |
| 注意力 | GQA 16 个 Q 头 / **8 个 KV 头**，`head_dim=128`，RoPE `rope_theta=1e6` |
| 词表 | 151936；`tie_word_embeddings=true`（lm_head 与 embed_tokens 共享）；`eos=151645` `<\|im_end\|>`，`pad=151643` |
| 上下文 | 40960（≈40K；以 config 为准，tokenizer_config 的 131072 是假象） |
| 精度 | bfloat16，参数量约 0.60B |
| 基座 License | **Apache-2.0**（Qwen 官方 LICENSE） |

### 2. Amoeba 改造说明（设计，尚未产出正式权重）

- **Halt head 结构**：每层输出挂 `nn.Linear(hidden_size + 1, 1)`；0.6B 为 **`Linear(1025, 1) × 28`**（新增参数约 2.9 万）。输入为该层隐状态 `h_i` 与 token 级不确定性 `unc=tanh(-log(p_max))` 的拼接。冒烟 checkpoint 已实测核实：`ckpt_06b_smoke/halt_heads.pt` 的 cfg 为 `n_layer=28, hidden_size=1024, vocab_size=151936, eos=[151645], pad=151643`，权重形状 `(1, 1025)`。
- **实现方式**：用 `AutoModelForCausalLM.from_pretrained(...)` 加载（解析为 Qwen3ForCausalLM），沿用 `AmoebaModel` wrapper 逐层 forward，不修改 modeling 代码；中间层 logits 仍由 `norm + lm_head` 计算。必须使用 SDPA 并校验注意力类名含 `Sdpa`（wrapper 依赖自动因果掩码）。
- **动态激活**：逐 token 累计 `cum = 1-∏(1-p_l)`，`cum ≥ threshold` 时经 `norm + lm_head` 提前输出；因 `tie_word_embeddings=true`，保存 halt-only checkpoint 时**只存 halt heads，不要存 tied lm_head**。
- **特殊适配**：脚本里硬编码的 `eos_id=1` 必须改为 **151645**（否则生成永不停止）；语料需用 Qwen2Tokenizer 重新 tokenize（Qwen ChatML 模板，注意 `<think>`/`enable_thinking`），断言 `max(token_id) < 151936`；8 个 KV 头使生成 KV cache 更大，评测长上下文时注意显存。

### 3. 用法示例（发布后接口示意）

> 当前只有冒烟 checkpoint（step 3，随机初始化附近），**不可用于真实推理**。以下为权重发布后的加载方式示意。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer_qwen06b import AmoebaModel  # Qwen3 分支 wrapper（待发布）

BASE = "Qwen/Qwen3-0.6B"        # 发布后替换为 Hub 仓库
HALT = "./halt_heads.pt"        # 正式版 halt 权重（待发布）

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("你好，简单介绍一下你自己。", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
# 注意：eos_id=151645、pad_id=151643；生成时不要沿用 1B/2B 的 eos=1
```

### 4. 训练状态（诚实说明）

- **训练中 / 待发布**：2026-11-22 GPU 核实——`/data/work/ckpt_qwen06b_full/` 与 `/data/work/ckpt_06b_fixed/` **均不存在**；只有 `ckpt_06b_smoke/halt_heads.pt`（step 3，仅用于验证维度与梯度的冒烟测试），以及早期空目录 `ckpt_qwen06b_full_BUGGY_2354/`。`logs_06b_full.log` 显示 full 训练曾跑到约 step 560，但没有产出保存的权重（无 `[save]` 记录）。**目前没有可下载/可推理的正式 checkpoint。**
- **已知风险**：Qwen3 小基座上均匀先验与语义项更敏感（0.3B 脚本已将 γ 从 2.0 降到 1.0、η 降到 0.05）；2B/1B 上观察到的退出坍缩（浅层早退导致 PPL 暴涨）同样需要通过 `lam=0`、质量门控、冻结教师重训 halt 等方式修复。0.6B 修复训练完成后需通过"强制退出 CE 曲线 + PPL + 退出层分布 + 贪心生成"四件套验收才发布。
- **训练数据（计划/部分已用）**：CCI3-Data（Apache-2.0）等，与主线一致。

### 5. License

- 基座 `Qwen/Qwen3-0.6B`：**Apache-2.0**；衍生权重发布后整体采用 **Apache-2.0**（附 LICENSE + NOTICE）。
- Amoeba 新增部分（halt head、wrapper、训练/评测脚本、本 card）同样为 **Apache-2.0**。
- 商标/背书：独立研究项目，**与阿里云 / Qwen 团队无隶属或官方背书关系**；模型名不含 "official"。
- 数据署名：训练数据含 `nvidia/OpenMathReasoning`（CC-BY-4.0，见致谢）；未复制 LayerSkip（CC-BY-NC）代码。

### 6. Citation / 致谢

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-0.6B}
}
```

- 基座：Qwen Team, *Qwen3-0.6B* (Apache-2.0).
- 方法思想：Schuster et al., *CALM*; Elhoushi et al., *LayerSkip*（仅论文思想，未用其代码）。
- 数据：BAAI/CCI3-Data (Apache-2.0)、nvidia/OpenMathReasoning (**CC-BY-4.0**，特此署名)。

---

## English

**One-line summary**: Amoeba-0.6B is an independent research derivative of `Qwen/Qwen3-0.6B`, planning to add token-level halt heads to the Qwen3 architecture (28 layers, with q_norm/k_norm) for adaptive-depth inference, testing cross-family generality of the Amoeba method. **Status: training in progress / unreleased — only a step-3 smoke checkpoint exists on GPU; no usable weights.**

### 1. Base model

| Field | Value |
|---|---|
| Name | `Qwen/Qwen3-0.6B` |
| Architecture | `Qwen3ForCausalLM` (`model_type: qwen3`), natively supported by transformers ≥4.51; **no** `trust_remote_code`. Attention includes Qwen3-specific **q_norm / k_norm** (per-head RMSNorm, dim=128), executed automatically by the layer-by-layer wrapper — never bypass or edit modeling code |
| Layers | 28 |
| hidden_size / intermediate_size | 1024 / 3072 |
| Attention | GQA, 16 Q / **8 KV** heads, `head_dim=128`, RoPE `rope_theta=1e6` |
| Vocabulary | 151936; `tie_word_embeddings=true` (lm_head tied to embed_tokens); `eos=151645` `<\|im_end\|>`, `pad=151643` |
| Context | 40960 (~40K; config is authoritative, tokenizer_config's 131072 is not) |
| Precision | bfloat16, ~0.60B parameters |
| Base license | **Apache-2.0** (official Qwen LICENSE) |

### 2. Amoeba modification (design; no official weights yet)

- **Halt head**: one `nn.Linear(hidden_size + 1, 1)` per layer — for 0.6B, **`Linear(1025, 1) × 28`** (~29k new parameters). Input = layer hidden state `h_i` plus token-level uncertainty `unc=tanh(-log(p_max))`. Verified from the smoke checkpoint (`ckpt_06b_smoke/halt_heads.pt`): cfg `n_layer=28, hidden_size=1024, vocab_size=151936, eos=[151645], pad=151643`, weight shape `(1, 1025)`.
- **Implementation**: load via `AutoModelForCausalLM.from_pretrained(...)` (resolves to Qwen3ForCausalLM), keep the `AmoebaModel` wrapper's layer-by-layer forward, no modeling-code edits; intermediate logits still via `norm + lm_head`. SDPA is required and the attention class name must contain `Sdpa` (the wrapper relies on automatic causal masking).
- **Dynamic activation**: per-token `cum = 1-∏(1-p_l)`; exit at the first layer where `cum ≥ threshold` via `norm + lm_head`. Because `tie_word_embeddings=true`, halt-only checkpoints must store **halt heads only — never the tied lm_head**.
- **Adaptation notes**: hardcoded `eos_id=1` must become **151645** (otherwise generation never stops); retokenize corpora with Qwen2Tokenizer (Qwen ChatML template, mind `<think>`/`enable_thinking`) and assert `max(token_id) < 151936`; 8 KV heads enlarge generation KV cache — watch memory on long-context evaluation.

### 3. Usage (interface sketch for a future release)

> Only a step-3 smoke checkpoint (near-random initialization) exists — **not usable for real inference**. The snippet illustrates the post-release loading flow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer_qwen06b import AmoebaModel  # Qwen3-branch wrapper (to be released)

BASE = "Qwen/Qwen3-0.6B"        # replace with released Hub repo
HALT = "./halt_heads.pt"        # official halt weights (unreleased)

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("Hello, tell me about yourself.", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
# eos_id=151645, pad_id=151643; do not reuse eos=1 from 1B/2B
```

### 4. Training status (honest disclosure)

- **IN PROGRESS / UNRELEASED**: GPU verification on 2026-11-22 — `/data/work/ckpt_qwen06b_full/` and `/data/work/ckpt_06b_fixed/` **do not exist**; only `ckpt_06b_smoke/halt_heads.pt` (step 3, dimension/gradient smoke test) and an earlier empty dir `ckpt_qwen06b_full_BUGGY_2354/`. `logs_06b_full.log` shows a full run reached ~step 560 but produced no saved weights (no `[save]` record). **No downloadable/inferable official checkpoint exists.**
- **Known risks**: small Qwen3 bases are sensitive to the uniformity prior and semantic term (the 0.3B script already lowered γ 2.0→1.0, η→0.05). Exit collapse observed on 2B/1B (shallow exits, PPL explosion) must also be fixed here via `lam=0`, quality gating, frozen-teacher halt retraining. Release requires passing the four-part check: forced-exit CE curve + PPL + exit-layer distribution + greedy generation.
- **Training data (planned/partially used)**: CCI3-Data (Apache-2.0) etc., consistent with the mainline.

### 5. License

- Base `Qwen/Qwen3-0.6B` is **Apache-2.0**; derivative weights, once released, will be **Apache-2.0** (LICENSE + NOTICE).
- All Amoeba additions (halt heads, wrapper, training/eval scripts, this card) are **Apache-2.0**.
- Trademark: independent research, **not affiliated with or endorsed by Alibaba/Qwen**; no "official" in the model name.
- Data attribution: training data includes `nvidia/OpenMathReasoning` (**CC-BY-4.0**, credited); no LayerSkip (CC-BY-NC) code copied.

### 6. Citation / acknowledgements

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-0.6B}
}
```

- Base: Qwen Team, *Qwen3-0.6B* (Apache-2.0).
- Method inspiration: Schuster et al., *CALM*; Elhoushi et al., *LayerSkip* (ideas only, no code).
- Data: BAAI/CCI3-Data (Apache-2.0), nvidia/OpenMathReasoning (**CC-BY-4.0**, credited).
