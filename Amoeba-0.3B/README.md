> **Updated (Amoeba-DA).** This directory documents the original token-level early-exit research line, which was investigated and **rejected**. The shipped technique is **Amoeba-DA** (parallel draft heads + full-model verification): 2B 2.02×, 1B 1.30×, statistically lossless (not bit-exact). See the root `README.md` / `README_zh.md` and `docs/动态激活_完整实验矩阵.md`.

---
license: apache-2.0
base_model:
  - taobao-mnn/MNN-Pocket-0.3B
library_name: transformers
pipeline_tag: text-generation
tags:
  - dynamic-activation
  - early-exit
  - halt-head
  - qwen3
  - training-in-progress
---

# Amoeba-0.3B

## 中文

**一句话简介**：Amoeba-0.3B 是 `taobao-mnn/MNN-Pocket-0.3B` 的独立研究衍生模型，计划在小词表 Qwen3 架构（28 层、hidden 896）上挂载 token 级 halt head，验证动态激活方法在端侧极小基座上的收益与 scaling 规律。**当前状态：训练中 / 待发布——GPU 上仅有 step-3 冒烟 checkpoint，无可用正式权重。**

### 1. 基座模型（Base model）

| 项目 | 值 |
|---|---|
| 名称 | `taobao-mnn/MNN-Pocket-0.3B`（只使用主仓库 BF16 权重；`-MNN` / `-GGUF` 是部署产物，不可用于训练） |
| 架构 | `Qwen3ForCausalLM`（`model_type: qwen3`），transformers ≥4.51 原生支持，**不需要** `trust_remote_code`；attention 含 q_norm/k_norm（per-head RMSNorm, dim=64）；config 用 `dtype: bfloat16` 字段 |
| 层数 | 28（`layer_types` 全部为 `full_attention`） |
| hidden_size / intermediate_size | 896 / 2560 |
| 注意力 | GQA 14 个 Q 头 / 2 个 KV 头，`head_dim=64`，RoPE `rope_theta=1e6` |
| 词表 | **65536**（三者中最小）；`tie_word_embeddings=true`；`eos=65512` `<\|im_end\|>`，`pad=65510`，`bos=null` |
| 上下文 | 40960（官方称部署支持 32K） |
| 精度与参数 | bfloat16，约 300M 参数；config 中 `use_cache=false`，**推理前必须显式设为 True** |
| 基座 License | **Apache-2.0**（官方 LICENSE.txt；注意：不是 MIT） |

### 2. Amoeba 改造说明（设计，尚未产出正式权重）

- **Halt head 结构**：每层输出挂 `nn.Linear(hidden_size + 1, 1)`；0.3B 为 **`Linear(897, 1) × 28`**（新增参数约 2.5 万）。输入为该层隐状态 `h_i` 与 token 级不确定性 `unc=tanh(-log(p_max))` 的拼接。冒烟 checkpoint 已实测核实：`ckpt_03b_smoke/halt_heads.pt` 的 cfg 为 `n_layer=28, hidden_size=896, vocab_size=65536, eos=[65512], pad=65510`，权重形状 `(1, 897)`。
- **实现方式**：`AutoModelForCausalLM.from_pretrained(...)` 加载（解析为 Qwen3ForCausalLM），复用 0.6B 分支的 `AmoebaModel` wrapper 逐层 forward，不修改 modeling 代码；中间层 logits 由 `norm + lm_head` 计算。必须使用 SDPA 并校验注意力类名含 `Sdpa`。因 `tie_word_embeddings=true`，halt-only checkpoint 只存 halt heads。
- **动态激活**：逐 token 累计 `cum = 1-∏(1-p_l)`，`cum ≥ threshold` 时提前经 `norm + lm_head` 输出；小容量基座更容易把退出分布学成"全退浅层"或"全退末层"，训练时已把均匀先验 γ 从 2.0 降到 1.0、语义项 η 从 0.1 降到 0.05（脚本已核实），不可照搬 2B 超参。
- **特殊适配**：硬编码 `eos_id=1` 必须改为 **65512**（停止条件、ckpt 元信息同步改）；`pad_id=65510`；外部语料必须用本基座 tokenizer 重新 tokenize 并断言 `max(id) < 65536`（越界会直接崩溃）；词表小使 CE 显存很低，可把 `--ce-samples` 提到 64~128 提升决策质量；评测/`generate` 前设 `model.config.use_cache = True`。

### 3. 用法示例（发布后接口示意）

> 当前只有 step-3 冒烟 checkpoint（lr=0，仅验证维度与梯度），**不可用于真实推理**。以下为权重发布后的加载方式示意。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer_qwen03b import AmoebaModel  # Qwen3 分支 wrapper（待发布）

BASE = "taobao-mnn/MNN-Pocket-0.3B"   # 发布后替换为 Hub 仓库
HALT = "./halt_heads.pt"              # 正式版 halt 权重（待发布）

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
base.config.use_cache = True          # config 默认 false，生成前必须打开
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("你好，用一句话介绍你自己。", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
# eos_id=65512、pad_id=65510；token id 必须 < 65536
```

### 4. 训练状态（诚实说明）

- **训练中 / 待发布**：2026-11-22 GPU 核实——`/data/work/ckpt_qwen03b_full/` 与 `/data/work/ckpt_03b_fixed/` **均不存在**；只有 `ckpt_03b_smoke/halt_heads.pt`（step 3，lr=0，仅冒烟测试）和早期空目录 `ckpt_qwen03b_full_BUGGY_2354/`。`logs_03b_full.log` 显示 full 训练曾跑到约 step 460，但没有保存权重（无 `[save]` 记录）。**目前没有可下载/可推理的正式 checkpoint。**
- **已知风险**：基座较新（2026-11 发布），小词表 + 小容量下 halt 决策极不稳定；日志中逐层 CE 在 L4 后几乎平坦（≈29~31），属于典型的"浅层与深层无质量差异"——若直接早退会重现 2B/1B 的退出坍缩。需先按修复方案（`lam=0`、质量门控、冻结基座重训 halt）处理，并以"强制退出 CE 曲线 + PPL + 退出层分布 + 贪心生成"四件套验收后才发布。
- **训练数据（计划/部分已用）**：CCI3-Data（Apache-2.0）等，必须用 65536 词表 tokenizer 重新 tokenize。

### 5. License

- 基座 `taobao-mnn/MNN-Pocket-0.3B`：**Apache-2.0**（官方 LICENSE.txt，**不是 MIT**）；衍生权重发布后整体采用 **Apache-2.0**（附 LICENSE + NOTICE，保留上游版权声明）。
- Amoeba 新增部分（halt head、wrapper、训练/评测脚本、本 card）同样为 **Apache-2.0**。
- 商标/背书：独立研究项目，**与淘宝 / MNN 团队无隶属或官方背书关系**；模型名不含 "official"。
- 数据署名：训练数据含 `nvidia/OpenMathReasoning`（CC-BY-4.0，见致谢）；未复制 LayerSkip（CC-BY-NC）代码。

### 6. Citation / 致谢

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-0.3B}
}
```

- 基座：Taobao-MNN Team, *MNN-Pocket-0.3B* (Apache-2.0).
- 方法思想：Schuster et al., *CALM*; Elhoushi et al., *LayerSkip*（仅论文思想，未用其代码）。
- 数据：BAAI/CCI3-Data (Apache-2.0)、nvidia/OpenMathReasoning (**CC-BY-4.0**，特此署名)。

---

## English

**One-line summary**: Amoeba-0.3B is an independent research derivative of `taobao-mnn/MNN-Pocket-0.3B`, planning to add token-level halt heads to a small-vocabulary Qwen3 base (28 layers, hidden 896) to study dynamic-activation gains and scaling at the tiniest on-device scale. **Status: training in progress / unreleased — only a step-3 smoke checkpoint exists on GPU; no usable weights.**

### 1. Base model

| Field | Value |
|---|---|
| Name | `taobao-mnn/MNN-Pocket-0.3B` (main repo BF16 weights only; `-MNN` / `-GGUF` are deployment artifacts and unusable for training) |
| Architecture | `Qwen3ForCausalLM` (`model_type: qwen3`), natively supported by transformers ≥4.51; **no** `trust_remote_code`. Attention includes q_norm/k_norm (per-head RMSNorm, dim=64); config uses the `dtype: bfloat16` field |
| Layers | 28 (all `layer_types` are `full_attention`) |
| hidden_size / intermediate_size | 896 / 2560 |
| Attention | GQA, 14 Q / 2 KV heads, `head_dim=64`, RoPE `rope_theta=1e6` |
| Vocabulary | **65536** (smallest in the family); `tie_word_embeddings=true`; `eos=65512` `<\|im_end\|>`, `pad=65510`, `bos=null` |
| Context | 40960 (official docs say 32K deployment support) |
| Precision & size | bfloat16, ~300M params; config has `use_cache=false` — **set it to True explicitly before inference** |
| Base license | **Apache-2.0** (official LICENSE.txt; **not MIT**) |

### 2. Amoeba modification (design; no official weights yet)

- **Halt head**: one `nn.Linear(hidden_size + 1, 1)` per layer — for 0.3B, **`Linear(897, 1) × 28`** (~25k new parameters). Input = layer hidden state `h_i` plus token-level uncertainty `unc=tanh(-log(p_max))`. Verified from the smoke checkpoint (`ckpt_03b_smoke/halt_heads.pt`): cfg `n_layer=28, hidden_size=896, vocab_size=65536, eos=[65512], pad=65510`, weight shape `(1, 897)`.
- **Implementation**: load via `AutoModelForCausalLM.from_pretrained(...)` (resolves to Qwen3ForCausalLM), reuse the 0.6B-branch `AmoebaModel` wrapper's layer-by-layer forward, no modeling-code edits; intermediate logits via `norm + lm_head`. SDPA required; attention class must contain `Sdpa`. With `tie_word_embeddings=true`, halt-only checkpoints store halt heads only.
- **Dynamic activation**: per-token `cum = 1-∏(1-p_l)`; exit at the first layer where `cum ≥ threshold`. A tiny base tends to learn "exit all at shallow layers" or "exit all at the last layer"; the script already lowers the uniformity weight γ 2.0→1.0 and semantic weight η 0.1→0.05 (verified) — do not copy 2B hyperparameters.
- **Adaptation notes**: hardcoded `eos_id=1` must become **65512** (also fix stopping condition and ckpt metadata); `pad_id=65510`; retokenize all corpora with this base's tokenizer and assert `max(id) < 65536` (out-of-range ids crash training); the small vocabulary lowers CE memory, so `--ce-samples` can be raised to 64~128; set `model.config.use_cache = True` before evaluation/generation.

### 3. Usage (interface sketch for a future release)

> Only a step-3 smoke checkpoint (lr=0, dimension/gradient check) exists — **not usable for real inference**. The snippet illustrates the post-release loading flow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer_qwen03b import AmoebaModel  # Qwen3-branch wrapper (to be released)

BASE = "taobao-mnn/MNN-Pocket-0.3B"   # replace with released Hub repo
HALT = "./halt_heads.pt"              # official halt weights (unreleased)

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
base.config.use_cache = True          # config defaults to false; enable for generation
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("Hello, introduce yourself in one sentence.", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
# eos_id=65512, pad_id=65510; token ids must stay < 65536
```

### 4. Training status (honest disclosure)

- **IN PROGRESS / UNRELEASED**: GPU verification on 2026-11-22 — `/data/work/ckpt_qwen03b_full/` and `/data/work/ckpt_03b_fixed/` **do not exist**; only `ckpt_03b_smoke/halt_heads.pt` (step 3, lr=0, smoke test) and an earlier empty dir `ckpt_qwen03b_full_BUGGY_2354/`. `logs_03b_full.log` shows a full run reached ~step 460 but produced no saved weights (no `[save]` record). **No downloadable/inferable official checkpoint exists.**
- **Known risks**: the base is new (released 2026-11) and the small vocabulary/capacity makes halt decisions unstable; logs show per-layer CE nearly flat after L4 (~29–31) — the classic "no quality difference between shallow and deep layers" signature, which would reproduce the 2B/1B exit collapse if early exits were used. Fix first (`lam=0`, quality gating, halt retraining on a frozen base), then release only after the four-part check: forced-exit CE curve + PPL + exit-layer distribution + greedy generation.
- **Training data (planned/partially used)**: CCI3-Data (Apache-2.0) etc.; all corpora must be retokenized to the 65536 vocabulary.

### 5. License

- Base `taobao-mnn/MNN-Pocket-0.3B` is **Apache-2.0** (official LICENSE.txt; **not MIT**); derivative weights, once released, will be **Apache-2.0** (LICENSE + NOTICE, upstream notices preserved).
- All Amoeba additions (halt heads, wrapper, training/eval scripts, this card) are **Apache-2.0**.
- Trademark: independent research, **not affiliated with or endorsed by Taobao/MNN**; no "official" in the model name.
- Data attribution: training data includes `nvidia/OpenMathReasoning` (**CC-BY-4.0**, credited); no LayerSkip (CC-BY-NC) code copied.

### 6. Citation / acknowledgements

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-0.3B}
}
```

- Base: Taobao-MNN Team, *MNN-Pocket-0.3B* (Apache-2.0).
- Method inspiration: Schuster et al., *CALM*; Elhoushi et al., *LayerSkip* (ideas only, no code).
- Data: BAAI/CCI3-Data (Apache-2.0), nvidia/OpenMathReasoning (**CC-BY-4.0**, credited).
