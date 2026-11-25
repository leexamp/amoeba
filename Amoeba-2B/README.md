> **Updated (Amoeba-DA).** This directory documents the original token-level early-exit research line, which was investigated and **rejected**. The shipped technique is **Amoeba-DA** (parallel draft heads + full-model verification): 2B 2.02×, 1B 1.30×, statistically lossless (not bit-exact). See the root `README.md` / `README_zh.md` and `docs/动态激活_完整实验矩阵.md`.

---
license: apache-2.0
base_model:
  - openbmb/MiniCPM5-2B
library_name: transformers
pipeline_tag: text-generation
tags:
  - dynamic-activation
  - early-exit
  - halt-head
  - on-device
  - research-preview
---

# Amoeba-2B

## 中文

**一句话简介**：Amoeba-2B 是 `openbmb/MiniCPM5-2B` 的独立研究衍生模型，在每一层 Transformer 后挂一个极轻量的 token 级 halt head，让每个 token 自适应决定经过多少层（动态深度 / early exit），目标是在少量质量代价下换取端侧推理加速。**当前状态：训练中 / 待发布（详见第 5 节）。**

### 1. 基座模型（Base model）

| 项目 | 值 |
|---|---|
| 名称 | `openbmb/MiniCPM5-2B` |
| 架构 | `LlamaForCausalLM`（`model_type: llama`），标准 Transformer Decoder-only，无需 `trust_remote_code` |
| 层数 | 42 |
| hidden_size / intermediate_size | 2048 / 6144 |
| 注意力 | GQA 16 个 Q 头 / 2 个 KV 头，`head_dim=128`，RoPE `rope_theta=5e6` |
| 词表 | 130560（BPE，ChatML 风格；`eos=[1, 130073]`，`pad=1`） |
| 上下文 | 131072（128K） |
| 精度 | bfloat16，`tie_word_embeddings=false` |
| 基座 License | **Apache-2.0**（官方 README 明确） |

### 2. Amoeba 改造说明

- **Halt head 结构**：在基座 `model.layers[i]` 的每层输出后挂一个 `nn.Linear(hidden_size + 1, 1)` 二分类门控；2B 为 `Linear(2049, 1) × 42`，总新增参数约 8.6 万（相对 2.6B 基座可忽略）。输入 = 该层隐状态 `h_i` 拼接 token 级不确定性标量 `unc = tanh(-log(p_max))`。`+1` 维与权重形状已由 GPU checkpoint 实测核实（`halt_heads.pt` 中 `0.weight` 形状为 `(1, 2049)`，cfg 为 `n_layer=42, hidden_size=2048, vocab_size=130560`）。
- **实现方式**：以 wrapper（`AmoebaModel`）逐层 forward，**不修改 `modeling_llama.py` 内部循环**，保留基座全部权重、tokenizer、KV cache 与 `generate` 兼容性。中间层输出经 `norm + lm_head` 得到 logits；注意力必须使用 SDPA（eager 下需显式传 4D 因果 mask）。
- **动态激活（推理）**：每个 token 独立维护累计退出概率 `cum = 1 - ∏(1 - p_l)`，`cum` 首次超过阈值 `threshold` 即在该层经 `norm + lm_head` 输出并停止后续层计算；阈值越大越保守（越接近全层 42 层）。训练目标为逐层 CE + anchor + 均匀先验 + 语义一致性 + 熵正则（LayerSkip 思想自研实现，**未复制 LayerSkip 的 CC-BY-NC 代码**）。

### 3. 用法示例

> 依赖：随权重发布的 `amoeba_train_perlayer.py` / `amoeba_eval.py`（当前尚未公开发布）；下例接口以脚本实际定义为准（`AmoebaModel`、`generate_with_exits`）。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer import AmoebaModel  # 项目 wrapper（待随权重发布）

BASE = "openbmb/MiniCPM5-2B"     # 替换为发布后的 Hub 仓库
HALT = "./halt_heads.pt"         # 随本模型发布（修复版）

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
if "halt_heads" in ckpt:
    model.halt_heads.load_state_dict(ckpt["halt_heads"])  # 修复版 ckpt

ids = tok("你好，请解释什么是动态早退。", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

评测时请同时报告 **强制退出 CE 曲线 / PPL / 平均退出层 / 贪心生成质量**（`amoeba_eval.py`），不要只看训练日志的 CE。旧版 checkpoint 在 `threshold` 较小时会发生退出坍缩（见第 5 节），建议先用保守阈值并配合 top-1 置信度门（`conf_min≈0.95`）。

### 4. 训练状态（诚实说明）

- **主线修复训练中（训练中 / 待发布）**：2026-11-22 核实，GPU 上 `/data/work/ckpt_2b_fixed/` 目录当前为空；进程 `amoeba_train_perlayer.py --model-path /data/models/MiniCPM5-2B --mode baseline --lam 0 --halt-loss-scale 0.1 --p-max 0.10 --rot-r 8 --steps 400 --out-dir /data/work/ckpt_2b_fixed` 正在运行（约 step 120/400）。该轮按"退出坍缩修复方案 A"配置：关闭早退奖励 `lam=0`、减弱均匀先验、加厚 CE 采样，目标是先恢复全层质量再重训 halt 决策。**尚无修复版权重可用于发布。**
- **旧 checkpoint（不可作为发布版）**：`ckpt_2b_full_BUGGY_2354/halt_heads.pt`（step 800，实测形状 `Linear(2049,1)×42`）与 `ckpt_2b_tune/`、`ckpt_2b_full/` 属于早期训练。其逐层 CE 在 L8 之后几乎平坦（≈11.3），而质量门控扫描显示：低置信度门时早退 PPL 高达数千（退出坍缩到浅层、生成乱码），只有 `conf_min=0.95, threshold=0.9` 时早退 PPL≈24.8、平均退出层≈38.5（接近全层，加速收益很小）。这些 checkpoint 仅作研究记录，**不应下载使用**。
- **训练数据**：CCI3-Data（BAAI，Apache-2.0）+ 对话 SFT / agent / 数学推理数据（详见 Citation）。修复轮仅使用 CCI3 子集。

### 5. License

- 基座 `openbmb/MiniCPM5-2B`：**Apache-2.0**。本模型为 Apache-2.0 允许的衍生作品，整体以 **Apache-2.0** 发布（随仓库附 LICENSE + NOTICE，保留上游版权声明）。
- Amoeba 新增部分（halt head 结构、wrapper、训练与评测脚本、本 model card）同样采用 **Apache-2.0**。
- 商标/背书声明：这是独立研究项目，**与 OpenBMB 无隶属或官方背书关系**；模型名不含 "official"。
- 数据署名：训练数据含 `nvidia/OpenMathReasoning`（CC-BY-4.0，见致谢）；**未复制** LayerSkip（CC-BY-NC）代码，仅按其论文公式自行实现。

### 6. Citation / 致谢

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-2B}
}
```

- 基座：OpenBMB, *MiniCPM5-2B* (Apache-2.0).
- 方法与思想来源：Schuster et al., *Confident Adaptive Language Modeling* (CALM); Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*（仅引用论文思想，不含其代码）.
- 数据：BAAI/CCI3-Data (Apache-2.0)、open-thoughts/OpenThoughts-Agent-SFT-100K (Apache-2.0)、nvidia/OpenMathReasoning (**CC-BY-4.0**，特此署名)。
- 感谢 GPU 实验环境与所有开源基座/数据发布方。

---

## English

**One-line summary**: Amoeba-2B is an independent research derivative of `openbmb/MiniCPM5-2B`. It attaches an ultra-lightweight token-level halt head after every Transformer layer so each token adaptively decides how many layers to traverse (dynamic depth / early exit), aiming for on-device speedups at a small quality cost. **Current status: training in progress / unreleased (see §5).**

### 1. Base model

| Field | Value |
|---|---|
| Name | `openbmb/MiniCPM5-2B` |
| Architecture | `LlamaForCausalLM` (`model_type: llama`), standard decoder-only Transformer; no `trust_remote_code` needed |
| Layers | 42 |
| hidden_size / intermediate_size | 2048 / 6144 |
| Attention | GQA, 16 Q / 2 KV heads, `head_dim=128`, RoPE `rope_theta=5e6` |
| Vocabulary | 130560 (BPE, ChatML style; `eos=[1, 130073]`, `pad=1`) |
| Context | 131072 (128K) |
| Precision | bfloat16, `tie_word_embeddings=false` |
| Base license | **Apache-2.0** (per official README) |

### 2. Amoeba modification

- **Halt head**: after each `model.layers[i]` output we place a binary gate `nn.Linear(hidden_size + 1, 1)` — for 2B, `Linear(2049, 1) × 42`, ~86k new parameters. Input = hidden state `h_i` concatenated with a token-level uncertainty scalar `unc = tanh(-log(p_max))`. Shapes were verified on the GPU checkpoint (`halt_heads.pt`: `0.weight` = `(1, 2049)`, cfg `n_layer=42, hidden_size=2048, vocab_size=130560`).
- **Implementation**: an `AmoebaModel` wrapper performs layer-by-layer forward **without touching `modeling_llama.py`**, preserving all base weights, tokenizer, KV cache and `generate` compatibility. Intermediate logits are computed via `norm + lm_head`; SDPA attention is required (eager attention needs an explicit 4D causal mask).
- **Dynamic activation at inference**: each token accumulates exit probability `cum = 1 - ∏(1 - p_l)`; the first layer where `cum` exceeds `threshold` emits the token via `norm + lm_head` and skips all later layers. Higher threshold = more conservative (closer to all 42 layers). Training uses per-layer CE + anchor + uniformity prior + semantic consistency + entropy regularization, reimplemented from published ideas (**no LayerSkip CC-BY-NC code copied**).

### 3. Usage

> Requires the project wrapper scripts (`amoeba_train_perlayer.py` / `amoeba_eval.py`), not yet publicly released; adapt to the actual API at release time.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer import AmoebaModel  # project wrapper (to be released)

BASE = "openbmb/MiniCPM5-2B"     # replace with the released Hub repo
HALT = "./halt_heads.pt"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
if "halt_heads" in ckpt:
    model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("Hello, explain dynamic early exit.", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

For evaluation, always report **forced-exit CE curve / PPL / mean exit layer / greedy generation quality** (`amoeba_eval.py`), not just training CE. Older checkpoints collapse to shallow exits at small thresholds; use a conservative threshold plus a top-1 confidence gate (`conf_min≈0.95`) until the fix is validated.

### 4. Training status (honest disclosure)

- **Mainline fix run IN PROGRESS (unreleased)**: verified on 2026-11-22 — `/data/work/ckpt_2b_fixed/` is currently empty, while `amoeba_train_perlayer.py --model-path /data/models/MiniCPM5-2B --mode baseline --lam 0 --halt-loss-scale 0.1 --p-max 0.10 --rot-r 8 --steps 400 --out-dir /data/work/ckpt_2b_fixed` is running (~step 120/400). This is "fix plan A": `lam=0`, weaker uniformity prior, thicker CE sampling — first recover full-layer quality, then retrain halt decisions. **No fixed weights are available for release yet.**
- **Old checkpoints (not for release)**: `ckpt_2b_full_BUGGY_2354/halt_heads.pt` (step 800, verified `Linear(2049,1)×42`), `ckpt_2b_tune/`, `ckpt_2b_full/`. Their per-layer CE is nearly flat after L8 (~11.3); quality-gate scans show early-exit PPL in the thousands at low confidence gates (exit collapse / gibberish), while `conf_min=0.95, threshold=0.9` yields PPL≈24.8 at mean exit layer ≈38.5 (near full depth, little speedup). These are research records only, **do not download**.
- **Training data**: CCI3-Data (BAAI, Apache-2.0) plus dialogue SFT / agent / math-reasoning data (see citations); the fix run uses only a CCI3 subset.

### 5. License

- Base `openbmb/MiniCPM5-2B` is **Apache-2.0**; derivative weights are released under **Apache-2.0** (LICENSE + NOTICE included; upstream copyright notices preserved).
- All Amoeba additions (halt heads, wrapper, training/eval scripts, this card) are **Apache-2.0**.
- Trademark: independent research, **not affiliated with or endorsed by OpenBMB**; no "official" in the model name.
- Data attribution: training data includes `nvidia/OpenMathReasoning` (**CC-BY-4.0**, credited above); **no LayerSkip (CC-BY-NC) code** was copied — only its published ideas were reimplemented.

### 6. Citation / acknowledgements

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-2B}
}
```

- Base: OpenBMB, *MiniCPM5-2B* (Apache-2.0).
- Method inspiration: Schuster et al., *Confident Adaptive Language Modeling* (CALM); Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* (ideas only, no code).
- Data: BAAI/CCI3-Data (Apache-2.0), open-thoughts/OpenThoughts-Agent-SFT-100K (Apache-2.0), nvidia/OpenMathReasoning (**CC-BY-4.0**, credited).
- Thanks to the open-source base/data communities and the GPU environment used for experiments.
