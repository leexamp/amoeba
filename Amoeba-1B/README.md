> **Updated (Amoeba-DA).** This directory documents the original token-level early-exit research line, which was investigated and **rejected**. The shipped technique is **Amoeba-DA** (parallel draft heads + full-model verification): 2B 2.02×, 1B 1.30×, statistically lossless (not bit-exact). See the root `README.md` / `README_zh.md` and `docs/动态激活_完整实验矩阵.md`.

---
license: apache-2.0
base_model:
  - openbmb/MiniCPM5-1B
library_name: transformers
pipeline_tag: text-generation
tags:
  - dynamic-activation
  - early-exit
  - halt-head
  - on-device
  - research-preview
---

# Amoeba-1B

## 中文

**一句话简介**：Amoeba-1B 是 `openbmb/MiniCPM5-1B` 的独立研究衍生模型，通过对 24 层 Transformer 逐层挂载 token 级 halt head 实现按 token 自适应的动态深度推理；已完成一轮全参研究训练，权重已产出，但因退出坍缩问题仍待修复，当前为**研究预览 / 待正式发布**。

### 1. 基座模型（Base model）

| 项目 | 值 |
|---|---|
| 名称 | `openbmb/MiniCPM5-1B` |
| 架构 | `LlamaForCausalLM`（`model_type: llama`），与 2B 完全同构，无需 `trust_remote_code` |
| 层数 | 24 |
| hidden_size / intermediate_size | **1536 / 4608**（注意：不是旧文档中误写的 2048/6144） |
| 注意力 | GQA 16 个 Q 头 / 2 个 KV 头，`head_dim=128`，RoPE `rope_theta=5e6` |
| 词表 | 130560（与 2B 共用 tokenizer 与特殊 token：`eos=[1, 130073]`，`pad=1`） |
| 上下文 | 131072（128K） |
| 精度 | bfloat16，`tie_word_embeddings=false`，参数量约 1.08B |
| 基座 License | **Apache-2.0** |

### 2. Amoeba 改造说明

- **Halt head 结构**：每层输出挂 `nn.Linear(hidden_size + 1, 1)`；1B 为 **`Linear(1537, 1) × 24`**，新增参数约 3.7 万。输入为该层隐状态 `h_i` 与 token 级不确定性标量 `unc=tanh(-log(p_max))` 的拼接。已由 GPU 上 `ckpt_1b_boost/halt_heads.pt` 元信息实测核实：cfg 为 `n_layer=24, hidden_size=1536, vocab_size=130560, eos=[1,130073], pad=1`，权重形状 `(1, 1537)`。
- **动态激活**：与 2B 相同——推理时逐层累计 `cum = 1-∏(1-p_l)`，`cum ≥ threshold` 即经共享 `norm + lm_head` 输出该 token 并提前退出；每个 token 独立决定退出层，可与 loop 机制正交。wrapper 逐层 forward，不修改 `modeling_llama.py`，保护 KV cache / generate。
- **训练配置（已核实）**：`ckpt_1b_boost/halt_heads.pt` 为 step 1000、`mode=full`，含自蒸馏（`kd=true, kd_weight=1.0, kd_temp=2.0, hidden_kd_weight=0.1`）、`halt_loss_scale=0.5`、`base_lr=2e-5 / head_lr=1e-4`、基座解冻（`base_unfrozen=true`）、layer dropout `p_max=0.2`、轮转出口 `rot_r=4`。基座与 tokenizer 完整保存在 `ckpt_1b_boost/full_model/`（含 `config.json`、`generation_config.json`、`model.safetensors` 约 2.1GB、`tokenizer.json` 等，已核实）。

### 3. 用法示例

> 依赖项目 wrapper `amoeba_train_perlayer.py`（待随权重发布）；接口以实际脚本为准。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer import AmoebaModel  # 项目 wrapper（待发布）

BASE = "./full_model"            # 发布后替换为 Hub 仓库（微调后的基座 + tokenizer）
HALT = "./halt_heads.pt"         # step-1000 研究用 halt 决策权重

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("你好，请写一段关于动态激活的简短说明。", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

由于已发现退出坍缩（第 4 节），请使用**保守阈值**（如 0.95）并同时报告强制退出 CE / PPL / 平均退出层；`threshold` 过小会退到浅层并生成乱码。

### 4. 训练状态（诚实说明）

- **已产出研究用 checkpoint，待修复后发布**：`ckpt_1b_boost/full_model/`（完整基座权重）+ `ckpt_1b_boost/halt_heads.pt`（step 1000）均已落盘，GPU 上已核实。另有早期 `ckpt_1b/`、`ckpt_1b_perlayer/` 作为对照。
- **存在已知问题**：日志显示训练不稳定——`lm_ce` 在 14→25→12 间过山车，KD loss 在 1.8↔345 间锯齿震荡，梯度范数出现 7722~8311 的尖峰；逐层 CE 在 L4 之后几乎平坦（≈11.7），即"浅层与深层预测质量无差别"，符合退出坍缩特征。质量门控验证尚未在本版本完成。**该 checkpoint 仅供研究复现，不应作为稳定发布版。**
- **修复方向**：按项目"退出坍缩修复方案"——`lam=0`、弱化均匀先验/熵正则、冻结教师自蒸馏（严禁 `kd_weight=1.0` 的移动教师）、加厚 CE 采样，并在冻结基座上单独重训/校准 halt 决策；修复完成并通过对齐口径的质量-速度评测后再正式发布。

### 5. License

- 基座 `openbmb/MiniCPM5-1B`：**Apache-2.0**；衍生权重整体以 **Apache-2.0** 发布（附 LICENSE + NOTICE，保留上游版权声明）。
- Amoeba 新增部分（halt head、wrapper、训练/评测脚本、本 card）同样为 **Apache-2.0**。
- 商标/背书：独立研究项目，**与 OpenBMB 无隶属或官方背书关系**。
- 数据署名：训练语料含 `nvidia/OpenMathReasoning`（CC-BY-4.0，见致谢）；未复制 LayerSkip（CC-BY-NC）代码。

### 6. Citation / 致谢

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-1B}
}
```

- 基座：OpenBMB, *MiniCPM5-1B* (Apache-2.0).
- 方法思想：Schuster et al., *CALM*; Elhoushi et al., *LayerSkip*（仅论文思想，未用其代码）。
- 数据：BAAI/CCI3-Data (Apache-2.0)、open-thoughts/OpenThoughts-Agent-SFT-100K (Apache-2.0)、nvidia/OpenMathReasoning (**CC-BY-4.0**，特此署名)。

---

## English

**One-line summary**: Amoeba-1B is an independent research derivative of `openbmb/MiniCPM5-1B`. It attaches token-level halt heads to all 24 Transformer layers for per-token adaptive depth. One full research training run has completed and weights exist, but exit collapse is a known unresolved issue — currently a **research preview / pending official release**.

### 1. Base model

| Field | Value |
|---|---|
| Name | `openbmb/MiniCPM5-1B` |
| Architecture | `LlamaForCausalLM` (`model_type: llama`), fully isomorphic to the 2B model; no `trust_remote_code` |
| Layers | 24 |
| hidden_size / intermediate_size | **1536 / 4608** (not the outdated 2048/6144 assumption) |
| Attention | GQA, 16 Q / 2 KV heads, `head_dim=128`, RoPE `rope_theta=5e6` |
| Vocabulary | 130560 (same tokenizer/specials as 2B: `eos=[1, 130073]`, `pad=1`) |
| Context | 131072 (128K) |
| Precision | bfloat16, `tie_word_embeddings=false`, ~1.08B parameters |
| Base license | **Apache-2.0** |

### 2. Amoeba modification

- **Halt head**: one `nn.Linear(hidden_size + 1, 1)` per layer — for 1B, **`Linear(1537, 1) × 24`**, ~37k new parameters. Input = layer hidden state `h_i` plus token-level uncertainty `unc=tanh(-log(p_max))`. Verified from GPU checkpoint metadata (`ckpt_1b_boost/halt_heads.pt`): cfg `n_layer=24, hidden_size=1536, vocab_size=130560, eos=[1,130073], pad=1`, weight shape `(1, 1537)`.
- **Dynamic activation**: as in 2B — at inference each token accumulates `cum = 1-∏(1-p_l)` and exits at the first layer where `cum ≥ threshold`, via the shared `norm + lm_head`. Layer-by-layer wrapper forward leaves `modeling_llama.py` untouched, preserving KV cache / generate.
- **Training config (verified)**: `ckpt_1b_boost/halt_heads.pt` is step 1000, `mode=full`, with self-distillation (`kd=true, kd_weight=1.0, kd_temp=2.0, hidden_kd_weight=0.1`), `halt_loss_scale=0.5`, `base_lr=2e-5 / head_lr=1e-4`, base unfrozen, layer dropout `p_max=0.2`, rotating exits `rot_r=4`. Full fine-tuned base and tokenizer are stored in `ckpt_1b_boost/full_model/` (`model.safetensors` ~2.1GB, verified).

### 3. Usage

> Requires the project wrapper (`amoeba_train_perlayer.py`, to be released); adapt to the actual API.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amoeba_train_perlayer import AmoebaModel  # project wrapper (to be released)

BASE = "./full_model"            # replace with released Hub repo (fine-tuned base + tokenizer)
HALT = "./halt_heads.pt"         # step-1000 research halt weights

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model = AmoebaModel(base).eval().cuda()

ckpt = torch.load(HALT, map_location="cpu", weights_only=False)
model.halt_heads.load_state_dict(ckpt["halt_heads"])

ids = tok("Hello, briefly describe dynamic activation.", return_tensors="pt").input_ids.cuda()
out = model.generate_with_exits(ids, max_new_tokens=64, threshold=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

Because exit collapse is known (§4), use a **conservative threshold** (e.g. 0.95) and report forced-exit CE / PPL / mean exit layer alongside generation; small thresholds collapse to shallow layers and produce gibberish.

### 4. Training status (honest disclosure)

- **Research checkpoint produced; pending fixed release**: `ckpt_1b_boost/full_model/` (full base weights) + `ckpt_1b_boost/halt_heads.pt` (step 1000) are on disk (verified). Earlier `ckpt_1b/` and `ckpt_1b_perlayer/` serve as ablations.
- **Known issues**: logs show instability — `lm_ce` swings 14→25→12, KD loss oscillates 1.8↔345, gradient norms spike to 7722–8311; per-layer CE is nearly flat after L4 (~11.7), i.e. shallow and deep layers have indistinguishable prediction quality — a signature of exit collapse. Quality-gate validation has not been completed for this version. **Research reproduction only, not a stable release.**
- **Fix direction**: per the project's exit-collapse plan — `lam=0`, weaker uniformity/entropy terms, frozen-teacher distillation (never `kd_weight=1.0` with a moving teacher), thicker CE sampling, then retrain/calibrate halt decisions on a frozen base; official release only after aligned-quality speed/quality evaluation passes.

### 5. License

- Base `openbmb/MiniCPM5-1B` is **Apache-2.0**; derivative weights released under **Apache-2.0** (LICENSE + NOTICE, upstream notices preserved).
- All Amoeba additions (halt heads, wrapper, training/eval scripts, this card) are **Apache-2.0**.
- Trademark: independent research, **not affiliated with or endorsed by OpenBMB**.
- Data attribution: training data includes `nvidia/OpenMathReasoning` (**CC-BY-4.0**, credited); no LayerSkip (CC-BY-NC) code copied.

### 6. Citation / acknowledgements

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: Token-Level Dynamic Activation via Per-Layer Halt Heads},
  year   = {2026},
  note   = {Independent research project; model card for Amoeba-1B}
}
```

- Base: OpenBMB, *MiniCPM5-1B* (Apache-2.0).
- Method inspiration: Schuster et al., *CALM*; Elhoushi et al., *LayerSkip* (ideas only, no code).
- Data: BAAI/CCI3-Data (Apache-2.0), open-thoughts/OpenThoughts-Agent-SFT-100K (Apache-2.0), nvidia/OpenMathReasoning (**CC-BY-4.0**, credited).
