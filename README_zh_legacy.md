# Amoeba：验证式动态激活语言模型家族

> **Amoeba** 是一族具备 **Amoeba-DA（验证式动态激活）** 与工具调用（agent）能力的语言模型。Amoeba-DA 用 K 个并行草稿头从同一份 hidden 一次性猜出 `t+2..t+K+1`，再由**未修改的基座模型**一次前向并行验证；只有当候选与基座自身贪心 argmax 一致时才接受，因此输出与全层贪心在统计意义上等价。
>
> **状态（已更新）**：早期的 token 级早退（halt head）路线已被系统否定——16 组实测实验在这套 42 层深窄架构上全部失败。当前发布的技术是 **Amoeba-DA**：2B 实测 **2.02×**（27.1 → 54.7 tok/s，RTX 4080S），1B 实测 **1.30×**。工具调用已包含（2B/1B 用 LoRA SFT，0.6B/0.3B 用原生工具模板）。详见第 5 节与 `docs/动态激活_完整实验矩阵.md`。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心创新](#2-核心创新)
- [3. 模型家族](#3-模型家族)
- [4. 使用方法](#4-使用方法)
- [5. 性能与加速](#5-性能与加速)
- [6. 训练方法](#6-训练方法)
- [7. License 与致谢](#7-license-与致谢)
- [8. 已知限制](#8-已知限制)
- [9. 引用](#9-引用)

---

## 1. 项目简介

传统 Transformer 对每个 token 都执行**全部层**的计算，无论这个 token 是简单的标点、常见的停用词，还是需要深度推理的关键字。然而直觉上，不同 token 的"难度"并不相同——大量 token 在浅层就已经可以被可靠预测。

**Amoeba** 的核心目标是：**让不同 token 走不同深度**。我们在预训练基座（MiniCPM5、Qwen3、MNN-Pocket 等）的每一层后挂一个极轻量的 **halt head**（`Linear(hidden+1, 1)`，参数量仅数万级）。推理时，每个 token 逐层累积"可以退出"的概率，一旦累积概率达到阈值就提前退出，跳过后续所有层。

设计动机与取舍：

- **按需计算**：简单 token 浅层退出，难 token 走得更深，期望在不显著掉质量的前提下节省算力。
- **端侧 / 本地优先**：在消费级显卡或端侧设备上，用可接受的质量损失换取可观的延迟下降，是划算的权衡。
- **跨架构验证**：模型族覆盖 3 个模型族（MiniCPM5 / Qwen3 / MNN）× 4 个尺寸，系统研究"动态激活收益 vs 模型规模"的 scaling 关系。
- **诚实的研究定位**：early-exit 方向"研究多、落地少"，核心硬伤是掉质量。Amoeba 的差异化在于用**熵正则 + 语义一致性**的 halt 损失改善"置信度判断不准"这一痛点，并以开放、可复现的方式记录成败。

> 本项目是独立研究项目，与 OpenBMB、Alibaba、Taobao/MNN 等上游机构**无官方关联或背书**。

---

## 2. 核心创新

### 2.1 每层 halt head 与 token 级累积退出

在基座的每一层 `l` 之后挂一个 halt head：

```
score_l = Linear(hidden_size + 1, 1)( [h_l, unc_l] )
p_l     = sigmoid(score_l)
```

- `h_l`：该层输出的隐藏状态；
- `unc_l`：token 级不确定性标量，`unc = tanh(-log(p_max))`，其中 `p_max` 是该位置词表分布的最大概率；
- `+1` 即把不确定性作为一个额外特征拼接到 hidden 上。

推理时按层累积退出概率：

```
remaining_0 = 1
q_l         = remaining_l · p_l              # 该 token 在第 l 层退出的概率
remaining_{l+1} = remaining_l · (1 - p_l)
```

当累积退出概率 `F(l) = 1 - remaining_{l+1}` 首次达到阈值 `τ` 时，该 token 退出。**阈值越高越保守（走更多层）**，详见[使用方法](#4-使用方法)。

### 2.2 每层强化训练（per-layer exit strengthening）

早期 checkpoint 暴露出一个关键问题：**基座只在最后一层被 CE 监督过，中间层的隐藏状态并不"可直接读出正确词"**，强制在浅层退出会导致 CE 爆炸（实测 L6 CE≈13.5、PPL≈7×10⁵，而末层 L42 CE≈2.39）。为此我们采用每层强化方案：

- **逐层独立 LM loss（核心）**：对多个启用出口分别计算 `CE(lm_head(norm(h_l)), y)`，并**解冻基座**让梯度真正回传到 decoder 层，逼迫中间层学会"可退出"。早期 `--freeze-base` 的训练中，这条 loss 只是"没有梯度的只读探针"——这是必须修正的范畴错误。
- **Layer dropout（放大器）**：训练时以 `p_l = D(l)·p_max` 的概率随机跳过前面的层，迫使后层学会"补位"、浅层自己承载预测信息（借鉴 LayerSkip 思想，自行实现）。
- **自蒸馏 / 语义一致性（精修）**：以最后一层 L42 的 logits / hidden 作为教师，用 KL 散度 + cosine 相似度约束中间出口，使其语义与末层对齐，让 halt head 的"何时可退"判断更可靠。
- **halt 损失**：均匀先验（避免退出分布全挤在某一层）+ 熵正则（防止决策过早塌缩成 one-hot）+ 语义一致性 + 时间惩罚。

> 核心洞察：**halt head 只是"裁判"，基座中间层的表示质量才是"运动员"。运动员没练好之前，裁判无从发挥。** 因此训练重点是先让中间层真正可退出，再校准退出决策。

### 2.3 关键工程约束

- **不改 `modeling_*.py` 内部前向循环**，而是用 wrapper 持有基座并逐层 forward，避免破坏 KV cache / `generate`。
- 所有维度（`hidden_size`、层数、特殊 token id）从 config 动态读取，跨基座复用同一套 wrapper 逻辑。

---

## 3. 模型家族

| 版本 | 基座 | 架构 | 层 | hidden | 词表 | 状态 |
|---|---|---|---:|---:|---:|---|
| **Amoeba-2B** | MiniCPM5-2B | LlamaForCausalLM | 42 | 2048 | 130560 | 主线；基座权重就绪，动态激活训练与调优进行中 |
| **Amoeba-1B** | MiniCPM5-1B | LlamaForCausalLM | 24 | 1536 | 130560 | 基座权重就绪；复制流程最容易 |
| **Amoeba-0.6B** | Qwen3-0.6B | Qwen3ForCausalLM | 28 | 1024 | 151936 | 适配完成，已启动训练 |
| **Amoeba-0.3B** | MNN-Pocket-0.3B | Qwen3ForCausalLM | 28 | 896 | 65536 | 适配完成，已启动训练（小词表 / 小容量，坑最多） |
| **Amoeba-120M** | 自研小模型（testbed） | — | 16 | 768 | 4104 | 机制验证平台；基座质量受历史 4104 词表路线限制，仅用于验证动态激活机制 |

补充说明：

- **2B / 1B** 为 MiniCPM5 系列，标准 `LlamaForCausalLM`，128K 上下文，与 halt head 改造最契合。
- **0.6B / 0.3B** 为 `Qwen3ForCausalLM`（含 q_norm/k_norm），transformers 原生支持；0.3B 来自 2026-11 新发布的 MNN-Pocket，词表仅 65536。
- **120M testbed** 已验证动态激活机制本身可用（阈值 1.0→退出 16 层/全层；0.8→7.1 层；0.5→3.77 层，能按阈值自适应控制深度），但因基座质量太差（生成重复垃圾），**不用于评估早退质量损失**。
- 跨架构、跨尺寸的横向对照旨在产出 scaling 曲线：哪个尺寸早退收益最大、质量-速度权衡如何随规模变化。

---

## 4. 使用方法

### 4.1 加载基座并包装

Amoeba 以 wrapper 的形式工作：加载标准基座后，用 `AmoebaModel` 持有它并挂载每层 halt head。下面代码为**简化示意**，说明核心接口；生产实现还包含 KV cache、位置嵌入与逐层调度的完整处理。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "openbmb/MiniCPM5-2B"   # 按版本替换为对应基座

tok  = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",   # 依赖 SDPA 自动因果掩码
)
L = base.config.num_hidden_layers
H = base.config.hidden_size

class AmoebaModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        # 每层一个 halt head：输入 = hidden + 不确定性，故为 H+1 -> 1
        self.halt_heads = nn.ModuleList(
            [nn.Linear(H + 1, 1) for _ in range(L)]
        )

    # 计算某层的退出概率 p_l（示意：h_l 为该层 hidden，p_max 为分布最大概率）
    def exit_prob(self, l, h_l, p_max):
        unc = torch.tanh(-torch.log(p_max.clamp_min(1e-8)))   # token 级不确定性
        return torch.sigmoid(self.halt_heads[l](torch.cat([h_l, unc], dim=-1)))

model = AmoebaModel(base).cuda().eval()
# 训练完成后加载 halt head 权重：
# model.halt_heads.load_state_dict(torch.load("halt_heads.pt", map_location="cpu")["halt_heads"])
```

### 4.2 动态推理（threshold 参数）

- `threshold`（记作 `τ`）是**累积退出概率阈值**：当某 token 的累积退出概率 `F(l)` 首次 ≥ τ 时退出。
- **τ 越高越保守、走越多层**：`τ=1.0` 表示永不早退（跑完整层，作为全层延迟基线）；`τ` 越小越激进、越早退出。
- 阈值**不是"置信度"**，也不是质量门控，而是一个需要按任务在验证集上校准的操作点（见「已知限制」）。

```python
@torch.no_grad()
def generate_dynamic(model, input_ids, threshold=0.8, max_new_tokens=64, eos_id=1):
    """示意：逐 token 累积退出概率，达标即停。生产版需配合 KV cache。"""
    out = input_ids
    for _ in range(max_new_tokens):
        remaining = torch.ones(out.shape[0], 1, device=out.device)
        used_layers = []
        hidden = None
        for l, layer in enumerate(model.base.model.layers):
            # 实际实现需计算 position_embeddings / attention_mask / KV cache
            hidden = layer(hidden if hidden is not None else model.base.model.embed_tokens(out))
            logits = model.base.lm_head(model.base.model.norm(hidden[:, -1:, :]))
            p_max  = logits.float().softmax(-1).max(-1).values          # [B, 1]
            p      = model.exit_prob(l, hidden[:, -1, :], p_max)         # [B, 1]
            remaining = remaining * (1 - p)
            if ((1 - remaining) >= threshold).all():                     # 全 batch 达标
                used_layers.append(l + 1)
                break
        nxt = logits.argmax(-1)
        out = torch.cat([out, nxt], dim=-1)
        if l == model.base.config.num_hidden_layers - 1:
            pass  # 末层必退
        if (nxt == eos_id).all():
            break
    return out
```

经验参考（120M testbed 实测）：`τ=1.0 → 平均 16 层（全层）`、`τ=0.8 → 约 7.1 层`、`τ=0.5 → 约 3.77 层`。**这些层数不能脱离质量单独解读**——必须同时看退出点上的 CE/PPL。

### 4.3 强制退出评测

为衡量"每一层单独作为出口"的质量，可强制所有样本在第 `l` 层退出，计算 CE/PPL，画出"退出层 vs 质量"曲线：

```python
@torch.no_grad()
def forced_exit_ce(model, input_ids, targets, exit_layer):
    h = model.base.model.embed_tokens(input_ids)
    for l, layer in enumerate(model.base.model.layers):
        h = layer(h)                       # 示意：省略 position/mask/cache
        if l == exit_layer:
            logits = model.base.lm_head(model.base.model.norm(h))
            return nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
    raise ValueError("exit_layer 越界")

# 扫描全部层，得到强制退出 CE 曲线（L6≈13.5 ... L42≈2.39 为未强化时的基线形态）
for l in [5, 11, 17, 23, 29, 35, 41]:
    print(f"L{l+1}: CE={forced_exit_ce(model, inp, tgt, l):.3f}")
```

> 注：不同基座的 `eos_id` 不同——MiniCPM5 为 `1`，Qwen3-0.6B 为 `151645`，MNN-Pocket-0.3B 为 `65512`；Qwen3 系列还需注意 tied embeddings 与 `q_norm/k_norm`。完整实现请参考项目内的 wrapper 与评测脚本。

---

## 5. 性能与加速（Amoeba-DA 实测）

### 5.1 实测加速

| 模型 | 全层贪心基线 | Amoeba-DA | 加速 | 每步平均接受 token |
|---|---:|---:|---:|---:|
| Amoeba-2B | 27.1 tok/s | 54.7 tok/s | **2.02×** | 2.61 |
| Amoeba-1B | 44.7 tok/s | 58.0 tok/s | **1.30×** | — |

2B 草稿头在 chat 留出集上的 top-1 一致率为 0.469 / 0.348 / 0.293 / 0.240（offset 2–5）。关键发现：用 **on-policy 自生成对话文本**（CE 1.31）而非离线网页语料（CE 3.71）训头，α₂ 从 0.138 提升到 0.469，加速从 1.13× 提升到 2.02×。

### 5.2 质量与一致性口径

Amoeba-DA 只在候选与基座 argmax 一致时接受。但在 **bf16** 下，批量验证与单 token 贪心对近邻（logits 差约 0.01）的舍入方向不同，因此输出**并非逐 bit 相同**：20 条中 6 条完全一致，位置 token 匹配率 **83.7%**，平均匹配前缀 75.9 token。准确说法是**统计无损**，不是逐位无损。

### 5.3 为什么否定 token 级早退

token 级早退、跳中间层、跳 attention、跳 MLP、置信度门控共 16 组实验全部失败。42 层深窄结构层间没有冗余：保留 76% 的层 CE 仍高达 11.7（≈ ln V，等同随机）。门控成本约 1.25 层（一次 `lm_head`），而可用退出深度 ≥40/42，理论天花板仅 ≈1.16×。完整数据见 `docs/动态激活_完整实验矩阵.md`。



---

## 5.5 使用注意与已知限制（部署前必读）

1. **贪心解码在长回答上会陷入重复循环**。2B 在长角色扮演中把同一句话重复 20+ 次。
   请使用推荐采样参数 `temperature=0.8, top_p=0.95, repetition_penalty=1.05`。
   所有能力对比都基于从官方 GGUF 模型卡恢复的 chat template（开启 `<think>`）；
   我们短子集贪心分数**不可与官方模型卡数字直接比较**（官方 MATH-500 94.6 等，
   使用 temperature=1.0 采样与更大生成预算）。
2. **工具调用需要带工具定义的 system message**，`agent_infer.py` 已处理。
   裸 chat template 直接问天气，模型会回答"没有实时数据"——这是 prompt 问题，不是能力问题。
3. **训练数据更正**：早期实验用过 CCI3-Data，但它是噪声很大的中文网页/新闻语料
   （CE 3.7，而自生成对话为 1.31），**未用于最终发布的 checkpoint**。
   发布模型的工具调用 SFT 使用的是从基座自生成的合成工具对话数据。
   使用 OpenMathReasoning 的数学 CoT 实验会损伤 MMLU，**已丢弃**，不在发布模型中。
4. **0.6B 工具模型**：第一版 2 epoch 过拟合（loss 0.0008）导致空回复；
   发布的是 `ckpt_06b_agent_v2`（1 epoch / lr 5e-5）。
5. **草稿头只随 2B/1B 发布**：0.6B/0.3B 实测 0.60× / 0.36×（变慢），不作为加速特性交付。
6. **本环境 vLLM 不可用**（0.11 与 torch 2.15+cu132 ABI 不兼容），
   所有速度数据来自 HuggingFace `generate`，受 CPU/Python 限制。

---

## 6. 训练方法

Amoeba 的训练分阶段推进，核心是"先练运动员（基座中间层），再校准裁判（halt head）"：

1. **复现 baseline**：先按 LayerSkip 思路自行实现 layer dropout + 逐层 exit loss，回答"改造基座能否跑通、掉多少质量"。LayerSkip 在 1.5B 上报告：exit layer 6 速度约 2×，但质量从 85.39% 掉到 82.9%（约 2.5%），末层还有约 3% 回退——这是我们要改善的基线。
2. **每层强化（核心）**：解冻基座，用 uniform / inverse-CE 权重的逐层独立 LM loss 把梯度打进 decoder 层，配合末层 CE anchor 防止灾难性遗忘；叠加 layer dropout 强化（继续预训练 `p_max≈0.1`，任务微调可升到 0.2）。
3. **语义对齐（精修）**：以末层为教师做 logits KL + hidden cosine 的自蒸馏，提升中间出口与末层的语义一致性。
4. **halt head 重校准**：基座分布改变后，冻结基座、只训 halt heads，恢复完整 halt 损失（均匀先验 / 熵正则 / 语义一致性 / 时间惩罚），并按 CALM 的 LTT/FST 思路在验证集上校准退出阈值，替换硬编码值。
5. **评测与消融**：对比强制退出 CE 曲线、退出分布、质量-速度曲线；对 B（逐层 loss）/ C（layer dropout）/ A（蒸馏）各组合做小型消融。

显存与成本方面：以 RTX 4080S 32GB + bf16 + 8-bit AdamW 计，2B 全参微调约 15GB（加冻结教师副本约 19GB），完全可行。小尺寸版本依次把 2B 上验证的方法复制到 1B → 0.6B → 0.3B。

> 实现上我们**阅读论文、理解思想后自行编码**，不复制 LayerSkip 的仓库代码（见 License 一节）；同时参考 Apache-2.0 的 CALM 实现以少踩坑。

---

## 7. License 与致谢

### 7.1 权重与代码 License

- 本项目的 **Amoeba 权重以 Apache-2.0 发布**；仓库根目录包含 `LICENSE`（Apache-2.0 全文）与 `NOTICE`（列明上游版权声明）。
- 各基座 License：

| 基座 | License | 说明 |
|---|---|---|
| MiniCPM5-2B / MiniCPM5-1B | **Apache-2.0** | OpenBMB；可商用、可发布衍生模型 |
| Qwen3-0.6B | **Apache-2.0** | Alibaba/Qwen；可商用、可发布衍生模型 |
| MNN-Pocket-0.3B | **Apache-2.0** | Taobao/MNN；可商用、可发布衍生模型 |

- **LayerSkip（facebookresearch）**：代码为 **CC-BY-NC**（仅非商用研究）。我们**仅参考其论文思想**（layer dropout + exit loss），**未复制、未改写其任何代码**；需要代码参考时优先使用 Apache-2.0 的 **CALM**。论文可正常引用（citation 不受代码 license 限制）。
- **训练数据**：CCI3-Data、OpenThoughts-Agent-SFT-100K 为 Apache-2.0；`nvidia/OpenMathReasoning` 为 **CC-BY-4.0**（model card 中已署名）；Mixture-of-Thoughts 的 reasoning trace 蒸馏自 DeepSeek-R1（MIT，允许商用与蒸馏），数据卡未标注 license，发布时按保守口径处理。

### 7.2 各 checkpoint 的 base 归属

| Checkpoint | 基座（base model） | 修改说明 |
|---|---|---|
| Amoeba-2B | `openbmb/MiniCPM5-2B` | 在基座上新增 token 级 halt heads（动态激活），并对基座中间层做每层强化微调 |
| Amoeba-1B | `openbmb/MiniCPM5-1B` | 同上 |
| Amoeba-0.6B | `Qwen/Qwen3-0.6B` | 同上（Qwen3ForCausalLM 架构适配） |
| Amoeba-0.3B | `MNN/MNN-Pocket-0.3B` | 同上（Qwen3ForCausalLM 架构适配，小词表 65536） |

每个 checkpoint 的 model card 均声明：`base_model`、修改说明、"This is an independent research project, not affiliated with OpenBMB / Alibaba / Taobao" 等非官方背书声明。

### 7.3 致谢

感谢 OpenBMB（MiniCPM5）、Qwen 团队（Qwen3）、MNN 团队（MNN-Pocket）开源高质量基座；感谢 Google DeepMind 的 CALM 与 Meta 的 LayerSkip 在自适应计算 / early-exit 方向上的开创性工作；感谢 CCI3、OpenThoughts、NVIDIA OpenMathReasoning、DeepSeek-R1 等开放数据与模型。

---

## 8. 已知限制

1. **动态激活质量仍在调优**：当前 checkpoint 存在退出坍缩，早退生成质量显著低于全层基线；在方法稳定前不建议生产使用，相关加速数字不可信。
2. **纯文本优先，多模态后续**：当前版本聚焦纯文本；多模态动态激活是后续方向，尚未实现。
3. **退出阈值需按任务校准**：`threshold` 是未校准的操作点，并非置信度或质量门控。同一阈值在不同任务 / 数据分布上的质量-速度表现差异很大，需在验证集上联合 (质量约束, 延迟) 扫描选择。
4. **指标口径需谨慎解读**：训练侧的"软期望退出层"与评测侧的"硬退出分位数"定义不同；报告退出层时必须同时报告该层的 CE/PPL，避免"早退=高质量"的误读。
5. **小基座额外约束**：0.3B 词表仅 65536，外部语料必须重新 tokenize 并断言 id 越界；Qwen3 系列需处理 tied embeddings 与 q_norm/k_norm；部分基座 config 的 `use_cache=false`，评测前需显式打开。
6. **长上下文 / KV cache**：早退与 KV cache 的正确配合是持续工程项；配合不佳时早退可能无法转化为实际延迟收益。

---

## 9. 引用

如本项目对您的研究有帮助，请引用相关基线工作：

- Schuster et al., **Confident Adaptive Language Modeling (CALM)**, 2022. — 置信度自适应 per-token 退出。
- Elhoushi et al., **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding**, 2024. — layer dropout + early-exit loss + 自投机解码。
- 各基座论文与技术报告：MiniCPM5（OpenBMB）、Qwen3（Alibaba）、MNN-Pocket（Taobao/MNN）。

---

> **Amoeba** — *Go as deep as needed, no deeper.*
