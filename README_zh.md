# Amoeba

**面向小语言模型的、以能力保持为目标的动态激活研究。**

Amoeba 是一个包含四个 Apache-2.0 模型的模型族（0.3B / 0.6B / 1B / 2B），基于
MiniCPM5、Qwen3 和 MNN-Pocket 基座。它提供：

- **Amoeba-DA** —— 验证式投机解码：K 个并行瓶颈草稿头从单一缓存隐藏状态预测后续 token，
  冻结的基座模型在一个批量前向中验证所有候选。仅在 2B / 1B 上发布。
- **工具调用** —— 四个尺寸都经过后训练可发出工具调用，附带 ReAct 推理脚本。

> **研究定位。** 这是一项实证研究，不是通用加速器。核心发现是加速**依赖架构与执行框架**：
> Amoeba-DA 在全注意力的 MiniCPM5-2B 上取得干净的 2.015×，但在滑窗混合架构（Gemma、SparkX）
> 与 MoE/KDA 架构（Ling）上无收益。所有加速比都相对于 Python 瓶颈的 HuggingFace `generate`
> 测量；在 vLLM 下同一模型本身就快得多。论文完整报告了所有撤回、失败与放弃的结果。
> token 级 early-exit 经 16 组实验系统性否定，**没有**发布模型使用它。

## 结果（单卡 RTX 4080S 32GB；HF `generate`，贪心）

| 模型 | 基座 | Amoeba-DA 加速 | 能力 | 工具调用 |
|---|---|---|---|---|
| Amoeba-2B | MiniCPM5-2B | **2.015×**（26.6 → 53.6 tok/s） | 79 题：91.14% → 93.67%，**零倒退** | 4/4 |
| Amoeba-1B | MiniCPM5-1B | **1.30×** | — | 4/4 |
| Amoeba-0.6B | Qwen3-0.6B | 未发布 | — | 3/3 |
| Amoeba-0.3B | MNN-Pocket-0.3B | 未发布 | — | 3/3 |

我们**不**宣称逐位无损：2B 的 token 一致率为 0.734。主张是**下游能力保持**，在 79 道客观题上
以零"对→错"翻转验证（ROUGE-L 0.94）。另一条正交机制——SenseNova-Loop-9B 的动态 loop 深度路由
以 +0.04 nats CE 换取 45% 加速，是独立的质量-速度权衡，单独报告。Qwen 全系加速数字**全部撤回**
（验证引擎与其 KV cache 不兼容）。

## 仓库结构

```
Amoeba-2B/  Amoeba-1B/  Amoeba-0.6B/  Amoeba-0.3B/
  full_model/          # 独立的 HF checkpoint
  README.md            # 模型卡
  LICENSE
  amoeba_da_infer.py   # 草稿头推理（仅 2B/1B）
AMOEBA_PAPER_DRAFT.pdf / .tex / .docx   # 12 页研究报告
MANIFEST.json / DOWNLOAD.md
_LICENSES/                             # 上游许可证
```

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "Amoeba-2B/full_model"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, dtype="auto", device_map="auto")
out = model.generate(**tok("你好", return_tensors="pt").to(model.device),
                     max_new_tokens=128, do_sample=True,
                     temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0], skip_special_tokens=True))
```

贪心解码在长回答上可能循环重复——使用上面的采样参数。工具调用需要工具定义的 system message。

## 论文

`AMOEBA_PAPER_DRAFT.pdf` —— 《Amoeba: From Early Exit to Verified Speculative Decoding
and Dynamic Loop Routing》。章节覆盖被否定的 early-exit 路线、Amoeba-DA 的能力评估与架构边界、
替代机制（更多草稿头、forward-KL、KV 预测）、Loop 路由、vLLM 服务框架验证、完整的负面结果记录，
以及与 Medusa / EAGLE / JetSpec / LCKV / CLA / AttentionPredictor / KV-Direct 的审慎定位。
我们不声称草稿头+验证机制本身的新颖性。

## 许可证

所有 Amoeba 权重和本仓库均为 Apache-2.0。基座模型同样为 Apache-2.0。
没有复制 LayerSkip 代码（CC-BY-NC），其思想从论文独立实现。CCI3 等脏语料只用于已废弃实验，
**不在任何发布 checkpoint 中**。独立研究项目，与基座模型机构无关。

## 引用

```bibtex
@misc{amoeba2026,
  title  = {Amoeba: From Early Exit to Verified Speculative Decoding and Dynamic Loop Routing},
  author = {Weikai Yue and Hy 4 flash},
  year   = {2026},
  note   = {Independent research report}
}
```
