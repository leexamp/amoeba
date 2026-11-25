# Amoeba 模型族 License 说明（HuggingFace 发布用）

> 本文件随 Amoeba 模型权重一同发布，说明每个版本的基座来源、
> 所继承的开源许可证，以及发布时需要遵守的署名与声明义务。
> 配套全文见同目录下的 `APACHE-2.0.txt`（`MIT.txt` 为标准模板备用，当前各版本均不适用）。

## 1. 各版本继承的 base license

| Amoeba 版本 | 基座（base model） | 基座团队 | 继承的 License |
|---|---|---|---|
| Amoeba-2B | MiniCPM5-2B（OpenBMB） | OpenBMB | Apache-2.0 |
| Amoeba-1B | MiniCPM5-1B（OpenBMB） | OpenBMB | Apache-2.0 |
| Amoeba-0.6B | Qwen3-0.6B（Qwen team） | Qwen / Alibaba | Apache-2.0 |
| Amoeba-0.3B | MNN-Pocket-0.3B（MNN team） | MNN Team | Apache-2.0 |
| Amoeba-120M | 自研（Amoeba 团队从头训练） | Amoeba | Apache-2.0 |

### 1.1 Amoeba 自身的改造部分

- **halt head（逐 token 动态激活）+ 训练方法（熵正则化等）**：由 Amoeba
  团队原创实现，采用 **Apache-2.0** 发布。
- 因此所有基座版本（含 Amoeba-0.3B，其基座 MNN-Pocket-0.3B 同样为
  Apache-2.0）的衍生权重整体均沿用 **Apache-2.0**。

## 2. 参考但未复用代码的部分

- **LayerSkip（facebookresearch）**：其代码为 **CC-BY-NC（仅非商用）**。
  本项目**仅参考论文思想**（layer dropout + exit loss 的思路），
  **未复制、未改写其任何代码**。CC-BY-NC 代码未进入本仓库，
  论文按正常学术引用（citation）署名，不受代码许可证约束。
- **CALM（google-deepmind）**：Apache-2.0，可作为实现参考；如复用其
  代码片段，保留原 Apache-2.0 版权头。

## 3. 数据集

- **OpenThoughts agent 数据**（如使用，open-thoughts/OpenThoughts-Agent-SFT-100K）：
  **Apache-2.0**，按 Apache-2.0 要求保留署名与 NOTICE 即可。
- 其它训练数据（如 CCI3-Data、OpenMathReasoning 等）请对照各数据卡
  单独署名；CC-BY-4.0 数据（如 nvidia/OpenMathReasoning）须在 model card
  致谢中写明署名。

## 4. 版权归属

- **各基座模型的版权归原作者/原团队所有**
  （OpenBMB、Qwen/Alibaba、MNN Team 等）。
- 本仓库**只发布在基座之上的增量改造**（halt head、训练方法、衍生权重），
  不声称拥有基座本身的版权，也不替换或删除基座自带的版权声明。

## 5. 发布到 HuggingFace 时的 attribution 要求

1. 在 model card frontmatter 中声明 `license`（所有版本均为 apache-2.0），
   并用 `base_model:` 写明对应基座
   （MiniCPM5-2B / MiniCPM5-1B / Qwen3-0.6B / MNN-Pocket-0.3B）。
2. 正文明确声明本模型是基座模型的衍生工作，例如：
   *"This model is a derivative work of [base model] with added token-level
   dynamic activation (halt heads). This is an independent research project,
   not affiliated with OpenBMB / Alibaba / Taobao / MNN Team."*
3. 保留基座权重/代码中的原始版权头与 NOTICE 声明，不删改。
4. 须随附 `APACHE-2.0.txt` 全文；`MIT.txt` 仅作标准模板备用，当前发布版本均不适用。
5. 引用各基座论文与相关参考文献（CALM、LayerSkip 论文仅作思想引用）。
6. 遵守 Apache-2.0 第 6 条：不使用基座团队商标或暗示官方背书的名字
   （如不要命名 "MiniCPM5-Amoeba-Official"）。
7. 对使用到的 CC-BY-4.0 数据在 model card 致谢中署名。

---

简言之：**基座版权归原作者，Amoeba 仅发布增量改造；按上表继承对应
license，并在 HuggingFace model card 中完成 base_model / 修改声明 /
非官方背书 / 数据署名即可合规发布。**
