# -*- coding: utf-8 -*-
"""生成蒸馏监督数据：prelude 输出 -> 教师（原始 1D 拓扑）coda 前隐藏状态。

输入是 Amoeba 自生成 chat 语料的 arg（token id 序列），不引入任何外部脏语料。
输出 teacher_data.pt: {prelude_h: [N,SEQ,H], teacher_h: [N,SEQ,H], tokens: [N,SEQ]}
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from layer_utils import call_layer
import brick_config as cfg

def main():
    torch.manual_seed(cfg.SEED)
    tok = AutoTokenizer.from_pretrained(cfg.TOKENIZER)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    config = base.config
    L = config.num_hidden_layers
    assert L == cfg.N_LAYERS, (L, cfg.N_LAYERS)

    c = torch.load(cfg.CORPUS, map_location="cpu")
    arg = c["arg"]                      # [blocks, BLOCK] token id
    nb, block = arg.shape
    split = max(4, int(nb * cfg.TRAIN_SPLIT))
    print(f"corpus blocks {nb} (train {split}, val {nb-split})", flush=True)

    samples = []
    # 每个 block 切成不重叠的窗口
    for b in range(nb):
        for p in range(0, block - cfg.SEQ_LEN, cfg.SEQ_LEN):
            samples.append(arg[b, p:p + cfg.SEQ_LEN])
    nsplit = int(len(samples) * cfg.TRAIN_SPLIT)
    print(f"windows {len(samples)} (train {nsplit}, val {len(samples)-nsplit})", flush=True)

    pre_l, tea_l = [], []
    with torch.no_grad():
        for i, ids in enumerate(samples):
            ids = ids.unsqueeze(0).to("cuda")
            pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
            h = base.model.embed_tokens(ids)
            # prelude 输出
            for li in range(cfg.PRELUDE):
                h, _ = call_layer(base.model.layers[li], h, position_ids=pos)
            pre_l.append(h[0].cpu())
            # 教师：继续顺序跑到 coda 之前
            for li in range(cfg.PRELUDE, L - cfg.CODA):
                h, _ = call_layer(base.model.layers[li], h, position_ids=pos)
            tea_l.append(h[0].cpu())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)}", flush=True)

    data = {
        "prelude_h": torch.stack(pre_l),
        "teacher_h": torch.stack(tea_l),
        "tokens": torch.stack(samples),
        "n_train": nsplit,
    }
    os.makedirs(os.path.dirname(cfg.TEACHER_DATA), exist_ok=True)
    torch.save(data, cfg.TEACHER_DATA)
    print(f"saved {cfg.TEACHER_DATA}: prelude_h {data['prelude_h'].shape}, "
          f"teacher_h {data['teacher_h'].shape}", flush=True)

if __name__ == "__main__":
    main()
