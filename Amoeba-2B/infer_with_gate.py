# -*- coding: utf-8 -*-
"""
amoeba_infer_with_gate.py — Amoeba 动态激活推理（带质量门，推荐用法）

关键：原始 halt head 会坍缩到浅层（L2-L6）导致乱码。质量门（conf_min）
强制只有"该层对下一 token 的置信度 >= conf_min"才允许退出，把计算推到
质量足够的深层，从而恢复生成质量。

用法:
  python amoeba_infer_with_gate.py --model-path <full_model目录> \
      --ckpt <halt_heads.pt> --conf-min 0.85 --prompt "你好"
  # conf_min 越大越保守（质量越好、加速越少）；推荐 0.8~0.95
  # conf_min=1.0 等价全层（无早退）；conf_min=0.0 等价无门（会坍缩，勿用）
"""
import sys, argparse, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--conf-min", type=float, default=0.85, help="质量门：每层置信度下限，越大越保守")
ap.add_argument("--threshold", type=float, default=0.9, help="halt 累积退出概率阈值")
ap.add_argument("--prompt", default="你好")
ap.add_argument("--max-new", type=int, default=96)
ap.add_argument("--temperature", type=float, default=0.7)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model_path)
base = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map=device).eval()
base.config.use_cache = False
n_layer = base.config.num_hidden_layers
hidden_size = base.config.hidden_size

# halt heads（与训练一致：Linear(hidden+1,1)）
halt_heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size + 1, 1) for _ in range(n_layer)])
if args.ckpt:
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("halt_heads", obj)
    cleaned = {k.replace("halt_heads.", ""): v for k, v in sd.items()}
    halt_heads.load_state_dict(cleaned, strict=True)
    print(f"[load] halt heads from {args.ckpt}", flush=True)
halt_heads = halt_heads.to(device).eval()


def unc_of(logits_1):
    p = F.softmax(logits_1.float(), dim=-1)
    pmax = p.max(dim=-1).values.clamp_min(1e-8)
    return torch.tanh(-torch.log(pmax))


def call_layer(layer, hidden, **kw):
    import inspect
    sig = inspect.signature(layer.forward)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        out = layer(hidden, **kw)
    else:
        out = layer(hidden, {k: v for k, v in kw.items() if k in sig.parameters})
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def generate(prompt):
    ctx = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(ctx, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
    exits = []
    for _ in range(args.max_new):
        T = ids.shape[1]
        emb = base.model.embed_tokens(ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pe = base.model.rotary_emb(emb, pos)
        common = dict(attention_mask=None, position_ids=pos, past_key_value=None,
                      past_key_values=None, output_attentions=False, use_cache=False,
                      cache_position=torch.arange(T, device=device), position_embeddings=pe)
        h = emb
        hiddens = []
        for layer in base.model.layers:
            h = call_layer(layer, h, **common)
            hiddens.append(h)
        remaining = torch.ones(1, device=device)
        chosen = None
        used = n_layer
        for i in range(n_layer):
            hl = hiddens[i][:, -1, :]
            logits = base.lm_head(base.model.norm(hl.unsqueeze(1)))[:, 0, :]
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values  # [1]
            p = torch.sigmoid(halt_heads[i](torch.cat([hl, unc_of(logits).unsqueeze(-1)], dim=-1)).squeeze(-1))
            remaining = remaining * (1 - p)
            quality_ok = (conf >= args.conf_min) | (i >= n_layer - 2)
            if ((1 - remaining >= args.threshold) & quality_ok).all():
                chosen = logits; used = i + 1; break
        if chosen is None:
            chosen = base.lm_head(base.model.norm(hiddens[-1][:, -1, :].unsqueeze(1)))[:, 0, :]
            used = n_layer
        if args.temperature <= 0:
            nxt = chosen.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(chosen / args.temperature, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        exits.append(used)
        if nxt.item() in (1, 151645, 65512, 2):
            break
    reply = tok.decode(ids[0][ids.shape[1]-0:].tolist(), skip_special_tokens=True) if False else tok.decode(ids[0][len(tok(ctx, add_special_tokens=False)['input_ids']):].tolist(), skip_special_tokens=True)
    return reply, exits


print(f"[cfg] conf_min={args.conf_min} threshold={args.threshold} layers={n_layer}", flush=True)
reply, exits = generate(args.prompt)
print(f"用户: {args.prompt}\n助手: {reply}")
print(f"[退出层均值 {sum(exits)/len(exits):.1f}/{n_layer} | 生成 {len(exits)} token]", flush=True)
