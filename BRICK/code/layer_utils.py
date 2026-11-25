# -*- coding: utf-8 -*-
"""兼容 HF 各版本 LlamaDecoderLayer.forward 签名的层调用辅助。

本机实测 transformers 5.24.1 签名（2026-11-25）：
    forward(hidden_states, attention_mask=None, position_ids=None,
            cache: Cache|None=None, use_cache=False,
            position_embeddings: tuple[cos,sin]|None=None, **kwargs)
关键差异（5.x vs 4.x）：
  - 参数名 `cache` 取代 `past_key_value`
  - attention 需要 `position_embeddings=(cos,sin)`，传 position_ids 会报
    "cannot unpack non-iterable NoneType"（已在本机复现并修复）
本辅助用 model.rotary_emb 预计算 cos/sin，兼容两种签名。
"""
import torch


def rotary_embeddings(model, position_ids):
    """从模型的 rotary_emb 预计算 (cos, sin)。5.x 必须显式传入。"""
    if hasattr(model, "rotary_emb"):
        cos, sin = model.rotary_emb(model.model.embed_tokens.weight.unsqueeze(0), position_ids)
        return cos, sin
    return None


def call_layer(layer, h, position_ids=None, past=None, cache_position=None,
               use_cache=False, attention_mask=None, position_embeddings=None,
               rotary_emb=None):
    """返回 (hidden_states, cache)。兼容 transformers 4.x 和 5.24.x。"""
    kwargs_common = dict(attention_mask=attention_mask, position_ids=position_ids)
    if position_embeddings is not None:
        kwargs_common["position_embeddings"] = position_embeddings

    attempts = [
        # 5.24.x: cache= + position_embeddings=
        lambda: layer(hidden_states=h, cache=past, use_cache=use_cache, **kwargs_common),
        # 5.0-5.2x: past_key_value= (deprecated 但仍接受)
        lambda: layer(hidden_states=h, past_key_value=past, use_cache=use_cache, **kwargs_common),
        # 4.45-4.x: past_key_value= + cache_position=
        lambda: layer(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_value=past, cache_position=cache_position, use_cache=use_cache),
        # 最简回退
        lambda: layer(h, position_ids=position_ids),
    ]
    last = None
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, (tuple, list)):
                return out[0], (out[1] if len(out) > 1 else past)
            return out, past
        except (TypeError, ValueError) as e:
            last = e
    raise last


def precompute_rope(model, position_ids):
    """给整个前向预计算一次 (cos,sin)，供所有层复用（5.x 必需）。"""
    try:
        return model.model.rotary_emb(model.model.embed_tokens.weight.unsqueeze(0), position_ids)
    except Exception:
        return None
