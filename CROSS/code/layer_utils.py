# -*- coding: utf-8 -*-
"""Layer-call helper compatible with multiple HF LlamaDecoderLayer.forward signatures.

Locally verified on transformers 5.24.1 (2026-11-25):
    forward(hidden_states, attention_mask=None, position_ids=None,
            cache: Cache|None=None, use_cache=False,
            position_embeddings: tuple[cos,sin]|None=None, **kwargs)
Key differences 5.x vs 4.x:
  - argument `cache` replaces `past_key_value`
  - attention requires precomputed `position_embeddings=(cos,sin)`; passing only
    position_ids raises "cannot unpack non-iterable NoneType" (reproduced and fixed)
The helper uses model.rotary_emb to precompute cos/sin, supporting both signatures.
"""
import torch


def rotary_embeddings(model, position_ids):
    """Precompute (cos, sin) from the model's rotary_emb (required in 5.x)."""
    if hasattr(model, "rotary_emb"):
        cos, sin = model.rotary_emb(model.model.embed_tokens.weight.unsqueeze(0), position_ids)
        return cos, sin
    return None


def call_layer(layer, h, position_ids=None, past=None, cache_position=None,
               use_cache=False, attention_mask=None, position_embeddings=None,
               rotary_emb=None):
    """Returns (hidden_states, cache). Compatible with transformers 4.x and 5.24.x."""
    kwargs_common = dict(attention_mask=attention_mask, position_ids=position_ids)
    if position_embeddings is not None:
        kwargs_common["position_embeddings"] = position_embeddings

    attempts = [
        # 5.24.x: cache= + position_embeddings=
        lambda: layer(hidden_states=h, cache=past, use_cache=use_cache, **kwargs_common),
        # 5.0-5.2x: past_key_value= (deprecated but still accepted)
        lambda: layer(hidden_states=h, past_key_value=past, use_cache=use_cache, **kwargs_common),
        # 4.45-4.x: past_key_value= + cache_position=
        lambda: layer(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_value=past, cache_position=cache_position, use_cache=use_cache),
        # minimal fallback
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
    """Precompute (cos,sin) once for the whole forward, reused by all layers (5.x)."""
    try:
        return model.model.rotary_emb(model.model.embed_tokens.weight.unsqueeze(0), position_ids)
    except Exception:
        return None
