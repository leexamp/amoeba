# -*- coding: utf-8 -*-
"""CROSS wrapper: execute K programs on a frozen model, fuse with CrossRouter."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from layer_utils import call_layer, precompute_rope
from cross_router import CrossRouter
from cross_selector import ProgramSelector
import cross_config as cfg


class CROSSModel(nn.Module):
    def __init__(self, base, programs=None, top_k=cfg.TOP_K, gumbel=True,
                 select_before_execute=False, selector_top_k=None, fusion_level="hidden",
                 force_program_0=False):
        super().__init__()
        self.base = base
        self.layers = base.model.layers
        self.norm = base.model.norm
        self.lm_head = base.lm_head
        self.embed = base.model.embed_tokens
        self.n_layers = base.config.num_hidden_layers
        self.programs = programs or cfg.default_programs(self.n_layers)
        self.k = len(self.programs)
        self.top_k = top_k
        self.hidden = base.config.hidden_size

        # select_before_execute: selector predicts programs BEFORE running them,
        # so only selector_top_k programs are executed (real compute saving).
        self.select_before_execute = select_before_execute
        self.selector_top_k = selector_top_k or top_k
        # force P0 (identity, the best single program) to always be selected
        self.force_program_0 = force_program_0

        self.router = CrossRouter(self.hidden, self.k, top_k=top_k, gumbel=gumbel,
                                  fusion_level=fusion_level)
        self.selector = ProgramSelector(self.hidden, self.k)

        # Keep trainable modules in the same dtype as the frozen backbone (bf16 on real 1B).
        # A fresh nn.Module defaults to fp32; loading a bf16 state_dict can leave mixed
        # dtypes and cause "mat1 and mat2 must have the same dtype" at runtime.
        self.router = self.router.to(base.dtype)
        self.selector = self.selector.to(base.dtype)

        for p in base.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        return list(self.router.parameters()) + list(self.selector.parameters())

    def _execute_program(self, idx, h0, pos, rope):
        h = h0
        for li in self.programs[idx]:
            h, _ = call_layer(self.layers[li], h, position_ids=pos, position_embeddings=rope)
        return h

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                attention_mask=None, use_cache=False, labels=None, **kwargs):
        h0 = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(h0.shape[1], device=h0.device).unsqueeze(0).expand(h0.shape[0], -1)
        rope = precompute_rope(self.base, position_ids)

        if self.select_before_execute:
            # ★ SELECT BEFORE EXECUTE: only run the selected programs (real compute saving).
            # Selector sees the cheap prelude hidden (first 4 layers), then picks top-k.
            sel_h = h0
            for li in range(min(4, self.n_layers)):
                sel_h, _ = call_layer(self.layers[li], sel_h, position_ids=position_ids,
                                      position_embeddings=rope)
            sel_logits = self.selector(sel_h.mean(dim=1))    # [B,K]
            n_pick = self.selector_top_k
            if self.force_program_0:
                # Always keep P0 (identity = best single program), choose top-(k-1) from the rest
                sel_logits[:, 0] = float("inf")              # force P0
                selected = torch.topk(sel_logits, n_pick, dim=-1).indices
            else:
                selected = torch.topk(sel_logits, n_pick, dim=-1).indices
            # For simplicity use the first batch item's selection for the whole batch
            idxs = selected[0].tolist()
            outs = [self._execute_program(j, h0, position_ids, rope) for j in idxs]
            # pad to k with a zero-weight placeholder for the router (which expects k inputs)
            while len(outs) < self.k:
                outs.append(outs[0] * 0)
            fused, w, logits_router = self.router(outs)
        else:
            outs = [self._execute_program(j, h0, position_ids, rope) for j in range(self.k)]
            fused, w, logits_router = self.router(outs)     # ★ the CROSS fusion
        if self.router.fusion_level == "prob":
            # Jensen guarantee: CE of the averaged distribution <= weighted average of member CEs.
            # fuse the per-program next-token distributions, not the hidden states.
            per_prog_logits = [self.lm_head(self.norm(h)) for h in outs]
            probs = torch.stack([torch.softmax(lg.float(), dim=-1) for lg in per_prog_logits], dim=0)
            # w: [B,T,K]; probs: [K,B,T,V]
            wf = w.permute(2, 0, 1).unsqueeze(-1).float()    # [K,B,T,1]
            logits = torch.log((probs * wf).sum(dim=0) + 1e-10).to(fused.dtype)
        else:
            logits = self.lm_head(self.norm(fused))

        out = CausalLMOutputWithPast(logits=logits, past_key_values=None)
        out.fusion_weights = w                              # for diagnostics
        out.program_hidden = outs
        if labels is not None:
            out.loss = F.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.size(-1)),
                                       labels[..., 1:].reshape(-1))
        return out

    def generate(self, **kwargs):
        old = self.base.forward
        self.base.forward = self.forward
        try:
            return self.base.generate(**kwargs)
        finally:
            self.base.forward = old
