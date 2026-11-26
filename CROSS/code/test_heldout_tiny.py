# -*- coding: utf-8 -*-
"""Tiny smoke test for probability-level fusion and force-P0."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from cross_model import CROSSModel
import cross_config as cfg

torch.manual_seed(0)
conf=LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=12,
                 num_attention_heads=4,num_key_value_heads=2,vocab_size=1000,
                 max_position_embeddings=128,use_cache=True)
base=LlamaForCausalLM(conf).eval()
progs=cfg.default_programs(12,k=4)

for fl in ["hidden","prob"]:
    m=CROSSModel(base,programs=progs,top_k=2,gumbel=False,fusion_level=fl).eval()
    ids=torch.randint(0,1000,(2,16))
    o=m(input_ids=ids,labels=ids)
    print(f"{fl}: logits {tuple(o.logits.shape)} loss {o.loss.item():.4f}")
    o.loss.backward()
    print(f"  grads {sum(1 for p in m.trainable_parameters() if p.grad is not None)}")

# force P0
m2=CROSSModel(base,programs=progs,top_k=2,gumbel=False,
              select_before_execute=True,selector_top_k=2,force_program_0=True).eval()
o=m2(input_ids=ids)
print(f"force-P0: logits {tuple(o.logits.shape)} OK")
print("TINY HELDOUT SMOKE PASSED")
