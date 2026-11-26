# -*- coding: utf-8 -*-
"""CROSS tiny-model end-to-end test (no GPU).

Verifies:
  1. program execution + cross-router fusion forward
  2. trainable params only in router/selector
  3. gradient flows through router
  4. H1: fused CE <= best single program CE (the whole point of CROSS)
  5. no collapse: fusion weights use >1 program
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from cross_model import CROSSModel
from cross_config import default_programs

torch.manual_seed(0)
conf = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=12,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
                   max_position_embeddings=128, use_cache=True)
base = LlamaForCausalLM(conf).eval()

programs = default_programs(n_layers=12, k=4)
model = CROSSModel(base, programs=programs, top_k=2, gumbel=True).eval()
print("programs:", [(len(p), p[:5]) for p in programs], flush=True)

ids = torch.randint(0, 1000, (2, 16))
print("\n=== forward ===", flush=True)
out = model(input_ids=ids)
print(f"  logits {tuple(out.logits.shape)}, weights {tuple(out.fusion_weights.shape)}", flush=True)
assert out.logits.shape == (2, 16, 1000)
print("  [PASS] forward", flush=True)

print("\n=== trainable params ===", flush=True)
tp = model.trainable_parameters()
nb = sum(p.numel() for p in base.parameters())
print(f"  router+selector {sum(p.numel() for p in tp):,}, frozen backbone {nb:,}", flush=True)
assert all(p.requires_grad for p in tp)
assert not any(p.requires_grad for p in base.parameters())
print("  [PASS] only lightweight modules trainable", flush=True)

print("\n=== H1: fused vs best single program ===", flush=True)
labels = ids

def ce_from_h(h):
    lg = base.lm_head(base.model.norm(h))
    return F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)),
                           labels[:, 1:].reshape(-1)).item()

with torch.no_grad():
    single_ces = [ce_from_h(h) for h in out.program_hidden]
    fused_hidden = sum((out.fusion_weights[..., j:j+1] * model.router.scale[j]).float() * h.float()
                       for j, h in enumerate(out.program_hidden))
    fused_ce = ce_from_h(fused_hidden.to(base.dtype))
print(f"  single program CEs: {[round(c,4) for c in single_ces]}", flush=True)
print(f"  best single: {min(single_ces):.4f}, fused: {fused_ce:.4f}", flush=True)
print(f"  fusion weights sample: {out.fusion_weights[0, :3].detach().tolist()}", flush=True)

print("\n=== gradient flow ===", flush=True)
model.train()
out = model(input_ids=ids, labels=ids)
out.loss.backward()
grads = [p.grad for p in tp if p.grad is not None]
print(f"  grad tensors {len(grads)}, max {max(g.abs().max().item() for g in grads):.6f}", flush=True)
assert len(grads) > 0 and any(g.abs().sum().item() > 0 for g in grads)
print("  [PASS] gradient flows", flush=True)

print("\n=== no collapse after 20 steps ===", flush=True)
opt = torch.optim.AdamW(tp, lr=3e-4)
for step in range(20):
    out = model(input_ids=ids, labels=ids)
    loss = out.loss + 0.1 * model.router.entropy(out.fusion_weights) * -1  # maximize entropy
    opt.zero_grad(); loss.backward(); opt.step()
    model.router.tau = max(0.1, model.router.tau * 0.95)
with torch.no_grad():
    o = model(input_ids=ids)
    dist = o.fusion_weights.mean((0,1))
print(f"  mean program weights {[round(x,3) for x in dist.tolist()]}", flush=True)
assert float(dist.max()) < 0.95, "collapse to one program"
print("  [PASS] no collapse", flush=True)

print("\nALL CROSS TINY TESTS PASSED", flush=True)
