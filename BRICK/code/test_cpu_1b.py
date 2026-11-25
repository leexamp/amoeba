# -*- coding: utf-8 -*-
"""本机真实 MiniCPM5-1B（24层 hidden1536, bf16 CPU）验证 BRICK 代码路径。

内存: 权重 2061MB + 开销，本机自由 4.4GB —— 勉强够，用 bf16。
CPU AVX2 无 bf16 硬件加速，单次前向可能数秒到数十秒。
测试内容：
  1. 加载 1B + 原始模型单次前向（计时）
  2. BRICK 拓扑（8+[8,6]+2）前向能跑
  3. 层调用顺序等价性（branch 串联 = 原始模型）
  4. 短序列 generate（max_new_tokens=2，验证 KV cache 不崩）
"""
import sys, time, gc
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from layer_utils import call_layer, precompute_rope
from brick_model import BrickModel

MODEL = r"E:\AMOEBA_MODELS\MiniCPM5-1B"
SEQ = 16

class OneBConfig:
    N_LAYERS = 24; PRELUDE = 8; CODA = 2; BRANCH_SIZES = (8, 6)
    def branches(self):
        s = self.PRELUDE; out = []
        for n in self.BRANCH_SIZES: out.append(list(range(s, s+n))); s += n
        assert s == self.N_LAYERS - self.CODA
        return out

def report_mem(tag):
    import psutil
    p = psutil.Process()
    print(f"  [{tag}] RSS {p.memory_info().rss/1024**3:.2f}GB "
          f"(系统可用 {psutil.virtual_memory().available/1024**3:.2f}GB)", flush=True)

t0 = time.time()
print("=== 加载 MiniCPM5-1B (bf16, CPU) ===", flush=True)
report_mem("before")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, local_files_only=True).eval()
report_mem("after load")
print(f"  加载耗时 {time.time()-t0:.1f}s", flush=True)

ids = tok("你好，请解释一下注意力机制。", return_tensors="pt",
          add_special_tokens=False)["input_ids"][:, :SEQ]
pos = torch.arange(ids.shape[1]).unsqueeze(0)
rope = precompute_rope(base, pos)

print("\n=== 原始模型前向 (seq=16) ===", flush=True)
t = time.time()
with torch.no_grad():
    out_orig = base(input_ids=ids, use_cache=False).logits
print(f"  耗时 {time.time()-t:.1f}s, logits {tuple(out_orig.shape)}, "
      f"max={out_orig.abs().max().item():.3f}", flush=True)

print("\n=== 顺序调用 24 层 vs 原始模型（等价性）===", flush=True)
t = time.time()
with torch.no_grad():
    h = base.model.embed_tokens(ids)
    for li in range(24):
        h, _ = call_layer(base.model.layers[li], h, position_ids=pos, position_embeddings=rope)
    seq_logits = base.lm_head(base.model.norm(h))
diff = (out_orig - seq_logits).abs().max().item()
print(f"  耗时 {time.time()-t:.1f}s, max diff = {diff:.6f}", flush=True)
assert diff < 1e-2, f"层调用必须等价（bf16 有微小数值误差），实测 {diff}"

print("\n=== BRICK 拓扑 (8+[8,6]+2) 前向 ===", flush=True)
model = BrickModel(base, config=OneBConfig()).eval()
t = time.time()
with torch.no_grad():
    out_brick = model(input_ids=ids, use_cache=False).logits
print(f"  耗时 {time.time()-t:.1f}s, diff vs 原始 = "
      f"{(out_orig-out_brick).abs().max().item():.4f}（2D 重排有差异是设计使然）", flush=True)

print("\n=== 短序列 generate（验证 KV cache）===", flush=True)
t = time.time()
with torch.no_grad():
    gen = model.generate(input_ids=ids[:, :4], max_new_tokens=2,
                         do_sample=False, use_cache=True)
print(f"  耗时 {time.time()-t:.1f}s, generated shape {tuple(gen.shape)}", flush=True)
print(f"  输出 token: {gen[0].tolist()}", flush=True)

report_mem("final")
print("\nALL 1B CPU TESTS PASSED", flush=True)
