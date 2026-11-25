# -*- coding: utf-8 -*-
"""
Amoeba-DA inference: parallel draft heads + frozen-base batched verification.
Minimal, self-contained reference implementation. The released heads are medusa_chat.pt.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

class DraftHead(nn.Module):
    """Residual bottleneck head: h + W2 SiLU(W1 h), W2 init=0."""
    def __init__(self, H, bn=512):
        super().__init__()
        self.fc1 = nn.Linear(H, bn, bias=False)
        self.fc2 = nn.Linear(bn, H, bias=False)
        nn.init.zeros_(self.fc2.weight)
    def forward(self, h):
        return h + self.fc2(F.silu(self.fc1(h.float()))).to(h.dtype)

class AmoebaDA:
    def __init__(self, model_path, head_path, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.device = device
        H = self.m.config.hidden_size
        sd = torch.load(head_path, map_location=device, weights_only=False)
        self.heads = nn.ModuleList([DraftHead(H).to(device) for _ in range(len(sd))])
        for k, key in enumerate(sd): self.heads[k].load_state_dict(sd[key])
        self.heads.eval()
        self.K = len(self.heads)
        self.eos = self.m.generation_config.eos_token_id
        if not isinstance(self.eos, (list, tuple)): self.eos = [self.eos]

    @torch.no_grad()
    def _fwd(self, ids, cache, total):
        kw = dict(input_ids=ids, use_cache=True, output_hidden_states=True,
                  attention_mask=torch.ones((ids.shape[0], total), device=ids.device,
                                            dtype=torch.long))
        kw["cache"] = cache
        try: return self.m(**kw)
        except TypeError:
            kw.pop("cache"); kw["past_key_values"] = cache; return self.m(**kw)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=96):
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]
        cache = DynamicCache(); plen = ids.shape[1]
        o = self._fwd(ids, cache, plen)
        h = o.hidden_states[-1][:, -1, :]
        cur = int(o.logits[0, -1].argmax())
        seq = []
        for _ in range(max_new_tokens):
            seq.append(cur)
            if cur in self.eos or len(seq) >= max_new_tokens: break
            cand = []
            if self.K > 0:
                hs = torch.stack([hd(h) for hd in self.heads], 0)   # [K,1,H]
                cand = self.m.lm_head(self.m.model.norm(hs)).argmax(-1)[:, 0].tolist()
            vt = torch.tensor([[cur] + cand], device=self.device, dtype=torch.long)
            o = self._fwd(vt, cache, cache.get_seq_length() + vt.shape[1])
            vlog, vh = o.logits[0], o.hidden_states[-1][0]
            n = 1
            for i in range(len(cand)):
                if int(vlog[i].argmax()) == cand[i]: n += 1
                else: break
            for j in range(n - 1):
                seq.append(cand[j])
                if cand[j] in self.eos: break
            remove = vt.shape[1] - n
            if remove > 0:
                try: cache.crop(-remove)
                except Exception: break
            idx = min(n - 1, vlog.shape[0] - 1)
            h = vh[idx:idx + 1]
            cur = int(vlog[idx].argmax())
            if seq and seq[-1] in self.eos: break
        return self.tok.decode(seq, skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Amoeba-2B/full_model")
    ap.add_argument("--heads", default="Amoeba-2B/medusa_chat.pt")
    ap.add_argument("--prompt", default="1+1等于几？")
    a = ap.parse_args()
    m = AmoebaDA(a.model, a.heads)
    print(m.generate(a.prompt))
