# -*- coding: utf-8 -*-
"""BRICK 评估：(1) 79 题能力保持（与 Amoeba 对比）；(2) 速度基准。

用法:
  python eval_brick.py --mode both
  python eval_brick.py --mode speed
  python eval_brick.py --mode cap --questions /data/work/cap_score_2b.json
"""
import sys, os, json, time, argparse, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from brick_model import load_brick
import brick_config as cfg


# ---------------- 速度基准 ----------------
def bench_speed(model, tok, ntok=cfg.BENCH_TOKS, reps=cfg.BENCH_REPS):
    prompt = "请详细解释一下Transformer架构中注意力机制的工作原理。"
    ids = tok(prompt, return_tensors="pt").to(model.base.device)
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=ntok, do_sample=False,
                                 use_cache=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  rep{dt:.2f}s -> {ntok/dt:.1f} tok/s", flush=True)


# ---------------- 能力评估 ----------------
def load_questions(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("questions", d.get("items", list(d.values())))
    out = []
    for x in d:
        if not isinstance(x, dict):
            continue
        q = x.get("question", x.get("q", x.get("prompt", x.get("input"))))
        a = x.get("answer", x.get("a", x.get("expected", x.get("output", x.get("reference")))))
        if q is not None and a is not None:
            out.append((str(q), str(a)))
    return out


def norm(s):
    s = str(s).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def rouge_l(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return float(a == b)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    l = dp[len(a)][len(b)]
    return 2 * l / (len(a) + len(b)) if (len(a) + len(b)) else 1.0


def correct(pred, ref):
    p, r = norm(pred), norm(ref)
    if not r:
        return False
    return r in p or (len(r) <= 12 and r == p[:len(r)])


def eval_capability(model, tok, questions):
    dev = model.base.device
    n_ok = 0
    rows = []
    has_tt = bool(getattr(tok, "chat_template", None))
    for i, (q, ref) in enumerate(questions):
        if has_tt:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        else:
            # MiniCPM5 无内置 chat_template：手工 <|im_start|> 格式（沿用 Amoeba）
            prompt = (f"<|im_start|>system\n你是一个有用的人工智能助手。<|im_end|>\n"
                      f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=cfg.MAX_NEW,
                                 do_sample=True, temperature=cfg.GEN_TEMP,
                                 top_p=cfg.GEN_TOP_P, repetition_penalty=cfg.GEN_REP_PEN,
                                 use_cache=True)
        gen = out[0][ids["input_ids"].shape[1]:]
        pred = tok.decode(gen, skip_special_tokens=True)
        ok = correct(pred, ref)
        n_ok += ok
        rows.append({"q": q[:60], "ref": ref[:40], "pred": pred[:80],
                     "ok": ok, "rouge_l": round(rouge_l(pred, ref), 3)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)} acc={n_ok/(i+1):.3f}", flush=True)
    acc = n_ok / len(questions)
    rl = sum(r["rouge_l"] for r in rows) / len(rows)
    print(f"\n[能力] {n_ok}/{len(questions)} = {acc*100:.2f}%  ROUGE-L {rl:.3f}", flush=True)
    return {"acc": acc, "n_ok": n_ok, "n": len(questions), "rouge_l": rl, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cap", "speed", "both"], default="both")
    ap.add_argument("--questions", default=cfg.QUESTIONS)
    ap.add_argument("--ckpt", default=cfg.FUSION_CKPT)
    a = ap.parse_args()

    print("[加载 BRICK 模型 ...]", flush=True)
    model, tok = load_brick(device="cuda")
    if os.path.exists(a.ckpt):
        model.load_fusion(a.ckpt)
        print(f"  loaded fusion from {a.ckpt}", flush=True)
    else:
        print(f"  WARNING: {a.ckpt} 不存在，使用初始化 fusion（等权平均）", flush=True)
    model.eval()

    if a.mode in ("speed", "both"):
        print("\n=== 速度基准（HF generate，贪心；注意 harness 依赖）===", flush=True)
        bench_speed(model, tok)

    if a.mode in ("cap", "both"):
        print("\n=== 能力评估 ===", flush=True)
        if not os.path.exists(a.questions):
            print(f"题库不存在: {a.questions}；跳过。请指定 --questions", flush=True)
            return
        qs = load_questions(a.questions)
        print(f"题库 {len(qs)} 题", flush=True)
        res = eval_capability(model, tok, qs)
        out = os.path.join(cfg.WORK, "brick_cap_result.json")
        os.makedirs(cfg.WORK, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"saved {out}", flush=True)
        # 与 Amoeba 基线 91.14% 对比
        print(f"\n对比 Amoeba 2B 基线 91.14%: "
              f"BRICK {res['acc']*100:.2f}% (Δ {(res['acc']-0.9114)*100:+.2f}pp)", flush=True)

if __name__ == "__main__":
    main()
