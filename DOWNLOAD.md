# Model weights

Amoeba-DA weights are hosted on HuggingFace (GitHub file-size limits):

- [leexamp/Amoeba-2B](https://huggingface.co/leexamp/Amoeba-2B) — `huggingface-cli download leexamp/Amoeba-2B`
  — 2.015x speedup, 79 QA zero regressions, includes draft heads (`medusa_chat.pt`).
- [leexamp/Amoeba-1B](https://huggingface.co/leexamp/Amoeba-1B) — `huggingface-cli download leexamp/Amoeba-1B`
  — 1.30x speedup, includes draft heads.

The 0.6B and 0.3B models ship **tool-calling only** (no Amoeba-DA acceleration).
Their weights are not published; the inference scripts are in this repository
(`Amoeba-0.6B/`, `Amoeba-0.3B/`) for reproducibility.

The 120M early checkpoint is not a standalone HF model (needs conversion) and is omitted.
