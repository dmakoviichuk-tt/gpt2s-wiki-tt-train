# GPT-2 Small Training on Wikipedia (TT-Train)

This repo trains **GPT-2 small** (124M) on Wikipedia using **Tenstorrent TT-Train** bindings. It features:

- **Idempotent preprocessing** (streaming → memmap shards) — no recompute on reruns
- **Gradient accumulation**
- **Speedrun-style LR scheduler** (linear warmup → optional hold → linear decay) with optional **β1 momentum warmup**
- **`no_grad()`** context manager (Torch-like) for eval & sampling
- **GPT-2 tokenizer** (byte-level BPE), EOS between articles
- **Greedy sampler** & loss curve plot

> Works with *any* Wikipedia language snapshot via `wikimedia/wikipedia` configs (e.g., `20231101.en`, `20231101.de`).

---

## Quick Start

```bash
# 1) Deps
pip install requirements.txt

# 2) TT-Train binding path
tt-metal and tt-train are expected to be prebuilt from the https://github.com/tenstorrent/tt-metal/
export TT_METAL_HOME=/path/to/tt-metal
# Expected: $TT_METAL_HOME/tt-train/build/sources/ttml/_ttml.so

# 3) (Optional) pick Wikipedia snapshot
export WIKI_DUMP=20220301.en  # default in code

# 4) Run
python main.py
```

The first run **streams & tokenizes** Wikipedia, writing shards to `./wiki_memmap_gpt2/` with a `manifest.json`. Subsequent runs **reuse** these files if config is unchanged.

---

## File Layout

- `main.py` — single-file trainer (preprocess → train → eval → sample)
- `wiki_memmap_gpt2/` — generated shards & `manifest.json`
- `wiki_memmap_gpt2/loss_curve.png` — loss plot after training

---

## Configuration (inside `TrainConfig`)

| Field | Meaning | Default |
|---|---|---|
| `seq_len` | GPT-2 context length | 1024 |
| `batch_size` | micro-batch size (per device) | 16 |
| `accumulation_steps` | gradient accumulation steps | 1 |
| `steps` | total **micro-steps** | 5000 |
| `eval_every` | eval cadence (micro-steps) | 200 |
| `lr` | peak LR (`lr_max`) | 3e-4 |
| `min_lr` | final LR at end of decay | 0.0 |
| `warmup_steps` | optimizer-step warmup length | 2000 |
| `hold_steps` | optional flat hold at peak | 0 |
| `total_optim_steps` | total optimizer steps (auto if None) | None |
| `beta1_start/end` | optional β1 warmup range | None |
| `beta1_warmup_steps` | β1 warmup length | 0 |
| `n_layer/n_head/n_embd` | GPT-2s dims | 12 / 12 / 768 |
| `wiki_config` | Wikipedia dump (e.g. `20231101.en`) | env `WIKI_DUMP` or `20220301.en` |
| `max_docs` | cap docs for smoke tests | None |
| `out_dir` | shard output directory | `./wiki_memmap_gpt2` |
| `shard_tokens` | tokens/shard (~128M default) | 134,217,728 |
| `val_fraction` | validation tail fraction | 0.1 |

> **Global batch tokens** = `seq_len * batch_size * accumulation_steps * data_parallel`.

---

## Speedrun Scheduler

Linear **warmup** → optional **hold** → linear **decay** to `min_lr`. Applied **once per optimizer step** (i.e., after each accumulation window). Optional **β1 warmup** (momentum warmup).

Example:

```python
# in TrainConfig
lr = 3e-4
warmup_steps = 3000
hold_steps = 1000
min_lr = 3e-5
beta1_start = 0.85 # NOT SUPPORTED YET
beta1_end   = 0.90 # NOT SUPPORTED YET
beta1_warmup_steps = 3000
```

---

## Gradient Accumulation

Effective/global tokens per optimizer step:
```
tokens_per_step = seq_len * batch_size * accumulation_steps * data_parallel
```
The trainer averages gradients by scaling loss with `1/accumulation_steps`. Optimizer steps occur at the **end** of each accumulation window.

---

## Multilingual Wikipedia

Use any config from `wikimedia/wikipedia`, e.g. `20231101.de` (German), `20231101.uk` (Ukrainian). Set `TrainConfig.wiki_config` or `WIKI_DUMP` env var.

---

## Sampling

At the end of training, `main.py` prints a greedy continuation for:
```
"The history of machine learning"
```

---

## Troubleshooting

- **`_ttml` import error** — check `TT_METAL_HOME` path and that the built wheel/so is at `$TT_METAL_HOME/tt-train/build/sources/ttml/`.
- **OOM** — reduce `batch_size` or `seq_len`, or increase `accumulation_steps`.
- **Slow preprocessing** — set `max_docs` to a few thousand to smoke-test the pipeline.
- **Rebuild shards** — delete `wiki_memmap_gpt2/manifest.json` (or change config).

---

## License

MIT (for this training scaffold). Wikipedia data is under its respective licenses.
