#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train GPT-2 small on Wikipedia using GPT-2 tokenizer with TTML bindings.
Enhancements:
 - Streaming -> memmap shards (idempotent)
 - Gradient accumulation
 - no_grad() context manager
 - SpeedrunScheduler: linear warmup -> (optional) hold -> linear decay; optional beta1 momentum warmup

Requirements:
  pip install -U datasets transformers matplotlib

Environment:
  export TT_METAL_HOME=/path/to/tt-metal   (to locate _ttml)
"""

import os, sys, math, random, textwrap, time, json, hashlib, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

# --- Locate TTML bindings ---
if "TT_METAL_HOME" not in os.environ:
    raise RuntimeError("Please set TT_METAL_HOME to your tt-metal repo root (used to locate _ttml).")
sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml  # Tenstorrent TTML Python bindings

# -----------------------------
# Utilities / Repro
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def json_sha1(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

# -----------------------------
# torch.no_grad()-like context manager for TTML
# -----------------------------
class no_grad:
    def __enter__(self):
        self._ctx = _ttml.autograd.AutoContext.get_instance()
        self._prev = self._ctx.get_gradient_mode() if hasattr(self._ctx, "get_gradient_mode") else None
        self._ctx.set_gradient_mode(_ttml.autograd.GradMode.DISABLED)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._prev is not None:
            self._ctx.set_gradient_mode(self._prev)
        else:
            self._ctx.set_gradient_mode(_ttml.autograd.GradMode.ENABLED)
        return False

# -----------------------------
# Gradient Accumulator (Python port)
# -----------------------------
class GradientAccumulator:
    def __init__(self, accumulation_steps: int):
        self.m_accumulation_steps = int(max(1, accumulation_steps))
        self.m_total_loss: float = 0.0
        self.m_total_samples: int = 0
        self.m_steps: int = 0  # micro-steps seen
    def should_zero_grad(self) -> bool:
        return (self.m_steps % self.m_accumulation_steps) == 0
    def should_step(self) -> bool:
        return (self.m_steps % self.m_accumulation_steps) == (self.m_accumulation_steps - 1)
    def scale(self, tensor):
        if self.m_accumulation_steps > 1:
            return _ttml.ops.binary.__mul__(tensor, 1.0 / float(self.m_accumulation_steps))
        return tensor
    def update(self, loss_value: float, samples: int):
        self.m_total_loss += float(loss_value) * float(samples) * float(self.m_accumulation_steps)
        self.m_total_samples += int(samples)
        self.m_steps += 1
    def reset(self):
        self.m_total_loss = 0.0
        self.m_total_samples = 0
        self.m_steps = 0
    def average_loss(self) -> float:
        return (self.m_total_loss / float(self.m_total_samples)) if self.m_total_samples > 0 else 0.0

# -----------------------------
# Speedrun-style LR (and beta1) scheduler
# -----------------------------
@dataclass
class SchedulerConfig:
    lr_max: float = 3e-4
    min_lr: float = 0.0
    warmup_steps: int = 2000
    hold_steps: int = 0
    total_steps: int = 150_000  # full micro-steps or optimizer steps; we apply on optimizer steps
    # optional momentum warmup (beta1 ramp)
    beta1_start: Optional[float] = None  # e.g., 0.85
    beta1_end: Optional[float] = None    # e.g., 0.9
    beta1_warmup_steps: int = 0

class SpeedrunScheduler:
    """Linear warmup -> optional hold -> linear decay; optional beta1 warmup."""
    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        s = step
        w = max(0, self.cfg.warmup_steps)
        h = max(0, self.cfg.hold_steps)
        T = max(1, self.cfg.total_steps)
        peak = self.cfg.lr_max
        min_lr = self.cfg.min_lr

        if s <= w:
            # linear warmup 0 -> lr_max
            return peak * (s / max(1, w))
        elif s <= w + h:
            # hold at lr_max
            return peak
        else:
            # linear decay from lr_max at (w+h) to min_lr at T
            s2 = min(s, T)
            frac = (s2 - (w + h)) / max(1, (T - (w + h)))
            return peak + (min_lr - peak) * frac

    def beta1_at(self, step: int) -> Optional[float]:
        if self.cfg.beta1_start is None or self.cfg.beta1_end is None or self.cfg.beta1_warmup_steps <= 0:
            return None
        s = min(step, self.cfg.beta1_warmup_steps)
        t = s / float(self.cfg.beta1_warmup_steps)
        return (1.0 - t) * self.cfg.beta1_start + t * self.cfg.beta1_end

# Best-effort optimizer param setter (LR and beta1) for TTML AdamW
class OptimParamSetter:
    def __init__(self, optim):
        self.optim = optim
        self._warned_lr = False
        self._warned_beta1 = False

    def set_lr(self, lr: float):
        ok = False
        self.optim.set_lr(float(lr))

    def set_beta1(self, beta1: float):
        raise NotImplementedError("set_beta1 is not implemented in TTML AdamW optimizer.")

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    # Model
    seq_len: int = 1024
    batch_size: int = 8
    accumulation_steps: int = 8  # >1 enables grad accumulation
    steps: int = 5000 * 8           # number of *micro*-steps
    eval_every: int = 200
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2
    seed: int = 42

    # Data
    wiki_config: str = os.environ.get("WIKI_DUMP", "20220301.en")
    max_docs: Optional[int] = None
    tokenizer_name: str = "gpt2"
    eos_between_docs: bool = True

    # Preprocess / storage
    out_dir: str = "./wiki_memmap_gpt2"
    shard_tokens: int = 128 * 1024 * 1024

    # Train/val split
    val_fraction: float = 0.1

    # Scheduler (speedrun-style)
    use_scheduler: bool = True
    warmup_steps: int = 2000
    hold_steps: int = 0
    min_lr: float = 0.0
    total_optim_steps: Optional[int] = None  # if None, infer from steps//accum
    beta1_start: Optional[float] = None      # e.g., 0.85 to warm up momentum
    beta1_end: Optional[float] = None        # e.g., 0.9
    beta1_warmup_steps: int = 0

# -----------------------------
# Preprocessing (streaming -> shards)
# -----------------------------
def preprocess_stream_to_memmap(cfg: TrainConfig) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(cfg.out_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.out_dir, "manifest.json")

    intent = {
        "tokenizer_name": cfg.tokenizer_name,
        "wiki_config": cfg.wiki_config,
        "max_docs": cfg.max_docs,
        "eos_between_docs": cfg.eos_between_docs,
        "shard_tokens": int(cfg.shard_tokens),
        "seq_len": int(cfg.seq_len),
        "val_fraction": float(cfg.val_fraction),
        "version": 1,
    }
    intent_hash = json_sha1(intent)

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        if man.get("intent_hash") == intent_hash and all(os.path.exists(os.path.join(cfg.out_dir, s["filename"])) for s in man.get("shards", [])):
            print(f"[preprocess] Found matching manifest; reusing preprocessed data in {cfg.out_dir}")
            return man
        else:
            print("[preprocess] Manifest mismatch or missing shards; rebuilding.")

    print("[preprocess] Starting streaming tokenization & sharding...")
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)

    ds_iter = load_dataset("wikipedia", cfg.wiki_config, split="train", streaming=True)
    if cfg.max_docs is not None:
        ds_iter = ds_iter.take(cfg.max_docs)

    shard_idx = 0
    written_tokens_total = 0
    shards = []
    buffer = np.empty(cfg.shard_tokens, dtype=np.uint32)
    buf_len = 0

    def flush_shard(buf: np.ndarray, n_tokens: int, sidx: int):
        nonlocal written_tokens_total
        if n_tokens == 0: return None
        filename = f"shard_{sidx:05d}.bin"
        path = os.path.join(cfg.out_dir, filename)
        with open(path, "wb") as f:
            f.write(buf[:n_tokens].tobytes(order="C"))
        shards.append({"filename": filename, "num_tokens": int(n_tokens)})
        written_tokens_total += n_tokens
        print(f"[preprocess] Wrote {filename} | tokens={n_tokens:,} | total={written_tokens_total:,}")
        return path

    docs_seen = 0
    tic = time.time()
    for rec in ds_iter:
        text = rec.get("text", "") or ""
        ids = tok.encode(text, add_special_tokens=False)
        if cfg.eos_between_docs:
            ids = ids + [eos_id] if (len(ids) == 0 or ids[-1] != eos_id) else ids

        i = 0; L = len(ids)
        while i < L:
            can = min(L - i, cfg.shard_tokens - buf_len)
            if can > 0:
                buffer[buf_len:buf_len+can] = np.array(ids[i:i+can], dtype=np.uint32)
                buf_len += can; i += can
            if buf_len == cfg.shard_tokens:
                flush_shard(buffer, buf_len, shard_idx); shard_idx += 1; buf_len = 0

        docs_seen += 1
        if docs_seen % 500 == 0:
            dt = time.time() - tic
            print(f"[preprocess] docs={docs_seen:,} | tokens_written={written_tokens_total:,} | elapsed={dt:.1f}s")

    flush_shard(buffer, buf_len, shard_idx)

    if written_tokens_total == 0:
        raise RuntimeError("No tokens were written. Check dataset/config.")

    val_tokens = int(written_tokens_total * cfg.val_fraction)
    train_tokens = written_tokens_total - val_tokens

    shard_offsets = []; cum = 0
    for s in shards:
        shard_offsets.append(cum); cum += s["num_tokens"]

    manifest = {
        "intent": intent,
        "intent_hash": intent_hash,
        "tokenizer_name": cfg.tokenizer_name,
        "eos_id": int(eos_id),
        "seq_len": int(cfg.seq_len),
        "vocab_size": int(tok.vocab_size),
        "written_tokens_total": int(written_tokens_total),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "split_point": int(train_tokens),
        "shards": shards,
        "shard_offsets": shard_offsets,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return manifest

# -----------------------------
# Corpus loader
# -----------------------------
class MemmapCorpus:
    def __init__(self, out_dir: str, manifest: dict):
        self.out_dir = out_dir
        self.manifest = manifest
        self.seq_len = int(manifest["seq_len"])
        self.vocab_size = int(manifest["vocab_size"])
        self.split_point = int(manifest["split_point"])
        self.total_tokens = int(manifest["written_tokens_total"])
        self.shards = manifest["shards"]
        self.shard_offsets = manifest["shard_offsets"]
        self._mmap_cache: Dict[int, np.memmap] = {}

    def _open_shard(self, i: int) -> np.memmap:
        if i in self._mmap_cache:
            return self._mmap_cache[i]
        path = os.path.join(self.out_dir, self.shards[i]["filename"])
        mm = np.memmap(path, dtype=np.uint32, mode="r")
        self._mmap_cache[i] = mm
        return mm

    def _choose_positions(self, split: str, batch_size: int) -> List[Tuple[int, int]]:
        block = self.seq_len + 1
        if split == "train":
            range_start, range_end = 0, self.split_point
        else:
            range_start, range_end = self.split_point, self.total_tokens

        per_shard = []; total_windows = 0
        for i, s in enumerate(self.shards):
            off = self.shard_offsets[i]; n = int(s["num_tokens"])
            a = max(off, range_start); b = min(off + n, range_end)
            usable = max(0, b - a); windows = max(0, usable - block)
            per_shard.append((i, off, n, windows))
            total_windows += max(0, windows)
        if total_windows <= 0:
            raise RuntimeError(f"No windows for split={split}.")

        choices = []
        for i, off, n, windows in per_shard:
            if windows > 0:
                choices.extend([i] * max(1, windows // max(1, block)))
        if not choices:
            choices = [i for i, _, _, w in per_shard if w > 0]

        positions = []
        for _ in range(batch_size):
            si = random.choice(choices)
            off = self.shard_offsets[si]; n = int(self.shards[si]["num_tokens"])
            gs = max(off, range_start); ge = min(off + n, range_end) - block
            if ge < gs: continue
            gstart = random.randint(gs, ge)
            start_in_shard = gstart - off
            positions.append((si, start_in_shard))

        if len(positions) < batch_size:
            while len(positions) < batch_size:
                positions.extend(positions[: max(1, batch_size - len(positions))])
            positions = positions[:batch_size]
        return positions

    def get_batch(self, split: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        block = self.seq_len + 1
        pos = self._choose_positions(split, batch_size)
        X = np.empty((batch_size, self.seq_len), dtype=np.uint32)
        Y = np.empty((batch_size, self.seq_len), dtype=np.uint32)
        for b, (si, st) in enumerate(pos):
            mm = self._open_shard(si)
            window = np.asarray(mm[st:st+block])
            X[b, :] = window[:-1]; Y[b, :] = window[1:]
        return X, Y

# -----------------------------
# Model / Optim
# -----------------------------
def create_model_and_optim(cfg: TrainConfig, vocab_size: int):
    gcfg = _ttml.models.gpt2.GPT2TransformerConfig()
    gcfg.num_heads = cfg.n_head
    gcfg.embedding_dim = cfg.n_embd
    gcfg.num_blocks = cfg.n_layer
    gcfg.vocab_size = int((vocab_size + 31) // 32 * 32)
    gcfg.max_sequence_length = cfg.seq_len
    gcfg.dropout_prob = cfg.dropout

    model = _ttml.models.gpt2.create_gpt2_model(gcfg)

    adamw_cfg = _ttml.optimizers.AdamWConfig.make(
        float(cfg.lr),
        float(cfg.beta1),
        float(cfg.beta2),
        float(cfg.eps),
        float(cfg.weight_decay),
    )
    optim = _ttml.optimizers.AdamW(model.parameters(), adamw_cfg)
    return model, optim

# -----------------------------
# Train / Eval
# -----------------------------
def evaluate(model, corpus: MemmapCorpus, cfg: TrainConfig, tt_mask):
    with no_grad():
        model.eval()
        x_u32, y_u32 = corpus.get_batch("val", cfg.batch_size)
        vtt_x = _ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(cfg.batch_size, 1, 1, cfg.seq_len),
            _ttml.Layout.ROW_MAJOR,
            _ttml.autograd.DataType.UINT32
        )
        vtt_y = _ttml.autograd.Tensor.from_numpy(
            y_u32,
            _ttml.Layout.ROW_MAJOR,
            _ttml.autograd.DataType.UINT32
        )
        logits = model(vtt_x, tt_mask)
        loss = _ttml.ops.loss.cross_entropy_loss(logits, vtt_y, _ttml.ops.ReduceType.MEAN)
        val_loss = float(loss.to_numpy())
        _ttml.autograd.AutoContext.get_instance().reset_graph()
    model.train()
    return val_loss

def build_causal_mask(T: int) -> np.ndarray:
    m = np.tril(np.ones((T, T), dtype=np.float32))
    return m.reshape(1, 1, T, T)

def train(cfg: TrainConfig, model, optim, corpus: MemmapCorpus, vocab_size: int):
    set_seed(cfg.seed)
    loss_fn = _ttml.ops.loss.cross_entropy_loss
    reduce = _ttml.ops.ReduceType.MEAN

    tt_mask = _ttml.autograd.Tensor.from_numpy(
        build_causal_mask(cfg.seq_len),
        _ttml.Layout.TILE,
        _ttml.autograd.DataType.BFLOAT16
    )

    # Scheduler setup (applied on optimizer steps only)
    if cfg.use_scheduler:
        total_optim_steps = cfg.total_optim_steps
        if total_optim_steps is None:
            total_optim_steps = max(1, cfg.steps // max(1, cfg.accumulation_steps))
        sched = SpeedrunScheduler(SchedulerConfig(
            lr_max=cfg.lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            hold_steps=cfg.hold_steps,
            total_steps=total_optim_steps,
            beta1_start=cfg.beta1_start,
            beta1_end=cfg.beta1_end,
            beta1_warmup_steps=cfg.beta1_warmup_steps,
        ))
        setter = OptimParamSetter(optim)
    else:
        sched = None
        setter = None

    model.train()
    train_losses: List[float] = []
    val_losses: List[float] = []

    accum = GradientAccumulator(cfg.accumulation_steps)
    tokens_per_batch = cfg.batch_size * cfg.seq_len
    optim_steps_done = 0

    t0 = time.time()
    for step in range(0, cfg.steps):
        if accum.should_zero_grad():
            # Apply scheduler at the START of each optimizer step window
            if sched is not None:
                lr_now = sched.lr_at(optim_steps_done)
                setter.set_lr(lr_now)
                beta1_now = sched.beta1_at(optim_steps_done)
                if beta1_now is not None:
                    setter.set_beta1(beta1_now)
            optim.zero_grad()

        x_u32, y_u32 = corpus.get_batch("train", cfg.batch_size)
        tt_x = _ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(cfg.batch_size, 1, 1, cfg.seq_len),
            _ttml.Layout.ROW_MAJOR,
            _ttml.autograd.DataType.UINT32
        )
        tt_y = _ttml.autograd.Tensor.from_numpy(
            y_u32,
            _ttml.Layout.ROW_MAJOR,
            _ttml.autograd.DataType.UINT32
        )

        logits = model(tt_x, tt_mask)
        loss = loss_fn(logits, tt_y, reduce)

        scaled_loss = accum.scale(loss)
        scaled_loss.backward(False)
        _ttml.autograd.AutoContext.get_instance().reset_graph()

        tr_loss = float(loss.to_numpy())
        accum.update(tr_loss, tokens_per_batch)

        if accum.should_step():
            optim.step()
            optim_steps_done += 1

        train_losses.append(tr_loss)

        if ((step + 1) % cfg.eval_every) == 0 or step == 0:
            val_loss = evaluate(model, corpus, cfg, tt_mask)
            val_losses.append(val_loss)
            dt = time.time() - t0
            eff_step = optim_steps_done
            info = f"lr {sched.lr_at(eff_step):.6g}" if sched is not None else ""
            print(f"step {step+1:5d} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | eff_step {eff_step} {info} | elapsed {dt:.1f}s")

    return train_losses, val_losses

# -----------------------------
# Sampling
# -----------------------------
def sample_greedy(model, corpus: MemmapCorpus, start_ids: np.ndarray, max_new_tokens: int, seq_len: int, vocab_size: int):
    with no_grad():
        model.eval()
        seed_x, _ = corpus.get_batch("val", 1)
        running = seed_x[0].tolist()
        tail = start_ids.tolist()
        tail = tail[-min(len(tail), seq_len):]
        running[-len(tail):] = tail

        for _ in range(max_new_tokens):
            inp = np.array(running[-seq_len:], dtype=np.uint32).reshape(1, 1, 1, seq_len)
            tt_inp = _ttml.autograd.Tensor.from_numpy(inp)
            tt_mask = _ttml.autograd.Tensor.from_numpy(build_causal_mask(seq_len))

            logits = model(tt_inp, tt_mask)
            np_logits = logits.to_numpy().reshape(1, -1, vocab_size)[:, -1, :]
            next_id = int(np.argmax(np_logits, axis=-1)[0])
            running.append(next_id)

            _ttml.autograd.AutoContext.get_instance().reset_graph()

        return np.array(running[-max_new_tokens:], dtype=np.uint32)

# -----------------------------
# Main
# -----------------------------
def main():
    from transformers import AutoTokenizer

    cfg = TrainConfig()
    set_seed(cfg.seed)

    manifest = preprocess_stream_to_memmap(cfg)
    vocab_size = int(manifest["vocab_size"])
    corpus = MemmapCorpus(cfg.out_dir, manifest)

    model, optim = create_model_and_optim(cfg, vocab_size)

    train_losses, val_losses = train(cfg, model, optim, corpus, vocab_size)

    # Plot losses
    try:
        plt.figure()
        plt.plot(np.arange(1, len(train_losses)+1), train_losses, label="train")
        if len(val_losses) > 0:
            x_val = np.linspace(1, len(train_losses), num=len(val_losses))
            plt.plot(x_val, val_losses, marker="o", linestyle="--", label="val")
        plt.xlabel("Step"); plt.ylabel("Cross-Entropy Loss")
        plt.title("GPT-2 small on Wikipedia (TTML, memmap, accum, speedrun scheduler)")
        plt.legend(); plt.tight_layout()
        out_png = os.path.join(cfg.out_dir, "loss_curve.png")
        plt.savefig(out_png, dpi=150)
        print(f"Saved loss curve to {out_png}")
    except Exception as e:
        print(f"Plotting failed (continuing): {e}")

    # Sample
    try:
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        prompt = "The history of machine learning"
        start_ids = np.array(tok.encode(prompt, add_special_tokens=False), dtype=np.uint32)
        gen_tail = sample_greedy(model, corpus, start_ids, max_new_tokens=200, seq_len=cfg.seq_len, vocab_size=vocab_size)
        print("\nGenerated (greedy):\n")
        print(textwrap.fill(tok.decode(gen_tail.tolist()), width=100))
    except Exception as e:
        print(f"Sampling failed (continuing): {e}")

if __name__ == "__main__":
    main()
