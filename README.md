# Homeostatic Transformer (HoT) — Prototype

A PyTorch prototype of the **Homeostatic Transformer (HoT)**: an encoder-only
sequence classifier whose layers _adaptively_ route computation based on the
information content of the layer's output.

---

## Conceptual Overview

### Core Idea

Standard transformers apply the same expensive self-attention computation at
every layer, regardless of whether the layer "needs" it.  HoT introduces a
**homeostatic** feedback loop: each layer monitors its own output entropy and
uses that signal to select the cheapest path that still handles the current
information complexity.

---

### Components

#### Output Entropy Monitor (OEM)

At every layer, the OEM computes a normalized Shannon entropy over the layer
output.  The representation is averaged over the sequence dimension, passed
through a softmax, and the resulting distribution's entropy is divided by
log(D) to obtain a value in **[0, 1]**:

    H = -Σ p_i log(p_i)  /  log(D)

A value near 0 means the layer output is highly concentrated (low surprise);
a value near 1 means it is nearly uniform (high uncertainty).

---

#### Homeostatic Gate (HG)

The gate compares the OEM value to two per-layer thresholds (H_low, H_high)
and makes a **discrete routing decision for the whole sequence** (not per-token):

| Condition          | Path | Computation               | Complexity |
|--------------------|------|---------------------------|------------|
| H < H_low          | **A** | Residual passthrough      | O(1)       |
| H_low ≤ H ≤ H_high | **B** | Depthwise-sep 1-D conv    | O(N · k)  |
| H > H_high         | **C** | Full multi-head attention | O(N²)      |

**Gradient flow**: during training a soft (sigmoid-weighted) blend of all
paths is used so that gradients flow through the gate signal and through each
path.  During evaluation the gate is applied as a hard discrete decision.

---

#### Pathway Merger (PM)

After the selected path computes its output, the layer applies:

    x_next = LayerNorm(x + α · path_output)

where **α** is a learned scalar per layer (initialized to 0.1).

---

#### Comfort Zone Updater (CZU)

The CZU is a stateful object that manages per-layer thresholds across training:

1. **Warmup** (first W steps, default 1000): forces all layers to use **Path C**
   and collects entropy statistics.
2. **Initialization**: after warmup, sets H_low = P10 and H_high = P90 of the
   warmup entropy distribution for each layer, then **clamps** both values to the
   configured hard bounds (`h_low_clamp`, `h_high_clamp`).  This prevents
   thresholds from saturating near 1.0 even when the warmup entropy distribution
   is narrow.
3. **Schedule** (next `schedule_steps` steps, default 2000): instead of jumping
   instantly to the learned thresholds, `get_all_thresholds()` returns a linear
   interpolation from the initial values toward the clamped learned values.  This
   gives the model time to adapt paths B and A before facing a hard routing
   decision.
4. **Running**: every K steps (default 500) updates thresholds via EMA:

       H_low  ← β · H_low  + (1−β) · P10(recent entropies)
       H_high ← β · H_high + (1−β) · P90(recent entropies)

   After each EMA step the hard bounds are re-applied.

---

#### Routing Regularization

Two optional loss terms are added on top of the task loss **after the warmup
phase** to prevent routing collapse and encourage efficient computation:

| Term | Formula | Effect |
|------|---------|--------|
| **Balance loss** (`lambda_balance`) | KL(target_dist \|\| avg_route_weights) | Pulls the mean routing distribution toward the configured `target_dist`, preventing degenerate collapse to always-C. |
| **Compute-cost penalty** (`lambda_compute`) | mean(w_C across layers) | Directly penalises Path-C usage, incentivising the model to produce lower-entropy representations that route to cheaper paths. |

Both terms flow gradients back through the soft-routing weights all the way to
the model's representations, giving the model a direct signal to adjust its
entropy profile.

---



```
hot/                 # Core package
  __init__.py
  layers.py          # DepthwiseSepConv1d, compute_oem, HoTLayer
  model.py           # HoTEncoder (encoder-only classifier)
  czu.py             # CZU (Comfort Zone Updater)
  data.py            # Synthetic datasets + LRA ListOps fallback

configs/
  hot_synthetic_tiny.yaml   # Tiny config for quick experiments / Colab

train.py             # Training script
eval.py              # Evaluation script
requirements.txt
```

---

## Quick Start

### Install

```bash
pip install torch pyyaml numpy tqdm
# or
pip install -r requirements.txt
```

### Train (synthetic bracket-matching, ~2 min on a T4 GPU)

```bash
python train.py --config configs/hot_synthetic_tiny.yaml
```

### Evaluate a saved checkpoint

```bash
python eval.py runs/<run_dir>/best_model.pt
```

---

## Google Colab

Enable a **GPU runtime** (`Runtime → Change runtime type → T4 GPU`), then run:

```python
# ── Cell 1: clone & install ──────────────────────────────────────────────────
!git clone https://github.com/hairishbabujohn/hot-transformer-prototype.git
%cd hot-transformer-prototype
!pip -q install torch pyyaml numpy tqdm
```

```python
# ── Cell 2: verify GPU ───────────────────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

```python
# ── Cell 3: train ────────────────────────────────────────────────────────────
!python train.py --config configs/hot_synthetic_tiny.yaml
```

```python
# ── Cell 4: evaluate the best checkpoint ────────────────────────────────────
import glob
ckpt = sorted(glob.glob("runs/**/best_model.pt", recursive=True))[-1]
print("Checkpoint:", ckpt)
!python eval.py {ckpt}
```

Expected output (T4 GPU, ~2 minutes):

```
[train] device=cuda
[train] params=76,994
step=100/10000 loss=0.6883 phase=warmup A=0.00% B=0.00% C=100.00% ...
...
  [eval] step=500  val_acc=0.8200 routing A=0.0% B=52.4% C=47.6%
...
  [eval] step=10000 val_acc=0.97xx routing A=xx.x% B=xx.x% C=xx.x%
[done] best_val_acc=0.97xx
```

---

## Config Reference

| Key                              | Default          | Description                                          |
|----------------------------------|------------------|------------------------------------------------------|
| `model.d_model`                  | 64               | Hidden dimension                                     |
| `model.n_layers`                 | 4                | Number of HoT layers                                 |
| `model.n_heads`                  | 4                | Attention heads (Path C)                             |
| `model.conv_kernel_size`         | 7                | Convolution kernel size (Path B)                     |
| `model.gate_temperature`         | 0.05             | Sigmoid temperature for soft routing                 |
| `data.dataset`                   | `synthetic_bracket` | Task name                                         |
| `data.seq_len`                   | 64               | Sequence length                                      |
| `data.batch_size`                | 32               | Mini-batch size                                      |
| `czu.warmup_steps`               | 1000             | Steps forcing Path C during warmup                   |
| `czu.update_every`               | 500              | Steps between EMA threshold updates                  |
| `czu.ema_beta`                   | 0.95             | EMA decay factor                                     |
| `czu.h_low_clamp_min`            | 0.2              | Hard lower bound for H_low                           |
| `czu.h_low_clamp_max`            | 0.75             | Hard upper bound for H_low                           |
| `czu.h_high_clamp_min`           | 0.35             | Hard lower bound for H_high                          |
| `czu.h_high_clamp_max`           | 0.90             | Hard upper bound for H_high (key knob: keep < 1)     |
| `czu.schedule_steps`             | 2000             | Post-warmup steps for gradual threshold transition   |
| `routing_reg.lambda_balance`     | 0.05             | Weight for routing load-balancing KL loss            |
| `routing_reg.lambda_compute`     | 0.02             | Weight for Path-C compute-cost penalty               |
| `routing_reg.target_dist`        | [0.33, 0.34, 0.33] | Target routing distribution [A, B, C]             |
| `training.steps`                 | 10000            | Total training steps                                 |
| `training.lr`                    | 0.001            | AdamW learning rate                                  |
| `training.eval_every`            | 500              | Validation interval                                  |

### Tuning Guide

**If routing still collapses to all-C after warmup:**
- Decrease `czu.h_high_clamp_max` (e.g. 0.85) so H_high stays below the typical
  entropy value, forcing the model to reduce entropy to escape Path C.
- Increase `routing_reg.lambda_compute` (e.g. 0.05) to strengthen the penalty.
- Increase `routing_reg.lambda_balance` (e.g. 0.1) to push harder toward the
  target distribution.

**If val_acc drops too much after warmup:**
- Decrease `routing_reg.lambda_balance` / `lambda_compute` (e.g. halve them).
- Increase `czu.schedule_steps` (e.g. 3000) for a gentler threshold transition.
- Adjust `routing_reg.target_dist` to allow more Path-C usage (e.g.
  `[0.1, 0.3, 0.6]`).

**Expected successful behaviour (synthetic_tiny):**
- During warmup (steps 1–1000): A=0% B=0% C=100%, val_acc rising to ~0.95.
- After warmup (steps 1001–3000, schedule active): gradual mix of paths appears;
  all three paths should each account for > 5% of layer decisions on average.
- After schedule (steps 3000+): stable routing mix, val_acc within ~1–2% of the
  C-only warmup baseline (≥ 0.94).
- Thresholds should stay in the range [0.2, 0.90] and not saturate at ~1.0.

---

## License

MIT — see [LICENSE](LICENSE).