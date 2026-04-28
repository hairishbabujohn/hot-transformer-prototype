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
   warmup entropy distribution for each layer.
3. **Running**: every K steps (default 500) updates thresholds via EMA:

       H_low  ← β · H_low  + (1−β) · P10(recent entropies)
       H_high ← β · H_high + (1−β) · P90(recent entropies)

---

## Repository Structure

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

| Key                         | Default | Description                                    |
|-----------------------------|---------|------------------------------------------------|
| `model.d_model`             | 64      | Hidden dimension                               |
| `model.n_layers`            | 4       | Number of HoT layers                          |
| `model.n_heads`             | 4       | Attention heads (Path C)                       |
| `model.conv_kernel_size`    | 7       | Convolution kernel size (Path B)               |
| `model.gate_temperature`    | 0.05    | Sigmoid temperature for soft routing           |
| `data.dataset`              | `synthetic_bracket` | Task name                        |
| `data.seq_len`              | 64      | Sequence length                                |
| `data.batch_size`           | 32      | Mini-batch size                                |
| `czu.warmup_steps`          | 1000    | Steps forcing Path C during warmup             |
| `czu.update_every`          | 500     | Steps between EMA threshold updates            |
| `czu.ema_beta`              | 0.95    | EMA decay factor                               |
| `training.steps`            | 10000   | Total training steps                           |
| `training.lr`               | 0.001   | AdamW learning rate                            |
| `training.eval_every`       | 500     | Validation interval                            |

---

## License

MIT — see [LICENSE](LICENSE).