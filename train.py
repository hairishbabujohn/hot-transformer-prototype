"""
Train a Homeostatic Transformer (HoT) encoder classifier.

Usage
-----
    python train.py --config configs/hot_synthetic_tiny.yaml

The script:
- Auto-selects CUDA if available.
- Runs the CZU warmup phase (forced Path C), then switches to adaptive routing.
- Logs routing distribution (% A / B / C per layer), entropy stats, and
  validation accuracy every ``logging.log_every`` steps.
- Saves the best validation-accuracy checkpoint to ``<save_dir>/<run_name>/best_model.pt``.
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from hot.model import HoTEncoder
from hot.czu import CZU
from hot.data import get_dataloaders


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Homeostatic Transformer (HoT)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_idx(route_info) -> int:
    """Return integer path index from either a hard int or soft weight tensor."""
    if isinstance(route_info, int):
        return route_info
    # Soft weights tensor [w_A, w_B, w_C]: pick dominant path for logging
    return int(route_info.argmax().item())


def route_distribution(routes: list) -> tuple:
    """Fraction of layers using Path A / B / C this step."""
    counts = [0, 0, 0]
    for r in routes:
        counts[_route_idx(r)] += 1
    total = len(routes) or 1
    return counts[0] / total, counts[1] / total, counts[2] / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: HoTEncoder, val_loader, czu: CZU, device: torch.device) -> tuple:
    """Evaluate on the validation set; return (accuracy, %A, %B, %C)."""
    model.eval()
    correct = total = 0
    route_counts = [0, 0, 0]

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        thresholds = czu.get_all_thresholds()
        logits, _, routes = model(x, thresholds, force_c=False)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        for r in routes:
            route_counts[_route_idx(r)] += 1

    acc = correct / total if total else 0.0
    total_r = sum(route_counts) or 1
    pct = tuple(c / total_r for c in route_counts)

    model.train()
    return (acc,) + pct


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ---- Config sections ----
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("training", {})
    zcfg = cfg.get("czu", {})
    lcfg = cfg.get("logging", {})
    rcfg = cfg.get("routing_reg", {})

    # ---- Data ----
    train_loader, val_loader, n_classes = get_dataloaders(dcfg)

    # ---- Model ----
    model = HoTEncoder(
        vocab_size=dcfg.get("vocab_size", 4),
        d_model=mcfg.get("d_model", 64),
        n_layers=mcfg.get("n_layers", 4),
        n_heads=mcfg.get("n_heads", 4),
        n_classes=n_classes,
        max_seq_len=dcfg.get("seq_len", 64),
        conv_kernel_size=mcfg.get("conv_kernel_size", 7),
        dropout=mcfg.get("dropout", 0.1),
        gate_temperature=mcfg.get("gate_temperature", 0.05),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={n_params:,}")

    # ---- CZU ----
    czu = CZU(
        n_layers=mcfg.get("n_layers", 4),
        warmup_steps=zcfg.get("warmup_steps", 1000),
        update_every=zcfg.get("update_every", 500),
        ema_beta=zcfg.get("ema_beta", 0.95),
        h_low_clamp=(
            zcfg.get("h_low_clamp_min", 0.2),
            zcfg.get("h_low_clamp_max", 0.75),
        ),
        h_high_clamp=(
            zcfg.get("h_high_clamp_min", 0.35),
            zcfg.get("h_high_clamp_max", 0.90),
        ),
        schedule_steps=zcfg.get("schedule_steps", 2000),
    )

    # ---- Routing regularization hyperparameters ----
    lambda_balance = rcfg.get("lambda_balance", 0.05)
    lambda_compute = rcfg.get("lambda_compute", 0.02)
    target_dist = rcfg.get("target_dist", [1 / 3, 1 / 3, 1 / 3])

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.get("lr", 1e-3),
        weight_decay=tcfg.get("weight_decay", 1e-2),
    )
    criterion = nn.CrossEntropyLoss()

    # ---- Run directory ----
    run_name = (
        f"hot_{cfg.get('run_name', 'exp')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(tcfg.get("save_dir", "runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run_dir={run_dir}")

    total_steps = tcfg.get("steps", 10000)
    eval_every = tcfg.get("eval_every", 500)
    log_every = lcfg.get("log_every", 100)
    grad_clip = tcfg.get("grad_clip", 1.0)

    best_val_acc = 0.0
    train_iter = iter(train_loader)
    t0 = time.time()

    model.train()

    for step in range(1, total_steps + 1):
        # ---- Fetch batch ----
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # ---- Forward ----
        thresholds = czu.get_all_thresholds()
        force_c = czu.force_path_c()
        logits, oem_vals, routes = model(x, thresholds, force_c=force_c)
        loss = criterion(logits, y)

        # ---- Routing regularization (post-warmup only) ----
        reg_loss_val = 0.0
        if not force_c and (lambda_balance > 0 or lambda_compute > 0):
            route_tensors = [r for r in routes if isinstance(r, torch.Tensor)]
            if route_tensors:
                # avg_w shape: (3,) — mean routing weight [w_A, w_B, w_C]
                avg_w = torch.stack(route_tensors).mean(dim=0)

                if lambda_balance > 0:
                    # KL(target || avg_w): minimising this pushes avg_w toward
                    # the target distribution, preventing collapse to all-C.
                    tgt = torch.tensor(
                        target_dist, dtype=avg_w.dtype, device=avg_w.device
                    )
                    l_bal = F.kl_div(
                        avg_w.clamp(min=1e-8).log(), tgt, reduction="sum"
                    )
                    loss = loss + lambda_balance * l_bal
                    reg_loss_val += (lambda_balance * l_bal).item()

                if lambda_compute > 0:
                    # Compute-cost penalty: penalise heavy use of Path C (O(N²)).
                    l_cmp = avg_w[2]
                    loss = loss + lambda_compute * l_cmp
                    reg_loss_val += (lambda_compute * l_cmp).item()

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # ---- Update CZU with this step's entropies ----
        czu.update(oem_vals)

        # ---- Logging ----
        if step % log_every == 0:
            pct_a, pct_b, pct_c = route_distribution(routes)
            mean_ent = sum(v.item() for v in oem_vals) / len(oem_vals)
            elapsed = time.time() - t0
            phase = "warmup" if force_c else "running"
            thresholds_str = " ".join(
                f"L{i}[{lo:.2f},{hi:.2f}]"
                for i, (lo, hi) in enumerate(czu.get_all_thresholds())
            )
            print(
                f"step={step}/{total_steps} loss={loss.item():.4f} "
                f"reg={reg_loss_val:.4f} "
                f"phase={phase} "
                f"A={pct_a:.2%} B={pct_b:.2%} C={pct_c:.2%} "
                f"ent={mean_ent:.3f} "
                f"thresholds={thresholds_str} "
                f"elapsed={elapsed:.1f}s"
            )

        # ---- Evaluation ----
        if step % eval_every == 0:
            val_acc, pct_a, pct_b, pct_c = evaluate(model, val_loader, czu, device)
            print(
                f"  [eval] step={step} val_acc={val_acc:.4f} "
                f"routing A={pct_a:.1%} B={pct_b:.1%} C={pct_c:.1%}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "czu_state": czu.state_dict(),
                    "config": cfg,
                    "val_acc": val_acc,
                }
                ckpt_path = os.path.join(run_dir, "best_model.pt")
                torch.save(ckpt, ckpt_path)
                print(f"  [ckpt] saved best (acc={val_acc:.4f}) -> {ckpt_path}")

    print(f"\n[done] best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
