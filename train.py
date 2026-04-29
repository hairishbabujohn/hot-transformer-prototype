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
import statistics

import torch
import torch.nn as nn
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
def evaluate(
    model: HoTEncoder,
    val_loader,
    czu: CZU,
    device: torch.device,
    return_metrics: bool = False,
    force_c: bool = False,
) -> tuple:
    """Evaluate on the validation set; return (accuracy, %A, %B, %C[, metrics])."""
    model.eval()
    correct = total = 0
    route_counts = [0, 0, 0]

    n_layers = model.n_layers
    total_samples = 0
    sum_g = torch.zeros(3, device=device)
    sum_gate_entropy = torch.tensor(0.0, device=device)
    sum_impacts = torch.zeros(3, device=device)
    sum_ratios = torch.zeros(3, device=device)
    sum_g_layer = torch.zeros(n_layers, 3, device=device)
    sum_ratio_layer = torch.zeros(n_layers, 3, device=device)

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        thresholds = czu.get_all_thresholds()
        if return_metrics:
            logits, _, routes, diagnostics = model(
                x, thresholds, force_c=force_c, return_diagnostics=True,
            )
        else:
            logits, _, routes = model(x, thresholds, force_c=force_c)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        for r in routes:
            route_counts[_route_idx(r)] += 1

        if return_metrics:
            batch_size = x.size(0)
            total_samples += batch_size
            for layer_idx, diag in enumerate(diagnostics):
                g = diag["g"].to(device)
                impacts = torch.stack([diag["A_impact"], diag["B_impact"], diag["C_impact"]]).to(device)
                ratios = torch.stack([diag["A_ratio"], diag["B_ratio"], diag["C_ratio"]]).to(device)

                sum_g += g * batch_size
                sum_gate_entropy += diag["gate_entropy"].to(device) * batch_size
                sum_impacts += impacts * batch_size
                sum_ratios += ratios * batch_size
                sum_g_layer[layer_idx] += g * batch_size
                sum_ratio_layer[layer_idx] += ratios * batch_size

    acc = correct / total if total else 0.0
    total_r = sum(route_counts) or 1
    pct = tuple(c / total_r for c in route_counts)

    metrics = None
    if return_metrics:
        if total_samples == 0:
            g_mean = torch.zeros(3, device=device)
            gate_entropy_mean = 0.0
            impact_mean = torch.zeros(3, device=device)
            ratio_mean = torch.zeros(3, device=device)
            g_layer_mean = torch.zeros(n_layers, 3, device=device)
            ratio_layer_mean = torch.zeros(n_layers, 3, device=device)
        else:
            denom = total_samples * n_layers
            g_mean = sum_g / denom
            gate_entropy_mean = (sum_gate_entropy / denom).item()
            impact_mean = sum_impacts / denom
            ratio_mean = sum_ratios / denom
            g_layer_mean = sum_g_layer / total_samples
            ratio_layer_mean = sum_ratio_layer / total_samples

        impact_norm = impact_mean / (impact_mean.sum() + 1e-9)
        alignment_error = (g_mean - impact_norm).abs().sum().item()

        metrics = {
            "g_mean": g_mean.tolist(),
            "A_mean": g_mean[0].item(),
            "B_mean": g_mean[1].item(),
            "C_mean": g_mean[2].item(),
            "gate_entropy_mean": gate_entropy_mean,
            "A_impact_mean": impact_mean[0].item(),
            "B_impact_mean": impact_mean[1].item(),
            "C_impact_mean": impact_mean[2].item(),
            "A_ratio_mean": ratio_mean[0].item(),
            "B_ratio_mean": ratio_mean[1].item(),
            "C_ratio_mean": ratio_mean[2].item(),
            "g_layer_mean": g_layer_mean.tolist(),
            "A_ratio_layer_mean": ratio_layer_mean[:, 0].tolist(),
            "B_ratio_layer_mean": ratio_layer_mean[:, 1].tolist(),
            "C_ratio_layer_mean": ratio_layer_mean[:, 2].tolist(),
            "alignment_error": alignment_error,
        }

    model.train()
    if return_metrics:
        return (acc,) + pct + (metrics,)
    return (acc,) + pct


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _init_model_and_czu(
    mcfg: dict,
    dcfg: dict,
    n_classes: int,
    zcfg: dict,
    device: torch.device,
) -> tuple:
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
    czu = CZU(
        n_layers=mcfg.get("n_layers", 4),
        warmup_steps=zcfg.get("warmup_steps", 1000),
        update_every=zcfg.get("update_every", 500),
        ema_beta=zcfg.get("ema_beta", 0.95),
    )
    return model, czu


def _layer_worst_offenders(metrics: dict) -> dict:
    g_layer = metrics.get("g_layer_mean", [])
    c_ratio_layer = metrics.get("C_ratio_layer_mean", [])

    a_vals = [layer[0] for layer in g_layer] if g_layer else []
    b_vals = [layer[1] for layer in g_layer] if g_layer else []

    min_a_val = min(a_vals) if a_vals else 0.0
    min_b_val = min(b_vals) if b_vals else 0.0
    max_c_val = max(c_ratio_layer) if c_ratio_layer else 0.0

    return {
        "min_a_val": min_a_val,
        "min_a_idx": a_vals.index(min_a_val) if a_vals else -1,
        "min_b_val": min_b_val,
        "min_b_idx": b_vals.index(min_b_val) if b_vals else -1,
        "max_c_val": max_c_val,
        "max_c_idx": c_ratio_layer.index(max_c_val) if c_ratio_layer else -1,
    }


def _check_acceptance(
    eval_records: list,
    best_val_acc_hot: float,
    best_val_acc_c_only: float,
    k: int = 5,
) -> tuple:
    if len(eval_records) < k:
        return False, f"fewer than {k} eval checkpoints ({len(eval_records)})"

    if best_val_acc_hot < best_val_acc_c_only:
        return (
            False,
            f"best_val_acc_hot {best_val_acc_hot:.4f} < best_val_acc_c_only {best_val_acc_c_only:.4f}",
        )

    last_records = eval_records[-k:]
    for record in last_records:
        metrics = record["metrics"]
        step = record["step"]

        if metrics["A_mean"] < 0.05:
            return False, f"step {step} A_mean {metrics['A_mean']:.3f} < 0.05"
        if metrics["B_mean"] < 0.05:
            return False, f"step {step} B_mean {metrics['B_mean']:.3f} < 0.05"
        if metrics["C_mean"] > 0.90:
            return False, f"step {step} C_mean {metrics['C_mean']:.3f} > 0.90"

        g_layer = metrics["g_layer_mean"]
        a_vals = [layer[0] for layer in g_layer]
        b_vals = [layer[1] for layer in g_layer]
        min_a_val = min(a_vals)
        min_b_val = min(b_vals)
        if min_a_val < 0.02:
            idx = a_vals.index(min_a_val)
            return False, f"step {step} A_layer_mean[{idx}] {min_a_val:.3f} < 0.02"
        if min_b_val < 0.02:
            idx = b_vals.index(min_b_val)
            return False, f"step {step} B_layer_mean[{idx}] {min_b_val:.3f} < 0.02"

        if metrics["gate_entropy_mean"] < 0.3:
            return (
                False,
                f"step {step} gate_entropy {metrics['gate_entropy_mean']:.3f} < 0.3",
            )
        if metrics["A_ratio_mean"] < 0.01:
            return False, f"step {step} A_ratio {metrics['A_ratio_mean']:.3f} < 0.01"
        if metrics["B_ratio_mean"] < 0.01:
            return False, f"step {step} B_ratio {metrics['B_ratio_mean']:.3f} < 0.01"
        if metrics["C_ratio_mean"] > 0.85:
            return False, f"step {step} C_ratio {metrics['C_ratio_mean']:.3f} > 0.85"

        c_ratio_layer = metrics["C_ratio_layer_mean"]
        max_c_val = max(c_ratio_layer)
        if max_c_val > 0.90:
            idx = c_ratio_layer.index(max_c_val)
            return False, f"step {step} C_ratio_layer[{idx}] {max_c_val:.3f} > 0.90"

        if metrics["alignment_error"] > 0.3:
            return (
                False,
                f"step {step} alignment_error {metrics['alignment_error']:.3f} > 0.3",
            )

    ema_vals = [record["ema_C"] for record in last_records]
    ema_std = statistics.pstdev(ema_vals) if len(ema_vals) > 1 else 0.0
    if ema_std > 0.1:
        return False, f"EMA(C_mean) std {ema_std:.3f} > 0.1"

    return True, ""


def _print_final_report(
    best_val_acc_hot: float,
    best_val_acc_c_only: float,
    eval_records: list,
    passed: bool,
    reason: str,
    k: int = 5,
) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n[final] status={status}")
    print(
        f"[final] best_val_acc_hot={best_val_acc_hot:.4f} "
        f"best_val_acc_c_only={best_val_acc_c_only:.4f}"
    )

    if eval_records:
        last_records = eval_records[-k:]
        for record in last_records:
            metrics = record["metrics"]
            worst = _layer_worst_offenders(metrics)
            print(
                f"  [eval] step={record['step']} val_acc={record['val_acc']:.4f} "
                f"A_mean={metrics['A_mean']:.3f} "
                f"B_mean={metrics['B_mean']:.3f} "
                f"C_mean={metrics['C_mean']:.3f} "
                f"gate_entropy={metrics['gate_entropy_mean']:.3f} "
                f"A_ratio={metrics['A_ratio_mean']:.3f} "
                f"B_ratio={metrics['B_ratio_mean']:.3f} "
                f"C_ratio={metrics['C_ratio_mean']:.3f} "
                f"alignment_error={metrics['alignment_error']:.3f} "
                f"min_A_layer={worst['min_a_val']:.3f}(L{worst['min_a_idx']}) "
                f"min_B_layer={worst['min_b_val']:.3f}(L{worst['min_b_idx']}) "
                f"max_C_ratio_layer={worst['max_c_val']:.3f}(L{worst['max_c_idx']}) "
                f"ema_C={record['ema_C']:.3f}"
            )

        if len(last_records) > 1:
            ema_std = statistics.pstdev([record["ema_C"] for record in last_records])
        else:
            ema_std = 0.0
        print(f"[final] ema_C_std_last_{len(last_records)}={ema_std:.4f}")
    else:
        print("[final] no eval records collected")

    if not passed:
        print(f"[final] first_failure={reason}")


def train_loop(
    model: HoTEncoder,
    czu: CZU,
    cfg: dict,
    train_loader,
    val_loader,
    device: torch.device,
    total_steps: int,
    eval_every: int,
    log_every: int,
    grad_clip: float,
    lr: float,
    weight_decay: float,
    run_dir=None,
    force_c_only: bool = False,
    track_metrics: bool = False,
    run_label: str = "hot",
) -> tuple:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    eval_records = []
    ema_c = None
    ema_c_history = []
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
        force_c = True if force_c_only else czu.force_path_c()
        logits, oem_vals, routes = model(x, thresholds, force_c=force_c)
        loss = criterion(logits, y)

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
            if force_c_only:
                phase = "c-only"
            else:
                phase = "warmup" if force_c else "running"
            thresholds_str = " ".join(
                f"L{i}[{lo:.2f},{hi:.2f}]"
                for i, (lo, hi) in enumerate(czu.get_all_thresholds())
            )
            print(
                f"[train:{run_label}] step={step}/{total_steps} loss={loss.item():.4f} "
                f"phase={phase} "
                f"A={pct_a:.2%} B={pct_b:.2%} C={pct_c:.2%} "
                f"ent={mean_ent:.3f} "
                f"thresholds={thresholds_str} "
                f"elapsed={elapsed:.1f}s"
            )

        # ---- Evaluation ----
        if step % eval_every == 0:
            if track_metrics:
                val_acc, pct_a, pct_b, pct_c, metrics = evaluate(
                    model, val_loader, czu, device, return_metrics=True, force_c=force_c_only,
                )
            else:
                val_acc, pct_a, pct_b, pct_c = evaluate(
                    model, val_loader, czu, device, return_metrics=False, force_c=force_c_only,
                )
                metrics = None
            print(
                f"  [eval:{run_label}] step={step} val_acc={val_acc:.4f} "
                f"routing A={pct_a:.1%} B={pct_b:.1%} C={pct_c:.1%}"
            )
            if track_metrics:
                c_mean = metrics["C_mean"]
                ema_c = c_mean if ema_c is None else 0.8 * ema_c + 0.2 * c_mean
                ema_c_history.append(ema_c)
                eval_records.append(
                    {
                        "step": step,
                        "val_acc": val_acc,
                        "metrics": metrics,
                        "ema_C": ema_c,
                    }
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if run_dir:
                    ckpt = {
                        "step": step,
                        "model_state": model.state_dict(),
                        "czu_state": czu.state_dict(),
                        "config": cfg,
                        "val_acc": val_acc,
                    }
                    ckpt_path = os.path.join(run_dir, "best_model.pt")
                    torch.save(ckpt, ckpt_path)
                    print(
                        f"  [ckpt:{run_label}] saved best (acc={val_acc:.4f}) -> {ckpt_path}"
                    )

    print(f"\n[done:{run_label}] best_val_acc={best_val_acc:.4f}")
    return best_val_acc, eval_records, ema_c_history


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

    # ---- Data ----
    train_loader, val_loader, n_classes = get_dataloaders(dcfg)

    total_steps = tcfg.get("steps", 10000)
    eval_every = tcfg.get("eval_every", 500)
    log_every = lcfg.get("log_every", 100)
    grad_clip = tcfg.get("grad_clip", 1.0)

    # ---- Run directory ----
    run_name = (
        f"hot_{cfg.get('run_name', 'exp')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(tcfg.get("save_dir", "runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run_dir={run_dir}")

    # ---- Baseline: C-only ----
    model_c, czu_c = _init_model_and_czu(mcfg, dcfg, n_classes, zcfg, device)
    n_params = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    print(f"[train:c-only] params={n_params:,}")
    best_val_acc_c_only, _, _ = train_loop(
        model_c,
        czu_c,
        cfg,
        train_loader,
        val_loader,
        device,
        total_steps,
        eval_every,
        log_every,
        grad_clip,
        tcfg.get("lr", 1e-3),
        tcfg.get("weight_decay", 1e-2),
        run_dir=None,
        force_c_only=True,
        track_metrics=False,
        run_label="c-only",
    )

    # ---- HoT adaptive training ----
    model_hot, czu_hot = _init_model_and_czu(mcfg, dcfg, n_classes, zcfg, device)
    n_params = sum(p.numel() for p in model_hot.parameters() if p.requires_grad)
    print(f"[train:hot] params={n_params:,}")
    best_val_acc_hot, eval_records, ema_c_history = train_loop(
        model_hot,
        czu_hot,
        cfg,
        train_loader,
        val_loader,
        device,
        total_steps,
        eval_every,
        log_every,
        grad_clip,
        tcfg.get("lr", 1e-3),
        tcfg.get("weight_decay", 1e-2),
        run_dir=run_dir,
        force_c_only=False,
        track_metrics=True,
        run_label="hot",
    )

    passed, reason = _check_acceptance(
        eval_records, best_val_acc_hot, best_val_acc_c_only, k=5,
    )
    _print_final_report(
        best_val_acc_hot,
        best_val_acc_c_only,
        eval_records,
        passed,
        reason,
        k=5,
    )


if __name__ == "__main__":
    main()
