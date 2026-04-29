"""
Train a Homeostatic Transformer (HoT) encoder classifier.

Usage
-----
    python train.py --config configs/hot_synthetic_tiny.yaml

The script:
- Auto-selects CUDA if available.
- Trains baseline (C-only) then adaptive routing.
"""

import argparse
import os
import time
from datetime import datetime
import statistics
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from hot.model import HoTEncoder
from hot.data import get_dataloaders


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Homeostatic Transformer (HoT)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--mode", type=str, choices=["c-only", "hot", "both"], default="both", help="Which model to run")
    parser.add_argument("--json_output", action="store_true", help="Print final results as JSON")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensuring determinism for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: HoTEncoder,
    val_loader,
    device: torch.device,
    return_metrics: bool = False,
    force_c: bool = False,
) -> tuple:
    """Evaluate on the validation set; return (accuracy, %A, %B, %C[, metrics])."""
    model.eval()
    correct = total = 0

    n_layers = model.n_layers
    total_samples = 0
    sum_g = torch.zeros(3, device=device)
    sum_gate_entropy = torch.tensor(0.0, device=device)
    sum_impacts = torch.zeros(3, device=device)
    sum_ratios = torch.zeros(3, device=device)
    sum_g_layer = torch.zeros(n_layers, 3, device=device)
    sum_ratio_layer = torch.zeros(n_layers, 3, device=device)
    
    all_losses = []
    all_gC = []
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    for x, y, sample_ids in val_loader:
        x, y, sample_ids = x.to(device), y.to(device), sample_ids.to(device)
        
        if return_metrics:
            logits, routes, entropies, diagnostics = model(
                x, sample_ids=sample_ids, force_c=force_c, return_diagnostics=True,
            )
        else:
            logits, routes, entropies = model(x, sample_ids=sample_ids, force_c=force_c)
            
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if return_metrics:
            batch_size = x.size(0)
            for i, ent in enumerate(entropies):
                sum_gate_entropy += ent.std()
            
            loss_ps = criterion_none(logits, y)
            all_losses.append(loss_ps.detach().cpu())
            
            # Stack routes: (n_layers, B, 3)
            g_stack = torch.stack(routes) # (L, B, 3)
            
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
        
        route_change_rate = 0.0
        if len(diagnostics) > 0:
            route_change_rate = torch.stack([d.get("route_change_rate", torch.tensor(0.0)) for d in diagnostics]).mean().item()
            
        metrics = {
            "g_mean": g_mean.tolist(),
            "A_mean": g_mean[0].item(),
            "B_mean": g_mean[1].item(),
            "C_mean": g_mean[2].item(),
            "gate_entropy_mean": gate_entropy_mean,
            "route_change_rate_mean": route_change_rate,
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
        return (acc, metrics["A_mean"], metrics["B_mean"], metrics["C_mean"], metrics)
        
    # If not returning metrics but need to pass back pct for logging, we can just compute it here.
    return (acc, 0.0, 0.0, 1.0) if force_c else (acc, 0.33, 0.33, 0.33)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _init_model(
    mcfg: dict,
    dcfg: dict,
    n_classes: int,
    device: torch.device,
) -> HoTEncoder:
    model = HoTEncoder(
        vocab_size=dcfg.get("vocab_size", 4),
        d_model=mcfg.get("d_model", 64),
        n_layers=mcfg.get("n_layers", 4),
        n_heads=mcfg.get("n_heads", 4),
        n_classes=n_classes,
        max_seq_len=dcfg.get("seq_len", 64),
        conv_kernel_size=mcfg.get("conv_kernel_size", 7),
        dropout=mcfg.get("dropout", 0.1),
    ).to(device)
    return model


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

    for record in eval_records[-k:]:
        metrics = record["metrics"]
        step = record["step"]

        g_layer = metrics["g_layer_mean"]
        a_vals = [layer[0] for layer in g_layer]
        b_vals = [layer[1] for layer in g_layer]
        c_vals = [layer[2] for layer in g_layer]

        for i in range(len(g_layer)):
            A = a_vals[i]
            B = b_vals[i]
            C = c_vals[i]
            if A < 0.05 or A > 0.70:
                return False, f"step {step} layer {i} A {A:.3f} outside [0.05, 0.70]"
            if B < 0.05 or B > 0.70:
                return False, f"step {step} layer {i} B {B:.3f} outside [0.05, 0.70]"
            if C < 0.05 or C > 0.90:
                return False, f"step {step} layer {i} C {C:.3f} outside [0.05, 0.90]"

        if metrics.get("gate_entropy_mean", 1.0) < 0.01:
            return False, f"step {step} global entropy std {metrics.get('gate_entropy_mean'):.4f} < 0.01"

        if metrics.get("route_change_rate_mean", 0.0) > 0.3:
            return False, f"step {step} route_change_rate {metrics.get('route_change_rate_mean'):.4f} > 0.3"

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
                f"C_high-C_low={metrics.get('C_high', 0) - metrics.get('C_low', 0):.3f} "
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
    
    # Optimizer setup: separate gate params
    gate_params = []
    base_params = []
    for n, p in model.named_parameters():
        if "gate" in n:
            gate_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": lr, "weight_decay": weight_decay},
        {"params": gate_params, "lr": lr * 0.3, "weight_decay": 0.0}
    ])
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    eval_records = []
    ema_c = None
    ema_c_history = []
    train_iter = iter(train_loader)
    t0 = time.time()

    model.train()
    
    tau_start = 2.0
    tau_end = 1.0
    tau_steps = 2000

    for step in range(1, total_steps + 1):
        g_stack = None
        # ---- Fetch batch ----
        try:
            x, y, sample_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, sample_ids = next(train_iter)
        x, y, sample_ids = x.to(device), y.to(device), sample_ids.to(device)

        # ---- Forward ----
        tau = tau_end + (tau_start - tau_end) * max(0.0, (tau_steps - step) / tau_steps)
        
        # Training Schedule
        if not force_c_only:
            force_c = model.czu.force_path_c()
        else:
            force_c = True
            
        logits, routes, entropies = model(x, sample_ids=sample_ids, force_c=force_c)
        loss = criterion(logits, y)

        if not force_c_only and len(routes) > 0:
            model.czu.update(entropies)
            
            g_stack = torch.stack(routes)
            g_stack_b = g_stack.transpose(0, 1) # (B, n_layers, 3)
            gC = g_stack_b[:, :, 2]  # (B, n_layers)

        assert not torch.isnan(loss), "Loss is NaN"
        if not force_c_only and len(routes) > 0:
            if g_stack is not None:
                assert not torch.isnan(g_stack).any(), "Gate probabilities contain NaN"

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        
        current_clip = 0.5 if step <= 1000 else grad_clip
        if current_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), current_clip)
            
        optimizer.step()
        
        # Clamp alpha for all layers
        with torch.no_grad():
            for layer in model.layers:
                layer.alpha.data.clamp_(0.1, 1.0)

        # ---- Logging ----
        if step % log_every == 0:
            if force_c_only:
                pct_a, pct_b, pct_c = 0.0, 0.0, 1.0
            elif len(routes) > 0:
                g_mean_total = torch.stack(routes).mean(dim=(0, 1))
                pct_a, pct_b, pct_c = g_mean_total.tolist()
            else:
                pct_a, pct_b, pct_c = 0.0, 0.0, 0.0
                
            elapsed = time.time() - t0
            
            alpha_str = " ".join(f"L{i}:{layer.alpha.item():.2f}" for i, layer in enumerate(model.layers))
            loss_str = f"loss={loss.item():.4f}"

            print(
                f"[train:{run_label}] step={step}/{total_steps} {loss_str} "
                f"A={pct_a:.2%} B={pct_b:.2%} C={pct_c:.2%} "
                f"alphas=[{alpha_str}] "
                f"elapsed={elapsed:.1f}s"
            )

        # ---- Evaluation ----
        if step % eval_every == 0:
            if track_metrics:
                val_acc, pct_a, pct_b, pct_c, metrics = evaluate(
                    model, val_loader, device, return_metrics=True, force_c=force_c_only,
                )
                gate_entropy = metrics.get("gate_entropy_mean", 0)
                eval_str = (
                    f"  [eval:{run_label}] step={step} val_acc={val_acc:.4f}\n"
                    f"      routing A={pct_a:.1%} B={pct_b:.1%} C={pct_c:.1%} | entropy_std={gate_entropy:.3f}\n"
                )
            else:
                val_acc, pct_a, pct_b, pct_c = evaluate(
                    model, val_loader, device, return_metrics=False, force_c=force_c_only,
                )
                metrics = None
                eval_str = f"  [eval:{run_label}] step={step} val_acc={val_acc:.4f} routing A={pct_a:.1%} B={pct_b:.1%} C={pct_c:.1%}"
                
            print(eval_str)
            
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

    # ---- Overrides ----
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.seq_len is not None:
        cfg["data"]["seq_len"] = args.seq_len

    # Apply global seed
    seed = cfg.get("data", {}).get("seed", 42)
    set_seed(seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ---- Config sections ----
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("training", {})
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

    results = {}

    # ---- Baseline: C-only ----
    if args.mode in ["c-only", "both"]:
        model_c = _init_model(mcfg, dcfg, n_classes, device)
        n_params_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
        print(f"[train:c-only] params={n_params_c:,}")
        best_val_acc_c_only, _, _ = train_loop(
            model_c,
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
        results["c_only"] = {"best_val_acc": best_val_acc_c_only, "params": n_params_c}
    else:
        best_val_acc_c_only = 0.0

    # ---- HoT adaptive training ----
    if args.mode in ["hot", "both"]:
        model_hot = _init_model(mcfg, dcfg, n_classes, device)
        n_params_hot = sum(p.numel() for p in model_hot.parameters() if p.requires_grad)
        print(f"[train:hot] params={n_params_hot:,}")
        best_val_acc_hot, eval_records, ema_c_history = train_loop(
            model_hot,
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
        
        last_metrics = eval_records[-1]["metrics"] if eval_records else {}
        results["hot"] = {
            "best_val_acc": best_val_acc_hot,
            "params": n_params_hot,
            "A_mean": last_metrics.get("A_mean", 0.0),
            "B_mean": last_metrics.get("B_mean", 0.0),
            "C_mean": last_metrics.get("C_mean", 0.0),
            "gate_entropy": last_metrics.get("gate_entropy_mean", 0.0),
            "c_mean_history": ema_c_history,
        }

        if args.mode == "both":
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

    if args.json_output:
        print("JSON_OUTPUT_START")
        print(json.dumps(results))
        print("JSON_OUTPUT_END")


if __name__ == "__main__":
    main()
