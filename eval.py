"""
Evaluate a saved HoT checkpoint.

Usage
-----
    python eval.py runs/<run_dir>/best_model.pt [--device cpu]

Reports:
- Validation accuracy.
- Per-path routing distribution (% A / B / C).
- Per-layer H_low / H_high thresholds.
"""

import argparse

import torch
import yaml

from hot.model import HoTEncoder
from hot.czu import CZU
from hot.data import get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HoT checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument(
        "--device", default=None,
        help="Device override (cuda / cpu). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def _route_idx(route_info) -> int:
    if isinstance(route_info, int):
        return route_info
    return int(route_info.argmax().item())


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    zcfg = cfg.get("czu", {})

    # ---- Data ----
    _, val_loader, n_classes = get_dataloaders(dcfg)

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
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- CZU ----
    czu = CZU(
        n_layers=mcfg.get("n_layers", 4),
        warmup_steps=zcfg.get("warmup_steps", 1000),
        update_every=zcfg.get("update_every", 500),
        ema_beta=zcfg.get("ema_beta", 0.95),
    )
    czu.load_state_dict(ckpt["czu_state"])

    # ---- Evaluate ----
    correct = total = 0
    route_counts = [0, 0, 0]

    with torch.no_grad():
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

    print(f"\n[eval] checkpoint : {args.checkpoint}")
    print(f"[eval] device     : {device}")
    print(f"[eval] val_acc    : {acc:.4f}  ({correct}/{total})")
    print(
        f"[eval] routing    : "
        f"A={route_counts[0]/total_r:.1%}  "
        f"B={route_counts[1]/total_r:.1%}  "
        f"C={route_counts[2]/total_r:.1%}"
    )
    print(f"\n[eval] per-layer thresholds (H_low / H_high):")
    for i in range(czu.n_layers):
        lo, hi = czu.get_thresholds(i)
        print(f"  layer {i}: H_low={lo:.4f}  H_high={hi:.4f}")


if __name__ == "__main__":
    main()
