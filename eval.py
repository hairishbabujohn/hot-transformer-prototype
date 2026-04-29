"""Evaluate a saved Homeostatic Transformer checkpoint."""

from __future__ import annotations

import argparse

import torch

from hot.data import get_dataloaders
from hot.model import HoTEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HoT checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cuda / cpu). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    zcfg = cfg.get("czu", {})

    _, val_loader, n_classes = get_dataloaders(dcfg)

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
        czu_warmup_steps=zcfg.get("warmup_steps", 1000),
        czu_update_every=zcfg.get("update_every", 500),
        czu_ema_beta=zcfg.get("ema_beta", 0.95),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "czu_state" in ckpt:
        model.czu.load_state_dict(ckpt["czu_state"])
    model.eval()

    correct = total = 0
    route_counts = torch.zeros(3, device=device)

    with torch.no_grad():
        for x, y, sample_ids in val_loader:
            x, y, sample_ids = x.to(device), y.to(device), sample_ids.to(device)
            logits, routes, entropies = model(x, sample_ids=sample_ids, force_c=False)
            del entropies

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            route_counts += torch.stack(routes).sum(dim=(0, 1))

    acc = correct / total if total else 0.0
    route_total = route_counts.sum().clamp_min(1.0)
    route_dist = route_counts / route_total

    print(f"\n[eval] checkpoint : {args.checkpoint}")
    print(f"[eval] device     : {device}")
    print(f"[eval] val_acc    : {acc:.4f}  ({correct}/{total})")
    print(
        "[eval] routing    : "
        f"A={route_dist[0].item():.1%}  "
        f"B={route_dist[1].item():.1%}  "
        f"C={route_dist[2].item():.1%}"
    )
    print("\n[eval] per-layer thresholds (H_low / H_high):")
    for i, (lo, hi) in enumerate(model.czu.get_all_thresholds()):
        print(f"  layer {i}: H_low={lo:.4f}  H_high={hi:.4f}")


if __name__ == "__main__":
    main()
