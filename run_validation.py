"""Run reproducibility and routing checks for the HoT prototype."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
CONFIG_FILE = ROOT / "configs" / "hot_synthetic_tiny.yaml"


def _python_has_requirements(python_exe: Path) -> bool:
    code = "import importlib.util; import sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ['torch','yaml','numpy']) else 1)"
    return subprocess.run([str(python_exe), "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def resolve_python() -> Path:
    env_python = os.environ.get("HOT_PYTHON")
    candidates = []
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(Path(sys.executable))
    candidates.append(ROOT.parent / "hsm-prototype" / "venv" / "Scripts" / "python.exe")
    candidates.append(ROOT.parent / "hsm-prototype" / ".venv" / "Scripts" / "python.exe")

    for candidate in candidates:
        if candidate.exists() and _python_has_requirements(candidate):
            return candidate

    raise RuntimeError(
        "No Python environment with torch, pyyaml, and numpy was found. "
        "Set HOT_PYTHON to a compatible python.exe."
    )


def run_experiment(python_exe: Path, seed=42, dataset=None, seq_len=None, mode="both"):
    cmd = [
        str(python_exe),
        str(TRAIN_SCRIPT),
        "--config",
        str(CONFIG_FILE),
        "--seed",
        str(seed),
        "--mode",
        mode,
        "--json_output",
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    if seq_len:
        cmd.extend(["--seq_len", str(seq_len)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        print("ERROR RUNNING COMMAND:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    out = result.stdout
    start_idx = out.find("JSON_OUTPUT_START")
    end_idx = out.find("JSON_OUTPUT_END")
    if start_idx == -1 or end_idx == -1:
        print("ERROR: Could not find JSON output block.")
        print("STDOUT:", out)
        raise RuntimeError("No JSON output")

    json_str = out[start_idx + len("JSON_OUTPUT_START") : end_idx].strip()
    return json.loads(json_str)


def main():
    parser = argparse.ArgumentParser(description="Validate HoT adaptive routing behavior")
    parser.add_argument("--skip_listops", action="store_true", help="Skip the optional LRA/ListOps phase")
    args = parser.parse_args()

    python_exe = resolve_python()
    print(f"Using Python: {python_exe}")

    print("=== Phase 1: Reproducibility ===")
    p1_results = []
    for i in range(3):
        print(f"  Run {i + 1}/3")
        res = run_experiment(python_exe, seed=42, mode="hot")
        p1_results.append(res["hot"]["best_val_acc"])

    p1_std = statistics.pstdev(p1_results) if len(p1_results) > 1 else 0.0
    print(f"Phase 1 Results: {p1_results}")
    print(f"Phase 1 StdDev: {p1_std:.5f}")
    assert p1_std <= 0.005, f"Reproducibility failed: std {p1_std} > 0.005"

    print("\n=== Phase 2: Robustness ===")
    seeds = [1, 7, 42, 123, 999]
    p2_baseline_accs = []
    p2_hot_accs = []
    p2_A = []
    p2_B = []
    p2_C = []

    for seed in seeds:
        print(f"  Seed {seed}")
        res = run_experiment(python_exe, seed=seed, mode="both")
        p2_baseline_accs.append(res["c_only"]["best_val_acc"])
        p2_hot_accs.append(res["hot"]["best_val_acc"])
        p2_A.append(res["hot"]["A_mean"])
        p2_B.append(res["hot"]["B_mean"])
        p2_C.append(res["hot"]["C_mean"])

        assert res["hot"]["A_mean"] > 0.01, f"Path A collapsed on seed {seed}"
        assert res["hot"]["B_mean"] > 0.01, f"Path B collapsed on seed {seed}"
        assert res["hot"]["C_mean"] < 0.90, f"Path C failed to reduce attention on seed {seed}"

    p2_base_mean = statistics.mean(p2_baseline_accs)
    p2_hot_mean = statistics.mean(p2_hot_accs)

    print(f"Baseline Acc Mean: {p2_base_mean:.4f}")
    print(f"HoT Acc Mean: {p2_hot_mean:.4f}")
    print(
        "Routing Means -> "
        f"A: {statistics.mean(p2_A):.3f}, "
        f"B: {statistics.mean(p2_B):.3f}, "
        f"C: {statistics.mean(p2_C):.3f}"
    )

    listops_summary = None
    if not args.skip_listops:
        print("\n=== Phase 3: Generalization (LRA ListOps) ===")
        print("  Running with seq_len=512 to keep compute bounded.")
        listops_summary = run_experiment(
            python_exe,
            seed=42,
            dataset="lra_listops",
            seq_len=512,
            mode="both",
        )
        base_acc = listops_summary["c_only"]["best_val_acc"]
        hot_acc = listops_summary["hot"]["best_val_acc"]

        print(f"ListOps Baseline Acc: {base_acc:.4f}")
        print(f"ListOps HoT Acc: {hot_acc:.4f}")
        print(
            "ListOps Routing -> "
            f"A={listops_summary['hot']['A_mean']:.3f}, "
            f"B={listops_summary['hot']['B_mean']:.3f}, "
            f"C={listops_summary['hot']['C_mean']:.3f}"
        )

        assert listops_summary["hot"]["A_mean"] > 0.01
        assert listops_summary["hot"]["B_mean"] > 0.01

    report = ROOT / "validation_results.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# HoT Adaptive Routing Validation Report\n\n")
        f.write("## Phase 1: Reproducibility\n")
        f.write(f"- Accuracies: {p1_results}\n")
        f.write(f"- Standard Deviation: {p1_std:.5f}\n\n")
        f.write("## Phase 2: Robustness\n")
        f.write(f"- Seeds Tested: {seeds}\n")
        f.write(f"- Baseline Mean Acc: {p2_base_mean:.4f}\n")
        f.write(f"- HoT Mean Acc: {p2_hot_mean:.4f}\n")
        f.write(
            f"- Average Routing: A={statistics.mean(p2_A):.3f}, "
            f"B={statistics.mean(p2_B):.3f}, C={statistics.mean(p2_C):.3f}\n\n"
        )
        if listops_summary is not None:
            f.write("## Phase 3: Generalization (LRA ListOps)\n")
            f.write(f"- Baseline Acc: {listops_summary['c_only']['best_val_acc']:.4f}\n")
            f.write(f"- HoT Acc: {listops_summary['hot']['best_val_acc']:.4f}\n")
            f.write(
                f"- Routing: A={listops_summary['hot']['A_mean']:.3f}, "
                f"B={listops_summary['hot']['B_mean']:.3f}, "
                f"C={listops_summary['hot']['C_mean']:.3f}\n\n"
            )
        f.write("## Baseline Parity\n")
        f.write("- C-only and HoT runs use the same data, optimizer, step count, and backbone.\n")

    print(f"\nAll checks passed. Report written to {report}")


if __name__ == "__main__":
    main()
