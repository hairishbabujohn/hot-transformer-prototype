"""
Data utilities for the HoT prototype.

Synthetic tasks (no external downloads required)
-------------------------------------------------
* ``synthetic_bracket``  – Binary bracket-matching task.
* ``synthetic_xor``      – Delayed XOR of two sequence positions.
* ``synthetic_copy``     – Copy-memory task.

External datasets (optional, with graceful fallback)
-----------------------------------------------------
* ``lra_listops``        – LRA ListOps via HuggingFace ``datasets`` library.
                           Falls back to ``synthetic_bracket`` if unavailable.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def bracket_matching_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    seed: int = 42,
) -> tuple:
    """Binary classification: is the bracket sequence balanced?

    Token vocabulary: 0 = '(', 1 = ')'.

    Balanced sequences are generated via the Cycle Lemma (O(N) per sample).
    Unbalanced sequences are random binary strings filtered to exclude any
    accidentally balanced ones.

    Returns:
        (seqs, labels): LongTensors of shape (n_samples, seq_len) and (n_samples,).
    """
    assert seq_len % 2 == 0, "seq_len must be even for bracket matching"
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    n = seq_len // 2

    # --- Balanced sequences (Cycle Lemma) ---
    balanced = []
    for _ in range(half):
        tokens = np.array([0] * n + [1] * n, dtype=np.int64)
        rng.shuffle(tokens)
        steps = 1 - 2 * tokens                     # 0 → +1,  1 → -1
        prefix = np.cumsum(steps)
        min_idx = int(np.argmin(prefix))
        rot = (min_idx + 1) % seq_len
        tokens = np.roll(tokens, -rot)
        balanced.append(tokens)

    # --- Unbalanced sequences (rejection sampling) ---
    def _is_balanced(seq: np.ndarray) -> bool:
        depth = 0
        for t in seq:
            depth += 1 if t == 0 else -1
            if depth < 0:
                return False
        return depth == 0

    unbalanced = []
    while len(unbalanced) < half:
        candidate = rng.integers(0, 2, size=seq_len, dtype=np.int64)
        if not _is_balanced(candidate):
            unbalanced.append(candidate)

    seqs = np.concatenate(
        [np.array(balanced), np.array(unbalanced)], axis=0
    )
    labels = np.concatenate(
        [np.ones(half, dtype=np.int64), np.zeros(half, dtype=np.int64)]
    )

    perm = rng.permutation(n_samples)
    return (
        torch.from_numpy(seqs[perm]),
        torch.from_numpy(labels[perm]),
    )


def delayed_xor_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    delay: int = 32,
    seed: int = 42,
) -> tuple:
    """Binary XOR of tokens at position 0 and position ``delay``.

    Returns:
        (seqs, labels): LongTensors of shape (n_samples, seq_len) and (n_samples,).
    """
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 2, (n_samples, seq_len), generator=g)
    labels = (seq[:, 0] ^ seq[:, delay]).long()
    return seq.long(), labels


def copy_memory_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    vocab_size: int = 16,
    n_classes: int = 8,
    seed: int = 42,
) -> tuple:
    """Copy-memory: recall the token planted at position 0.

    Returns:
        (seqs, labels): LongTensors of shape (n_samples, seq_len) and (n_samples,).
    """
    g = torch.Generator().manual_seed(seed)
    content_ids = torch.randint(0, n_classes, (n_samples,), generator=g)
    filler_id = n_classes
    seq = torch.full((n_samples, seq_len), filler_id, dtype=torch.long)
    seq[:, 0] = content_ids
    seq[:, -1] = vocab_size - 1       # cue marker
    return seq, content_ids


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _make_split(
    seqs: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.8,
) -> tuple:
    n = len(seqs)
    split = int(n * train_frac)
    indices = torch.arange(n)
    return (
        TensorDataset(seqs[:split], labels[:split], indices[:split]),
        TensorDataset(seqs[split:], labels[split:], indices[split:]),
    )


# ---------------------------------------------------------------------------
# Synthetic dataloaders
# ---------------------------------------------------------------------------

def get_synthetic_dataloaders(
    dataset_name: str = "bracket",
    seq_len: int = 64,
    batch_size: int = 32,
    vocab_size: int = 4,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    """Return (train_loader, val_loader, n_classes) for a synthetic task."""
    if "bracket" in dataset_name:
        seqs, labels = bracket_matching_dataset(
            n_samples=4000, seq_len=seq_len, seed=seed,
        )
        n_classes = 2
    elif "xor" in dataset_name:
        seqs, labels = delayed_xor_dataset(
            n_samples=4000, seq_len=seq_len, seed=seed,
        )
        n_classes = 2
    else:
        seqs, labels = copy_memory_dataset(
            n_samples=4000, seq_len=seq_len,
            vocab_size=vocab_size, n_classes=n_classes, seed=seed,
        )

    train_ds, val_ds = _make_split(seqs, labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_classes


# ---------------------------------------------------------------------------
# LRA ListOps (optional)
# ---------------------------------------------------------------------------

def get_lra_listops_dataloaders(
    seq_len: int = 2048,
    batch_size: int = 32,
    vocab_size: int = 32,
) -> tuple:
    """Load LRA ListOps via HuggingFace datasets.

    Raises:
        Any exception from ``datasets.load_dataset`` if unavailable.
    """
    from torch.utils.data import Dataset as TorchDataset
    from datasets import load_dataset  # type: ignore

    class _ListOpsDS(TorchDataset):
        def __init__(self, hf_ds, seq_len: int, vocab_size: int):
            self.data = hf_ds
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple:
            item = self.data[idx]
            ids = item["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            n = ids.shape[0]
            if n >= self.seq_len:
                ids = ids[: self.seq_len]
            else:
                ids = torch.cat([ids, torch.zeros(self.seq_len - n, dtype=torch.long)])
            ids = ids.clamp(0, self.vocab_size - 1)
            return ids, torch.tensor(item["label"], dtype=torch.long), idx

    ds = load_dataset("hf-internal-testing/long-range-arena", "listops")
    train_ds = _ListOpsDS(ds["train"], seq_len, vocab_size)
    val_ds = _ListOpsDS(ds["validation"], seq_len, vocab_size)
    n_classes = 10
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_classes


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> tuple:
    """Build dataloaders from a config dict.

    Expected keys:
        dataset    (str)  – task name, e.g. ``"synthetic_bracket"``.
        seq_len    (int)  – sequence length.
        batch_size (int)  – mini-batch size.
        vocab_size (int)  – vocabulary size.
        n_classes  (int)  – number of output classes.
        seed       (int)  – random seed.

    Returns:
        (train_loader, val_loader, n_classes)
    """
    dataset_name = cfg.get("dataset", "synthetic_bracket")
    seq_len = cfg.get("seq_len", 64)
    batch_size = cfg.get("batch_size", 32)
    vocab_size = cfg.get("vocab_size", 4)
    n_classes = cfg.get("n_classes", 2)
    seed = cfg.get("seed", 42)

    if dataset_name == "lra_listops":
        try:
            return get_lra_listops_dataloaders(
                seq_len=seq_len,
                batch_size=batch_size,
                vocab_size=vocab_size,
            )
        except Exception as exc:
            print(
                f"[data] LRA ListOps load failed ({exc}), "
                "falling back to synthetic bracket."
            )
            dataset_name = "synthetic_bracket"
            n_classes = 2

    return get_synthetic_dataloaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        n_classes=n_classes,
        seed=seed,
    )
