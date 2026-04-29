"""Encoder-only classifier built from Homeostatic Transformer layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .czu import CZU
from .layers import HoTLayer


class HoTEncoder(nn.Module):
    """Encoder-only sequence classifier using stacked HoT layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_classes: int,
        max_seq_len: int,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
        gate_temperature: float = 0.05,
        czu_warmup_steps: int = 1000,
        czu_update_every: int = 500,
        czu_ema_beta: float = 0.95,
        dataset_size: int | None = None,
    ) -> None:
        super().__init__()
        del dataset_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                HoTLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    gate_temperature=gate_temperature,
                )
                for _ in range(n_layers)
            ]
        )

        self.czu = CZU(
            n_layers=n_layers,
            warmup_steps=czu_warmup_steps,
            update_every=czu_update_every,
            ema_beta=czu_ema_beta,
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        sample_ids: torch.Tensor | None = None,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Run the encoder.

        ``sample_ids`` is accepted for backward compatibility with older
        dataloaders, but routing is now purely per-layer/per-sample as described
        in the patent.
        """
        del sample_ids

        batch_size, seq_len = x.shape
        device = x.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))

        routes = []
        entropies = []
        diagnostics = []

        for layer_idx, layer in enumerate(self.layers):
            H_low, H_high = self.czu.get_thresholds(layer_idx)
            if return_diagnostics:
                h, entropy, route_idx, route_info, diag = layer(
                    h,
                    H_low=H_low,
                    H_high=H_high,
                    force_c=force_c,
                    return_diagnostics=True,
                )
                diagnostics.append(diag)
            else:
                h, entropy, route_idx, route_info = layer(
                    h,
                    H_low=H_low,
                    H_high=H_high,
                    force_c=force_c,
                )

            del route_idx
            routes.append(route_info)
            entropies.append(entropy)

        pooled = self.norm_out(h).mean(dim=1)
        logits = self.classifier(pooled)
        if return_diagnostics:
            return logits, routes, entropies, diagnostics
        return logits, routes, entropies
