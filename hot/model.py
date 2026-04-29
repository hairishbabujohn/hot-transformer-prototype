"""
HoTEncoder: encoder-only sequence classifier built from HoT layers.
"""

import torch
import torch.nn as nn

from .layers import HoTLayer


class HoTEncoder(nn.Module):
    """Encoder-only classifier using Homeostatic Transformer layers.

    Architecture::

        Token Embedding + Positional Embedding
              ↓   (dropout)
        N × HoTLayer
              ↓
        LayerNorm → mean-pool over sequence → Linear classifier

    Args:
        vocab_size:       Size of token vocabulary.
        d_model:          Hidden dimension.
        n_layers:         Number of HoT layers.
        n_heads:          Attention heads (used by Path C in each layer).
        n_classes:        Number of output classes.
        max_seq_len:      Maximum sequence length (for positional embedding).
        conv_kernel_size: Kernel size for Path B conv (default 7).
        dropout:          Embedding dropout and attention dropout (default 0.1).
        gate_temperature: Sigmoid sharpness for soft routing (default 0.05).
    """

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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Input embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # HoT layers
        self.layers = nn.ModuleList([
            HoTLayer(d_model, n_heads, conv_kernel_size, dropout, gate_temperature)
            for _ in range(n_layers)
        ])

        # Output head
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
        thresholds: list,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Run the encoder and return classification logits.

        Args:
            x:          Token ID tensor of shape (B, N).
            thresholds: List of ``(H_low, H_high)`` tuples, one per layer.
                        Typically returned by ``CZU.get_all_thresholds()``.
            force_c:    If True, force all layers to use Path C (CZU warmup).
            return_diagnostics: If True, return per-layer diagnostics.

        Returns:
            logits:    (B, n_classes) classification logits.
            oem_vals:  List of scalar OEM tensors, one per layer.
            routes:    List of route info per layer.
                       During training: Tensor [w_A, w_B, w_C] (soft weights).
                       During eval / force_c: Integer path index (0 / 1 / 2).
            diagnostics: Optional list of per-layer diagnostics dicts.
        """
        B, N = x.shape
        device = x.device

        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))

        oem_vals = []
        routes = []
        diagnostics = []

        for i, layer in enumerate(self.layers):
            H_low, H_high = thresholds[i]
            if return_diagnostics:
                h, oem_val, route_info, diag = layer(
                    h, H_low, H_high, force_c=force_c, return_diagnostics=True,
                )
                diagnostics.append(diag)
            else:
                h, oem_val, route_info = layer(h, H_low, H_high, force_c=force_c)
            oem_vals.append(oem_val)
            routes.append(route_info)

        # Mean pooling over sequence dimension
        pooled = self.norm_out(h).mean(dim=1)   # (B, D)
        logits = self.classifier(pooled)
        if return_diagnostics:
            return logits, oem_vals, routes, diagnostics
        return logits, oem_vals, routes
