"""
HoTEncoder: encoder-only sequence classifier built from HoT layers.
"""

import torch
import torch.nn as nn

from .layers import HoTLayer
from .czu import CZU


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
        dataset_size: int = 100000,
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
            HoTLayer(d_model, n_heads, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])
        
        self.czu = CZU(n_layers, warmup_steps=500)
        self.route_memory = {
            i: torch.full((dataset_size,), 2, dtype=torch.long)
            for i in range(n_layers)
        }

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
        sample_ids: torch.Tensor,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Run the encoder and return classification logits.

        Args:
            x:          Token ID tensor of shape (B, N).
            sample_ids: Global sample identity tensor of shape (B,).
            force_c:    If True, bypasses routing and uses only C-path.
            return_diagnostics: If True, return per-layer diagnostics.

        Returns:
            logits:    (B, n_classes) classification logits.
            routes:    List of soft routing weights per layer (B, 3).
            entropies: List of per-layer entropy tensors (B,).
            diagnostics: Optional list of per-layer diagnostics dicts.
        """
        B, N = x.shape
        device = x.device
        
        # move memory to device lazily
        if self.route_memory[0].device != device:
            for i in range(self.n_layers):
                self.route_memory[i] = self.route_memory[i].to(device)

        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))

        routes = []
        entropies = []
        diagnostics = []

        for i, layer in enumerate(self.layers):
            prev_route = self.route_memory[i][sample_ids]
            H_low, H_high, e_min, e_max = self.czu.get_thresholds(i)

            if return_diagnostics:
                h, entropy_norm, current_route, route_info, diag = layer(
                    h, prev_route, H_low, H_high, e_min, e_max, force_c=force_c, return_diagnostics=True,
                )
                diagnostics.append(diag)
            else:
                h, entropy_norm, current_route, route_info = layer(
                    h, prev_route, H_low, H_high, e_min, e_max, force_c=force_c
                )
            
            self.route_memory[i][sample_ids] = current_route.detach()
            
            routes.append(route_info)
            entropies.append(entropy_norm)

        # Mean pooling over sequence dimension
        pooled = self.norm_out(h).mean(dim=1)   # (B, D)
        logits = self.classifier(pooled)
        if return_diagnostics:
            return logits, routes, entropies, diagnostics
        return logits, routes, entropies
