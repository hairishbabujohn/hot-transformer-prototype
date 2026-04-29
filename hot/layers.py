"""
Core layers for the Homeostatic Transformer (HoT).

Components
----------
compute_oem         -- Output Entropy Monitor: normalized Shannon entropy of a layer output.
DepthwiseSepConv1d  -- Path B operator: depthwise-separable 1-D convolution over sequence.
TrainableGate       -- Local routing gate based on feature statistics.
HoTLayer            -- Full HoT block (pre-norm, Gate routing, PM merge).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output Entropy Monitor (OEM)
# ---------------------------------------------------------------------------

def compute_oem(x: torch.Tensor) -> torch.Tensor:
    """Compute normalized Shannon entropy of the layer output per sequence.

    A softmax distribution is built by averaging the representation over the
    sequence dimension, then treating the D-dimensional vector as logits.
    Entropy is normalized by log(D) so the result lies in [0, 1].

    Args:
        x: Tensor of shape (B, N, D).

    Returns:
        Tensor of shape (B,) -- normalized entropy across the batch.
    """
    B, N, D = x.shape
    # Average over sequence positions -> (B, D) "summary" logits
    x_mean = x.mean(dim=1)

    # Numerically stable log-probabilities via log-softmax
    log_probs = F.log_softmax(x_mean, dim=-1)      # (B, D)
    probs = log_probs.exp()                         # (B, D)

    # Shannon entropy: H = -sum(p * log(p)); use log_probs for numerical safety
    entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)  # (B,)

    # Normalize to [0, 1]
    normalized = entropy / math.log(max(D, 2))     # (B,)
    return normalized


# ---------------------------------------------------------------------------
# Path B: Depthwise-Separable 1-D Convolution
# ---------------------------------------------------------------------------

class DepthwiseSepConv1d(nn.Module):
    """Depthwise-separable 1-D convolution along the sequence dimension with expansion.

    Complexity: O(N · k · D) per forward pass.

    Args:
        d_model:     Feature dimension.
        kernel_size: Convolution kernel size (default 7).
        expand_ratio: Expansion ratio for 1x1 conv (default 2).
    """

    def __init__(self, d_model: int, kernel_size: int = 7, expand_ratio: int = 4) -> None:
        super().__init__()
        pad = kernel_size // 2
        expanded_dim = d_model * expand_ratio
        
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=pad, groups=d_model, bias=False,
        )
        self.act1 = nn.GELU()
        self.expand = nn.Conv1d(d_model, expanded_dim, 1, bias=False)
        self.act2 = nn.GELU()
        self.proj = nn.Conv1d(expanded_dim, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        out = x.transpose(1, 2)          # (B, D, N)  -- Conv1d convention
        out = self.act1(self.dw(out))    # depthwise
        out = self.act2(self.expand(out)) # expansion
        out = self.proj(out)              # projection
        out = out.transpose(1, 2)        # (B, N, D)
        out = self.norm(out)              # layernorm
        return out


# ---------------------------------------------------------------------------
# Trainable Gate
# ---------------------------------------------------------------------------

class TrainableGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(5)
        self.mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 3)
        )
        # Initialize gate neutrally: final weights/bias = 0
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor, oem_val: torch.Tensor) -> torch.Tensor:
        """
        Extract features and compute gate logits.
        Args:
            x: (B, N, D)
            oem_val: (B,)
        Returns:
            logits: (B, 3)
        """
        B, N, D = x.shape
        x_mean = x.mean(dim=(1, 2))  # (B,)
        x_var = x.var(dim=(1, 2), unbiased=False)  # (B,)
        
        # Norms
        x_norm = torch.linalg.norm(x.reshape(B, -1), dim=-1) # (B,)
        
        # Temporal difference
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        t_diff = torch.linalg.norm(x_diff.reshape(B, -1), dim=-1) # (B,)
        
        features = torch.stack([x_mean, x_var, x_norm, t_diff, oem_val], dim=-1) # (B, 5)
        features = self.norm(features)
        
        logits = self.mlp(features) # (B, 3)
        return logits


# ---------------------------------------------------------------------------
# HoT Layer
# ---------------------------------------------------------------------------

class HoTLayer(nn.Module):
    """Homeostatic Transformer Layer.

    Pre-norm block with three selectable paths per forward call:

    * **Path A** – Residual passthrough
    * **Path B** – Depthwise-separable 1-D conv with expansion
    * **Path C** – Full multi-head self-attention

    The **Trainable Gate** selects a path based on local features.

    Gradient flow
    ~~~~~~~~~~~~~
    * Soft (differentiable) routing: all three paths are computed and blended
      with softmax weights.

    The **Pathway Merger (PM)** combines the selected path output with the
    residual using a learned per-layer scalar ``alpha``:

        x_next = x + alpha * path_output

    Args:
        d_model:          Model hidden dimension.
        n_heads:          Number of attention heads for Path C.
        conv_kernel_size: Kernel size for Path B conv (default 7).
        dropout:          Dropout applied inside Path C attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Pre-norm applied before routing
        self.norm_pre = nn.LayerNorm(d_model)
        
        # Gate
        self.gate = TrainableGate()

        # Path B
        self.path_b = DepthwiseSepConv1d(d_model, conv_kernel_size)

        # Path C
        self.path_c_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Pathway Merger: learned scalar alpha (initialized 0.25)
        self.alpha = nn.Parameter(torch.ones(1) * 0.25)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Run one HoT layer.

        Args:
            x:       Input tensor (B, N, D).
            tau:     Temperature for softmax gate.

        Returns:
            x_next:     Output tensor (B, N, D).
            oem_val:    Scalar entropy tensor.
            route_info: Tensor [B, 3] of soft gate weights.
            diagnostics: Optional dict of diagnostics.
        """
        B = x.shape[0]
        # --- Pre-norm ---
        x_norm = self.norm_pre(x)

        # --- Output Entropy Monitor ---
        oem_val = compute_oem(x_norm) # (B,)

        # --- Homeostatic Gate ---
        gate_logits = self.gate(x_norm, oem_val) # (B, 3)
        if force_c:
            g = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
        else:
            g = torch.softmax(gate_logits / tau, dim=-1) # (B, 3)
            g = g.clamp(min=1e-6)
            g = g / g.sum(dim=-1, keepdim=True)
        
        gA = g[:, 0].view(-1, 1, 1)
        gB = g[:, 1].view(-1, 1, 1)
        gC = g[:, 2].view(-1, 1, 1)

        # Compute all paths (soft routing)
        path_a_out = x
        
        path_b_out = self.path_b(x_norm)
        scale = x.norm(dim=(1, 2), keepdim=True) / (path_b_out.norm(dim=(1, 2), keepdim=True) + 1e-6)
        path_b_out = path_b_out * scale
        
        path_c_out, _ = self.path_c_attn(x_norm, x_norm, x_norm)

        path_out = gA * path_a_out + gB * path_b_out + gC * path_c_out

        # --- Pathway Merger: x_next = x + alpha * path_output ---
        x_next = x + self.alpha * path_out

        if not return_diagnostics:
            return x_next, oem_val.mean(), g

        # Note: Do not detach g here, we want diagnostics/metrics to be computable
        g_mean = g.mean(dim=0).detach() # (3,)
        
        with torch.no_grad():
            y_full = x + self.alpha * (gA * path_a_out + gB * path_b_out + gC * path_c_out)
            y_no_a = x + self.alpha * (gB * path_b_out + gC * path_c_out)
            y_no_b = x + self.alpha * (gA * path_a_out + gC * path_c_out)
            y_no_c = x + self.alpha * (gA * path_a_out + gB * path_b_out)

            def mean_norm(t: torch.Tensor) -> torch.Tensor:
                return torch.linalg.norm(t, dim=(1, 2)).mean()

            a_impact = mean_norm(y_full - y_no_a)
            b_impact = mean_norm(y_full - y_no_b)
            c_impact = mean_norm(y_full - y_no_c)
            x_norm_mag = mean_norm(x)

            denom = x_norm_mag + 1e-9
            a_ratio = a_impact / denom
            b_ratio = b_impact / denom
            c_ratio = c_impact / (a_impact + b_impact + c_impact + 1e-9)

            # Gate entropy computed on g before clamp/detach? g is already clamped.
            gate_entropy = -(g * (g + 1e-9).log()).sum(dim=-1).mean()

            diagnostics = {
                "x_block_in": x.detach(),
                "alpha": self.alpha.detach(),
                "g": g_mean,
                "A": path_a_out.detach(),
                "B": path_b_out.detach(),
                "C": path_c_out.detach(),
                "y_full": y_full.detach(),
                "y_noA": y_no_a.detach(),
                "y_noB": y_no_b.detach(),
                "y_noC": y_no_c.detach(),
                "A_impact": a_impact.detach(),
                "B_impact": b_impact.detach(),
                "C_impact": c_impact.detach(),
                "x_norm_mag": x_norm_mag.detach(),
                "A_ratio": a_ratio.detach(),
                "B_ratio": b_ratio.detach(),
                "C_ratio": c_ratio.detach(),
                "gate_entropy": gate_entropy.detach(),
            }

        return x_next, oem_val.mean(), g, diagnostics
