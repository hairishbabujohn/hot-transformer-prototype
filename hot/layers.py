"""
Core layers for the Homeostatic Transformer (HoT).

Components
----------
compute_oem         -- Output Entropy Monitor: normalized Shannon entropy of a layer output.
DepthwiseSepConv1d  -- Path B operator: depthwise-separable 1-D convolution over sequence.
HoTLayer            -- Full HoT block (pre-norm, OEM, HG routing, PM merge).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output Entropy Monitor (OEM)
# ---------------------------------------------------------------------------

def compute_oem(x: torch.Tensor) -> torch.Tensor:
    """Compute per-batch normalized Shannon entropy of the layer output.

    A softmax distribution is built by averaging the representation over the
    sequence dimension, then treating the D-dimensional vector as logits.
    Entropy is normalized by log(D) so the result lies in [0, 1].

    Args:
        x: Tensor of shape (B, N, D).

    Returns:
        Scalar tensor -- mean normalized entropy across the batch.
    """
    B, N, D = x.shape
    # Average over sequence positions -> (B, D) "summary" logits
    x_mean = x.mean(dim=1)

    # Numerically stable log-probabilities via log-softmax
    log_probs = F.log_softmax(x_mean, dim=-1)      # (B, D)
    probs = log_probs.exp()                         # (B, D)

    # Shannon entropy: H = -sum(p * log(p)); use log_probs for numerical safety
    # p * log(p) is 0 when p = 0 (in math), but can produce NaN in fp.
    # Replace NaN/inf with 0 via nan_to_num before summing.
    entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)  # (B,)

    # Normalize to [0, 1]
    normalized = entropy / math.log(max(D, 2))     # (B,)
    return normalized.mean()                        # scalar


# ---------------------------------------------------------------------------
# Path B: Depthwise-Separable 1-D Convolution
# ---------------------------------------------------------------------------

class DepthwiseSepConv1d(nn.Module):
    """Depthwise-separable 1-D convolution along the sequence dimension.

    Complexity: O(N · k · D) per forward pass.

    Args:
        d_model:     Feature dimension.
        kernel_size: Convolution kernel size (default 7).
    """

    def __init__(self, d_model: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        # Depthwise: one independent filter per channel
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=pad, groups=d_model, bias=False,
        )
        # Pointwise: 1×1 cross-channel mixing
        self.pw = nn.Conv1d(d_model, d_model, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        x = x.transpose(1, 2)          # (B, D, N)  -- Conv1d convention
        x = self.act(self.dw(x))        # depthwise
        x = self.pw(x)                  # pointwise
        return x.transpose(1, 2)        # (B, N, D)


# ---------------------------------------------------------------------------
# HoT Layer
# ---------------------------------------------------------------------------

class HoTLayer(nn.Module):
    """Homeostatic Transformer Layer.

    Pre-norm block with three selectable paths per forward call:

    * **Path A** – Residual passthrough: path output = zeros  (O(1))
    * **Path B** – Depthwise-separable 1-D conv                (O(N·k))
    * **Path C** – Full multi-head self-attention               (O(N²))

    The **Homeostatic Gate (HG)** selects a path based on the OEM entropy
    value compared to learned thresholds maintained by the CZU.

    Gradient flow
    ~~~~~~~~~~~~~
    * **Training** – soft (differentiable) routing: all three paths are
      computed and blended with sigmoid-derived weights, allowing gradients
      to flow back through both the gating signal and the path outputs.
    * **Eval** – hard (discrete) routing: only the selected path is executed.

    The **Pathway Merger (PM)** combines the selected path output with the
    residual using a learned per-layer scalar ``alpha``:

        x_next = LayerNorm(x + alpha * path_output)

    Args:
        d_model:          Model hidden dimension.
        n_heads:          Number of attention heads for Path C.
        conv_kernel_size: Kernel size for Path B conv (default 7).
        dropout:          Dropout applied inside Path C attention.
        gate_temperature: Sigmoid temperature for soft routing (default 0.05).
    """

    PATH_A = 0   # residual passthrough
    PATH_B = 1   # depthwise-separable conv
    PATH_C = 2   # full self-attention

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
        gate_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.gate_temperature = gate_temperature

        # Pre-norm applied before routing
        self.norm_pre = nn.LayerNorm(d_model)

        # Path B
        self.path_b = DepthwiseSepConv1d(d_model, conv_kernel_size)

        # Path C
        self.path_c_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Pathway Merger: learned scalar alpha (initialized small)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.norm_merge = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _soft_route(
        self,
        x_norm: torch.Tensor,
        h: torch.Tensor,
        H_low: float,
        H_high: float,
    ) -> tuple:
        """Differentiable soft routing used during training.

        Computes all three path outputs and returns a weighted blend so that
        gradients can flow back through both the path computations and the
        entropy-based gating signal.

        Returns:
            path_out: (B, N, D) blended path output.
            weights:  1-D Tensor [w_A, w_B, w_C] for logging.
        """
        temp = self.gate_temperature
        H_low_t = torch.as_tensor(H_low, dtype=h.dtype, device=h.device)
        H_high_t = torch.as_tensor(H_high, dtype=h.dtype, device=h.device)

        # Soft gates derived from the entropy scalar
        w_a = torch.sigmoid((H_low_t - h) / temp)    # peaks when h << H_low
        w_c = torch.sigmoid((h - H_high_t) / temp)   # peaks when h >> H_high
        w_b = (1.0 - w_a - w_c).clamp(min=0.0)

        # Normalize so weights sum to 1
        total = w_a + w_b + w_c
        w_a, w_b, w_c = w_a / total, w_b / total, w_c / total

        # Compute all paths (required so gradients flow through all branches)
        path_a_out = torch.zeros_like(x_norm)
        path_b_out = self.path_b(x_norm)
        path_c_out, _ = self.path_c_attn(x_norm, x_norm, x_norm)

        path_out = w_a * path_a_out + w_b * path_b_out + w_c * path_c_out
        weights = torch.stack([w_a.detach(), w_b.detach(), w_c.detach()])
        return path_out, weights

    def _hard_route(
        self,
        x_norm: torch.Tensor,
        h_val: float,
        H_low: float,
        H_high: float,
    ) -> tuple:
        """Discrete single-path routing used during evaluation.

        Returns:
            path_out: (B, N, D) output of the selected path.
            route:    Integer path index (PATH_A / PATH_B / PATH_C).
        """
        if h_val < H_low:
            return torch.zeros_like(x_norm), self.PATH_A
        elif h_val > H_high:
            path_out, _ = self.path_c_attn(x_norm, x_norm, x_norm)
            return path_out, self.PATH_C
        else:
            return self.path_b(x_norm), self.PATH_B

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        H_low: float,
        H_high: float,
        force_c: bool = False,
    ) -> tuple:
        """Run one HoT layer.

        Args:
            x:       Input tensor (B, N, D).
            H_low:   Lower entropy threshold for this layer.
            H_high:  Upper entropy threshold for this layer.
            force_c: When True (CZU warmup), always use Path C regardless of entropy.

        Returns:
            x_next:     Output tensor (B, N, D).
            oem_val:    Scalar entropy tensor (for CZU update).
            route_info: During training (soft) – Tensor [w_A, w_B, w_C].
                        During eval / force_c – Integer path index (0/1/2).
        """
        # --- Pre-norm ---
        x_norm = self.norm_pre(x)

        # --- Output Entropy Monitor ---
        oem_val = compute_oem(x_norm)

        # --- Homeostatic Gate ---
        if force_c:
            path_out, _ = self.path_c_attn(x_norm, x_norm, x_norm)
            route_info = self.PATH_C
        elif self.training:
            path_out, route_info = self._soft_route(x_norm, oem_val, H_low, H_high)
        else:
            path_out, route_info = self._hard_route(
                x_norm, oem_val.item(), H_low, H_high,
            )

        # --- Pathway Merger: x_next = LN(x + alpha * path_output) ---
        x_next = self.norm_merge(x + self.alpha * path_out)
        return x_next, oem_val, route_info
