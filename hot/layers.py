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
        


        # Path B
        self.path_b = DepthwiseSepConv1d(d_model, conv_kernel_size)

        # Path C
        self.path_c_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Pathway Merger: learned scalar alpha (initialized 0.15)
        self.alpha = nn.Parameter(torch.ones(1) * 0.15)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        prev_route: torch.Tensor,
        H_low: float,
        H_high: float,
        e_min: float,
        e_max: float,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        B = x.shape[0]
        x_norm = self.norm_pre(x)

        x_c = x_norm - x_norm.mean(dim=-1, keepdim=True)
        x_s = x_c / (x_c.std(dim=-1, keepdim=True) + 1e-6)
        scores = (x_s ** 2).mean(dim=-1)                 # (B, T)

        import math
        probs = torch.softmax(scores, dim=-1)
        raw_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # (B,)
        raw_entropy = raw_entropy / math.log(scores.size(-1))
        
        entropy_norm = (raw_entropy - e_min) / (e_max - e_min + 1e-6)
        entropy_norm = entropy_norm.clamp(0.0, 1.0)

        if force_c:
            gA = torch.zeros(B, dtype=torch.bool, device=x.device)
            gB = torch.zeros(B, dtype=torch.bool, device=x.device)
            gC = torch.ones(B, dtype=torch.bool, device=x.device)
            current_route = torch.full((B,), 2, dtype=torch.long, device=x.device)
        else:
            delta = 0.02
            stay_A = (prev_route == 0) & (entropy_norm < H_low + delta)
            stay_C = (prev_route == 2) & (entropy_norm > H_high - delta)

            new_A = entropy_norm < (H_low - delta)
            new_C = entropy_norm > (H_high + delta)

            gA = stay_A | new_A
            gC = stay_C | new_C
            gB = ~(gA | gC)
            
            current_route = torch.zeros(B, dtype=torch.long, device=x.device)
            current_route[gB] = 1
            current_route[gC] = 2

        mA = gA.float().view(-1, 1, 1)
        mB = gB.float().view(-1, 1, 1)
        mC = gC.float().view(-1, 1, 1)

        g = torch.stack([gA.float(), gB.float(), gC.float()], dim=-1)

        outA = x
        outB = self.path_b(x_norm)
        scale = x.norm(dim=(1, 2), keepdim=True) / (outB.norm(dim=(1, 2), keepdim=True) + 1e-6)
        outB = outB * scale
        outC, _ = self.path_c_attn(x_norm, x_norm, x_norm)

        path_out = mA * outA + mB * outB + mC * outC
        x_next = x + self.alpha * path_out

        if not return_diagnostics:
            return x_next, entropy_norm, current_route, g

        g_mean = g.mean(dim=0).detach() # (3,)
        
        with torch.no_grad():
            y_full = x + self.alpha * (gA.float().view(-1,1,1) * outA + gB.float().view(-1,1,1) * outB + gC.float().view(-1,1,1) * outC)
            y_no_a = x + self.alpha * (gB.float().view(-1,1,1) * outB + gC.float().view(-1,1,1) * outC)
            y_no_b = x + self.alpha * (gA.float().view(-1,1,1) * outA + gC.float().view(-1,1,1) * outC)
            y_no_c = x + self.alpha * (gA.float().view(-1,1,1) * outA + gB.float().view(-1,1,1) * outB)

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

            diagnostics = {
                "x_block_in": x.detach(),
                "alpha": self.alpha.detach(),
                "g": g_mean,
                "A": outA.detach(),
                "B": outB.detach(),
                "C": outC.detach(),
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
                "route_change_rate": (current_route != prev_route).float().mean(),
                "gate_entropy": torch.tensor(0.0),
            }

        return x_next, entropy_norm, current_route, g, diagnostics
