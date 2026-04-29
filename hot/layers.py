"""
Core layers for the Homeostatic Transformer (HoT).

The implementation follows the patent disclosure:

* OEM computes normalized Shannon entropy over the layer representation.
* HG compares entropy with per-layer comfort-zone thresholds.
* Path A is residual bypass, Path B is depthwise-separable local convolution,
  and Path C is full multi-head self-attention.
* PM applies LayerNorm(x + alpha * PathOutput), with alpha initialized to 1.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_oem(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute the normalized Output Entropy Monitor value.

    Args:
        x: Tensor of shape (B, N, D), where N is sequence length and D is the
           feature dimension.
        eps: Numerical stability constant.

    Returns:
        A tensor of shape (B,) containing mean normalized Shannon entropy over
        the sequence. Values are clamped to [0, 1].
    """
    if x.dim() != 3:
        raise ValueError(f"compute_oem expects (B, N, D), got shape {tuple(x.shape)}")

    d_model = x.size(-1)
    if d_model <= 1:
        return torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

    log_probs = torch.log_softmax(x, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy = entropy / math.log(d_model)
    return entropy.mean(dim=1).clamp(0.0, 1.0)


class DepthwiseSepConv1d(nn.Module):
    """Depthwise-separable 1-D convolution along the sequence dimension."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=0,
            groups=d_model,
            bias=False,
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        pad_total = self.kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        out = F.pad(out, (pad_left, pad_right))
        out = self.depthwise(out)
        out = self.activation(out)
        out = self.pointwise(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class HoTLayer(nn.Module):
    """Homeostatic Transformer layer with OEM, HG, three paths, and PM."""

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

        self.norm_pre = nn.LayerNorm(d_model)
        self.path_b = DepthwiseSepConv1d(d_model, conv_kernel_size, dropout)
        self.path_c_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.norm_pm = nn.LayerNorm(d_model)

    def _hard_routes(
        self,
        entropy: torch.Tensor,
        H_low: float,
        H_high: float,
        force_c: bool,
    ) -> torch.Tensor:
        if force_c:
            route_idx = torch.full_like(entropy, 2, dtype=torch.long)
        else:
            route_idx = torch.ones_like(entropy, dtype=torch.long)
            route_idx = torch.where(entropy < H_low, torch.zeros_like(route_idx), route_idx)
            route_idx = torch.where(entropy > H_high, torch.full_like(route_idx, 2), route_idx)
        return torch.nn.functional.one_hot(route_idx, num_classes=3).to(dtype=entropy.dtype)

    def _straight_through_routes(
        self,
        entropy: torch.Tensor,
        H_low: float,
        H_high: float,
        force_c: bool,
    ) -> torch.Tensor:
        hard = self._hard_routes(entropy, H_low, H_high, force_c)
        if force_c or not self.training:
            return hard

        temp = max(self.gate_temperature, 1e-6)
        below = torch.sigmoid((H_low - entropy) / temp)
        above = torch.sigmoid((entropy - H_high) / temp)
        middle = (1.0 - below) * (1.0 - above)
        soft = torch.stack([below, middle, above], dim=-1)
        soft = soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return hard - soft.detach() + soft

    def _compute_paths(self, x: torch.Tensor, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Path A bypasses sublayer computation; the PM residual term carries x.
        path_a = torch.zeros_like(x)
        path_b = self.path_b(x_norm)
        path_c, _ = self.path_c_attn(x_norm, x_norm, x_norm, need_weights=False)
        return path_a, path_b, path_c

    def _compute_eval_path(
        self,
        x: torch.Tensor,
        x_norm: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        path_out = torch.zeros_like(x)

        b_mask = routes[:, 1].bool()
        if b_mask.any():
            path_out[b_mask] = self.path_b(x_norm[b_mask])

        c_mask = routes[:, 2].bool()
        if c_mask.any():
            path_c, _ = self.path_c_attn(
                x_norm[c_mask],
                x_norm[c_mask],
                x_norm[c_mask],
                need_weights=False,
            )
            path_out[c_mask] = path_c

        return path_out

    def forward(
        self,
        x: torch.Tensor,
        H_low: float,
        H_high: float,
        force_c: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple:
        x_norm = self.norm_pre(x)
        entropy = compute_oem(x_norm)
        routes = self._straight_through_routes(entropy, H_low, H_high, force_c)

        need_all_paths = (self.training and not force_c) or return_diagnostics
        if need_all_paths:
            path_a, path_b, path_c = self._compute_paths(x, x_norm)
            path_out = (
                routes[:, 0].view(-1, 1, 1) * path_a
                + routes[:, 1].view(-1, 1, 1) * path_b
                + routes[:, 2].view(-1, 1, 1) * path_c
            )
        else:
            path_a = path_b = path_c = None
            path_out = self._compute_eval_path(x, x_norm, routes)

        x_next = self.norm_pm(x + self.alpha * path_out)
        route_idx = routes.detach().argmax(dim=-1)

        if not return_diagnostics:
            return x_next, entropy, route_idx, routes

        if path_a is None or path_b is None or path_c is None:
            path_a, path_b, path_c = self._compute_paths(x, x_norm)

        with torch.no_grad():
            hard_routes = self._hard_routes(entropy, H_low, H_high, force_c)
            route_mean = hard_routes.mean(dim=0)
            route_entropy = -(
                route_mean * torch.log(route_mean.clamp_min(1e-9))
            ).sum() / math.log(3)

            y_full = self.norm_pm(x + self.alpha * path_out)
            zero = torch.zeros_like(path_a)
            y_no_a = self.norm_pm(
                x
                + self.alpha
                * (
                    routes[:, 1].view(-1, 1, 1) * path_b
                    + routes[:, 2].view(-1, 1, 1) * path_c
                )
            )
            y_no_b = self.norm_pm(
                x
                + self.alpha
                * (
                    routes[:, 0].view(-1, 1, 1) * zero
                    + routes[:, 2].view(-1, 1, 1) * path_c
                )
            )
            y_no_c = self.norm_pm(
                x
                + self.alpha
                * (
                    routes[:, 0].view(-1, 1, 1) * zero
                    + routes[:, 1].view(-1, 1, 1) * path_b
                )
            )

            def mean_norm(t: torch.Tensor) -> torch.Tensor:
                return torch.linalg.norm(t, dim=(1, 2)).mean()

            a_impact = mean_norm(y_full - y_no_a)
            b_impact = mean_norm(y_full - y_no_b)
            c_impact = mean_norm(y_full - y_no_c)
            x_norm_mag = mean_norm(x)
            total_impact = a_impact + b_impact + c_impact + 1e-9

            diagnostics = {
                "x_block_in": x.detach(),
                "alpha": self.alpha.detach(),
                "g": route_mean.detach(),
                "A": path_a.detach(),
                "B": path_b.detach(),
                "C": path_c.detach(),
                "y_full": y_full.detach(),
                "y_noA": y_no_a.detach(),
                "y_noB": y_no_b.detach(),
                "y_noC": y_no_c.detach(),
                "A_impact": a_impact.detach(),
                "B_impact": b_impact.detach(),
                "C_impact": c_impact.detach(),
                "x_norm_mag": x_norm_mag.detach(),
                "A_ratio": (a_impact / (x_norm_mag + 1e-9)).detach(),
                "B_ratio": (b_impact / (x_norm_mag + 1e-9)).detach(),
                "C_ratio": (c_impact / total_impact).detach(),
                "route_change_rate": torch.tensor(0.0, device=x.device),
                "gate_entropy": route_entropy.detach(),
            }

        return x_next, entropy, route_idx, routes, diagnostics
