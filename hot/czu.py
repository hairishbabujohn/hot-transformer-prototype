"""Comfort Zone Updater for Homeostatic Transformer layers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class CZU:
    """Maintain per-layer entropy thresholds using warmup percentiles and EMA."""

    def __init__(
        self,
        n_layers: int,
        warmup_steps: int = 1000,
        update_every: int = 500,
        ema_beta: float = 0.95,
        init_H_low: float = 0.3,
        init_H_high: float = 0.7,
        min_threshold_gap: float = 0.0,
        max_buffer_size: int = 20000,
    ) -> None:
        self.n_layers = n_layers
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        self.ema_beta = ema_beta
        self.min_threshold_gap = min_threshold_gap
        self.max_buffer_size = max_buffer_size

        self.step = 0
        self.initialized = False

        self.H_low = [float(init_H_low)] * n_layers
        self.H_high = [float(init_H_high)] * n_layers

        self._warmup_buf: list[list[float]] = [[] for _ in range(n_layers)]
        self._recent_buf: list[list[float]] = [[] for _ in range(n_layers)]

    @property
    def in_warmup(self) -> bool:
        return self.step < self.warmup_steps

    def force_path_c(self) -> bool:
        return self.in_warmup

    def get_thresholds(self, layer_idx: int) -> tuple[float, float]:
        if not self.in_warmup and not self.initialized:
            self._init_from_warmup()
            self.initialized = True
        return self.H_low[layer_idx], self.H_high[layer_idx]

    def get_all_thresholds(self) -> list[tuple[float, float]]:
        return [self.get_thresholds(i) for i in range(self.n_layers)]

    def update(self, layer_entropies: Iterable) -> None:
        """Record entropy observations and update thresholds when due."""
        entropy_values = [self._to_float_list(v) for v in layer_entropies]

        if self.in_warmup:
            for i, values in enumerate(entropy_values):
                self._extend_bounded(self._warmup_buf[i], values)
        else:
            if not self.initialized:
                self._init_from_warmup()
                self.initialized = True

            for i, values in enumerate(entropy_values):
                self._extend_bounded(self._recent_buf[i], values)

            steps_after = self.step - self.warmup_steps + 1
            if self.update_every > 0 and steps_after % self.update_every == 0:
                self._ema_update_from_recent()

        self.step += 1

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "initialized": self.initialized,
            "H_low": list(self.H_low),
            "H_high": list(self.H_high),
            "warmup_steps": self.warmup_steps,
            "update_every": self.update_every,
            "ema_beta": self.ema_beta,
            "min_threshold_gap": self.min_threshold_gap,
            "_warmup_buf": [list(buf) for buf in self._warmup_buf],
            "_recent_buf": [list(buf) for buf in self._recent_buf],
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = int(state.get("step", 0))
        self.initialized = bool(state.get("initialized", False))
        self.H_low = [float(v) for v in state.get("H_low", self.H_low)]
        self.H_high = [float(v) for v in state.get("H_high", self.H_high)]
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.update_every = int(state.get("update_every", self.update_every))
        self.ema_beta = float(state.get("ema_beta", self.ema_beta))
        self.min_threshold_gap = float(state.get("min_threshold_gap", self.min_threshold_gap))

        warmup_buf = state.get("_warmup_buf")
        recent_buf = state.get("_recent_buf")
        if warmup_buf is not None:
            self._warmup_buf = [[float(v) for v in buf] for buf in warmup_buf]
        if recent_buf is not None:
            self._recent_buf = [[float(v) for v in buf] for buf in recent_buf]

        self._enforce_constraints()

    def _to_float_list(self, values) -> list[float]:
        if hasattr(values, "detach"):
            values = values.detach().flatten().cpu().tolist()
        elif hasattr(values, "flatten"):
            values = values.flatten().tolist()
        elif isinstance(values, (float, int)):
            values = [values]
        else:
            values = list(values)
        return [float(v) for v in values]

    def _extend_bounded(self, buf: list[float], values: list[float]) -> None:
        buf.extend(values)
        if len(buf) > self.max_buffer_size:
            del buf[: len(buf) - self.max_buffer_size]

    def _init_from_warmup(self) -> None:
        for i, buf in enumerate(self._warmup_buf):
            if len(buf) == 0:
                continue
            arr = np.asarray(buf, dtype=np.float64)
            self.H_low[i] = float(np.percentile(arr, 10))
            self.H_high[i] = float(np.percentile(arr, 90))
        self._enforce_constraints()

    def _ema_update_from_recent(self) -> None:
        for i, buf in enumerate(self._recent_buf):
            if len(buf) == 0:
                continue
            arr = np.asarray(buf, dtype=np.float64)
            p10 = float(np.percentile(arr, 10))
            p90 = float(np.percentile(arr, 90))
            beta = self.ema_beta
            self.H_low[i] = beta * self.H_low[i] + (1.0 - beta) * p10
            self.H_high[i] = beta * self.H_high[i] + (1.0 - beta) * p90
            buf.clear()
        self._enforce_constraints()

    def _enforce_constraints(self) -> None:
        for i in range(self.n_layers):
            low = min(max(float(self.H_low[i]), 0.0), 1.0)
            high = min(max(float(self.H_high[i]), 0.0), 1.0)

            if high < low:
                low, high = high, low

            if high - low < self.min_threshold_gap:
                mid = 0.5 * (low + high)
                low = mid - 0.5 * self.min_threshold_gap
                high = mid + 0.5 * self.min_threshold_gap

            if low < 0.0:
                high = min(1.0, high - low)
                low = 0.0
            if high > 1.0:
                low = max(0.0, low - (high - 1.0))
                high = 1.0

            self.H_low[i] = low
            self.H_high[i] = high
