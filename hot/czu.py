"""
Comfort Zone Updater (CZU) for the Homeostatic Transformer.

The CZU manages per-layer entropy thresholds (H_low, H_high) used by the
Homeostatic Gate to select a processing path (A / B / C).

Lifecycle
---------
1. **Warmup** (step < warmup_steps):
   All layers are forced to Path C (full attention).  Entropy values are
   collected into a per-layer buffer.

2. **Initialization** (first step after warmup):
   H_low  = P10 of warmup entropies  (clamped to ``h_low_clamp`` bounds)
   H_high = P90 of warmup entropies  (clamped to ``h_high_clamp`` bounds)
   (A minimum gap of ``min_threshold_gap`` is enforced between H_low and H_high.)

3. **Running** (step >= warmup_steps):
   Every ``update_every`` steps the thresholds are updated via EMA:
       H_low  ← beta * H_low  + (1-beta) * P10(recent)
       H_high ← beta * H_high + (1-beta) * P90(recent)
   After each EMA update the clamping and gap constraints are re-applied.

4. **Schedule** (first ``schedule_steps`` after warmup):
   ``get_all_thresholds()`` returns a linear interpolation between the
   initial threshold values and the (clamped) learned values, giving the
   model a gradual transition instead of a sudden jump.
"""

import numpy as np


class CZU:
    """Comfort Zone Updater.

    Args:
        n_layers:          Number of HoT layers in the model.
        warmup_steps:      Steps to force Path C and collect entropy stats (default 1000).
        update_every:      Steps between EMA threshold updates after warmup (default 500).
        ema_beta:          EMA decay factor for threshold updates (default 0.95).
        init_H_low:        Default H_low before warmup finishes (default 0.3).
        init_H_high:       Default H_high before warmup finishes (default 0.7).
        min_threshold_gap: Minimum enforced gap between H_low and H_high (default 0.05).
        h_low_clamp:       ``(min, max)`` hard bounds for H_low after learning
                           (default (0.2, 0.75)).  Prevents thresholds from
                           saturating near 0 or 1 regardless of the warmup entropy
                           distribution.
        h_high_clamp:      ``(min, max)`` hard bounds for H_high after learning
                           (default (0.35, 0.90)).
        schedule_steps:    Post-warmup steps over which thresholds are linearly
                           interpolated from ``init_H_low/H_high`` to the clamped
                           learned values.  Set to 0 to disable (default 2000).
    """

    def __init__(
        self,
        n_layers: int,
        warmup_steps: int = 1000,
        update_every: int = 500,
        ema_beta: float = 0.95,
        init_H_low: float = 0.3,
        init_H_high: float = 0.7,
        min_threshold_gap: float = 0.05,
        h_low_clamp: tuple = (0.2, 0.75),
        h_high_clamp: tuple = (0.35, 0.90),
        schedule_steps: int = 2000,
    ) -> None:
        self.n_layers = n_layers
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        self.ema_beta = ema_beta

        self.min_threshold_gap = min_threshold_gap
        self.h_low_clamp = h_low_clamp
        self.h_high_clamp = h_high_clamp
        self.schedule_steps = schedule_steps

        self.step = 0
        self.initialized = False   # True once thresholds are set from warmup data

        # Per-layer thresholds (used before initialization as reasonable defaults)
        self.H_low = [init_H_low] * n_layers
        self.H_high = [init_H_high] * n_layers

        # Remember initial values so the post-warmup schedule can interpolate
        # from them toward the learned (clamped) targets.
        self._sched_init_H_low = init_H_low
        self._sched_init_H_high = init_H_high

        # Entropy buffer collected during warmup
        self._warmup_buf: list[list] = [[] for _ in range(n_layers)]

        # Rolling buffer between EMA update steps
        self._ema_buf: list[list] = [[] for _ in range(n_layers)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def in_warmup(self) -> bool:
        """True while step < warmup_steps."""
        return self.step < self.warmup_steps

    def force_path_c(self) -> bool:
        """Returns True during the warmup phase (all layers use Path C)."""
        return self.in_warmup

    def get_thresholds(self, layer_idx: int) -> tuple:
        """Return ``(H_low, H_high)`` for the given layer index (schedule-aware)."""
        return self.get_all_thresholds()[layer_idx]

    def get_all_thresholds(self) -> list:
        """Return scheduled thresholds, one ``(H_low, H_high)`` tuple per layer.

        During the first ``schedule_steps`` after warmup the returned values
        are linearly interpolated from the initial defaults toward the
        clamped learned thresholds, preventing a sudden jump that would
        otherwise destabilize routing.
        """
        if not self.initialized or self.schedule_steps <= 0:
            return [(self.H_low[i], self.H_high[i]) for i in range(self.n_layers)]

        steps_after = self.step - self.warmup_steps
        alpha = min(1.0, steps_after / self.schedule_steps)

        result = []
        for i in range(self.n_layers):
            h_low = (1.0 - alpha) * self._sched_init_H_low + alpha * self.H_low[i]
            h_high = (1.0 - alpha) * self._sched_init_H_high + alpha * self.H_high[i]
            result.append((h_low, h_high))
        return result

    def update(self, layer_entropies: list) -> None:
        """Record entropy values for this training step and update thresholds.

        Args:
            layer_entropies: List of scalar entropy values (one per layer).
                             Accepts Python floats or PyTorch scalar tensors.
        """
        vals = [
            float(h.item()) if hasattr(h, "item") else float(h)
            for h in layer_entropies
        ]

        if self.in_warmup:
            for i, v in enumerate(vals):
                self._warmup_buf[i].append(v)
        else:
            # First step after warmup: initialize thresholds
            if not self.initialized:
                self._init_from_warmup()
                self.initialized = True

            for i, v in enumerate(vals):
                self._ema_buf[i].append(v)

            # Periodic EMA update
            steps_after = self.step - self.warmup_steps
            if steps_after > 0 and steps_after % self.update_every == 0:
                self._ema_update()

        self.step += 1

    # ------------------------------------------------------------------
    # State dict (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "initialized": self.initialized,
            "H_low": list(self.H_low),
            "H_high": list(self.H_high),
            "_sched_init_H_low": self._sched_init_H_low,
            "_sched_init_H_high": self._sched_init_H_high,
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = state["step"]
        self.initialized = state["initialized"]
        self.H_low = list(state["H_low"])
        self.H_high = list(state["H_high"])
        self._sched_init_H_low = state.get("_sched_init_H_low", self._sched_init_H_low)
        self._sched_init_H_high = state.get("_sched_init_H_high", self._sched_init_H_high)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_thresholds(self) -> None:
        """Clamp H_low / H_high to configured bounds and enforce minimum gap."""
        lo_min, lo_max = self.h_low_clamp
        hi_min, hi_max = self.h_high_clamp
        for i in range(self.n_layers):
            h_low = float(np.clip(self.H_low[i], lo_min, lo_max))
            h_high = float(np.clip(self.H_high[i], hi_min, hi_max))
            # Enforce minimum gap; shift outward symmetrically within global bounds
            if h_high - h_low < self.min_threshold_gap:
                half = self.min_threshold_gap / 2.0
                mid = (h_low + h_high) / 2.0
                h_low = max(lo_min, mid - half)
                h_high = min(hi_max, mid + half)
            self.H_low[i] = h_low
            self.H_high[i] = h_high

    def _init_from_warmup(self) -> None:
        """Initialize H_low / H_high from P10 / P90 of warmup entropies, then clamp."""
        for i in range(self.n_layers):
            buf = np.array(self._warmup_buf[i])
            if len(buf) >= 2:
                low = float(np.percentile(buf, 10))
                high = float(np.percentile(buf, 90))
                # Enforce a minimum gap to avoid degenerate routing
                if high - low < self.min_threshold_gap:
                    half = self.min_threshold_gap / 2.0
                    mid = (low + high) / 2.0
                    low = max(0.0, mid - half)
                    high = min(1.0, mid + half)
                self.H_low[i] = low
                self.H_high[i] = high
        # Apply hard bounds: prevents thresholds from saturating near 0 or 1
        self._clamp_thresholds()

    def _ema_update(self) -> None:
        """Update thresholds via EMA over the recent entropy buffer, then clamp."""
        for i in range(self.n_layers):
            buf = self._ema_buf[i]
            if buf:
                arr = np.array(buf)
                new_low = float(np.percentile(arr, 10))
                new_high = float(np.percentile(arr, 90))
                b = self.ema_beta
                self.H_low[i] = b * self.H_low[i] + (1.0 - b) * new_low
                self.H_high[i] = b * self.H_high[i] + (1.0 - b) * new_high
                self._ema_buf[i] = []   # clear for next window
        # Re-apply hard bounds after every EMA step
        self._clamp_thresholds()
