"""Experiment-local spectral lag and latent-boundary correction policies."""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import List

from .graph_descriptors import GraphDescriptors


_TIME_TOLERANCE = 1.0e-10


def _unit_fraction(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


class FixedGridHistory:
    """Fixed synchronization-grid history with linear interpolation."""

    def __init__(self, step: float) -> None:
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        self.step = float(step)
        self._times: List[float] = []
        self._values: List[float] = []

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(self._times)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def append(self, time: float, value: float) -> None:
        time = float(time)
        value = _unit_fraction(value, "history value")
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("history time must be finite and non-negative")

        expected = 0.0 if not self._times else self._times[-1] + self.step
        allowed = _TIME_TOLERANCE * max(1.0, abs(expected), abs(time))
        if abs(time - expected) > allowed:
            raise ValueError(
                f"history samples must follow the fixed grid: expected {expected}, got {time}"
            )
        # Store the exact expected grid point to prevent accumulated timestamp
        # drift from affecting bisect/interpolation decisions.
        self._times.append(expected)
        self._values.append(value)

    def value_at(self, time: float) -> float:
        if not self._times:
            raise RuntimeError("history has no samples")
        time = float(time)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("query time must be finite and non-negative")
        if time <= self._times[0]:
            return self._values[0]
        if time >= self._times[-1]:
            return self._values[-1]

        right = bisect_right(self._times, time)
        left = right - 1
        fraction = (time - self._times[left]) / self.step
        return self._values[left] + fraction * (
            self._values[right] - self._values[left]
        )


class LagPolicy:
    """Evaluate prevalence at ``max(0, t - tau_find)``."""

    def __init__(self, delay: float, sync_step: float, *, degenerate: bool = False) -> None:
        if math.isnan(delay) or delay < 0.0:
            raise ValueError("delay must be non-negative and not NaN")
        self.delay = float(delay)
        self.degenerate = bool(degenerate) or math.isinf(self.delay)
        self.history = FixedGridHistory(sync_step)

    @classmethod
    def from_descriptors(
        cls, descriptors: GraphDescriptors, sync_step: float
    ) -> "LagPolicy":
        return cls(
            descriptors.tau_find,
            sync_step,
            degenerate=descriptors.degenerate_descriptors,
        )

    def record(self, time: float, prevalence: float) -> None:
        self.history.append(time, prevalence)

    def contagiousness(self, time: float) -> float:
        if self.degenerate:
            return 0.0
        time = float(time)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        return self.history.value_at(max(0.0, time - self.delay))


class LatentBoundaryPolicy:
    """Segment-exact solution of dB/dt = kappa * (input - B)."""

    def __init__(
        self, kappa: float, *, initial: float = 0.0, degenerate: bool = False
    ) -> None:
        if math.isnan(kappa) or kappa < 0.0:
            raise ValueError("kappa must be non-negative and not NaN")
        self.kappa = float(kappa)
        self.degenerate = bool(degenerate)
        self.state = 0.0 if self.degenerate else _unit_fraction(initial, "initial")

    @classmethod
    def from_descriptors(cls, descriptors: GraphDescriptors) -> "LatentBoundaryPolicy":
        return cls(
            descriptors.kappa,
            initial=0.0,
            degenerate=descriptors.degenerate_descriptors,
        )

    def value_after(self, driving_prevalence: float, duration: float) -> float:
        driving = _unit_fraction(driving_prevalence, "driving_prevalence")
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be finite and non-negative")
        if self.degenerate:
            return 0.0
        if math.isinf(self.kappa):
            return self.state if duration == 0.0 else driving
        relaxation = -math.expm1(-self.kappa * duration)
        return self.state + relaxation * (driving - self.state)

    def advance(self, driving_prevalence: float, duration: float) -> float:
        self.state = self.value_after(driving_prevalence, duration)
        return self.state


class CombinedLagLatentPolicy:
    """Latent relaxation driven by fixed-grid lagged prevalence."""

    def __init__(self, lag: LagPolicy, latent: LatentBoundaryPolicy) -> None:
        self.lag = lag
        self.latent = latent

    @classmethod
    def from_descriptors(
        cls, descriptors: GraphDescriptors, sync_step: float
    ) -> "CombinedLagLatentPolicy":
        return cls(
            LagPolicy.from_descriptors(descriptors, sync_step),
            LatentBoundaryPolicy.from_descriptors(descriptors),
        )

    @property
    def state(self) -> float:
        return self.latent.state

    def record(self, time: float, prevalence: float) -> None:
        self.lag.record(time, prevalence)

    def advance(self, segment_start: float, duration: float) -> float:
        # The lagged input is left-continuous and held constant on this
        # scheduler segment.  A macro event can end a segment early; calling
        # advance again gives the exact semigroup update for the new segment.
        driving = self.lag.contagiousness(segment_start)
        return self.latent.advance(driving, duration)

    def value_after(self, segment_start: float, duration: float) -> float:
        """Return the exact end-of-segment value without mutating the policy."""

        driving = self.lag.contagiousness(segment_start)
        return self.latent.value_after(driving, duration)


def aggregate_hazard(
    beta_out: float,
    edge_weight: float,
    contagiousness: float,
    susceptibility: float,
    *,
    degenerate: bool = False,
) -> float:
    """Compute beta_out * W_ij * corrected_i_bar * susceptible_j_bar."""

    beta_out = float(beta_out)
    edge_weight = float(edge_weight)
    if not math.isfinite(beta_out) or beta_out < 0.0:
        raise ValueError("beta_out must be finite and non-negative")
    if not math.isfinite(edge_weight) or edge_weight < 0.0:
        raise ValueError("edge_weight must be finite and non-negative")
    contagiousness = _unit_fraction(contagiousness, "contagiousness")
    susceptibility = _unit_fraction(susceptibility, "susceptibility")
    if degenerate:
        return 0.0
    return beta_out * edge_weight * contagiousness * susceptibility
