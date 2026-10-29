"""Aggregate inter-community infection hazards for the TwoScale model."""
import random
from typing import List, Optional, Tuple

import numpy as np

from .._validation import integer, matrix, nonnegative, positive


class MacroEngine:
    """Compute beta * T * W[i,j] * I[i]/N[i] * S[j]/N[j]."""

    def __init__(
        self,
        W: np.ndarray,
        beta_macro: float,
        T: float = 1.0,
        community_sizes: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
    ):
        if community_sizes is None:
            raw = np.asarray(W)
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("W must be a nonempty square matrix")
            community_sizes = [1] * raw.shape[0]
        self.n = len(community_sizes)
        if self.n == 0:
            raise ValueError("Provide at least one community size")
        self.N = np.asarray(
            [integer(value, "community size", minimum=1) for value in community_sizes],
            dtype=float,
        )
        self.W = matrix(W, self.n)
        self.beta_macro = nonnegative(beta_macro, "beta_macro")
        self.T = positive(T, "macro_T")
        self.rng = random.Random() if rng is None else rng
        self.hazards = np.zeros_like(self.W)
        self.total_hazard = 0.0

    def _compute_hazards_matrix(self, I_counts: List[int], S_counts: List[int]) -> np.ndarray:
        I = np.asarray(I_counts, dtype=float)
        S = np.asarray(S_counts, dtype=float)
        if I.shape != (self.n,) or S.shape != (self.n,):
            raise ValueError("Provide one S and I count per community")
        if (
            not np.all(np.isfinite(I)) or not np.all(np.isfinite(S))
            or np.any(I < 0) or np.any(S < 0) or np.any(I + S > self.N)
        ):
            raise ValueError("Community counts must be finite and fit their population sizes")
        with np.errstate(over="ignore", invalid="ignore"):
            hazards = (I / self.N)[:, None] * (S / self.N)[None, :]
            hazards *= self.W
            hazards *= self.beta_macro * self.T
        if not np.all(np.isfinite(hazards)):
            raise ValueError("Inter-community rates overflow; reduce beta_macro or macro_T")
        return hazards

    def update_hazards(self, I_counts: List[int], S_counts: List[int]) -> None:
        self.hazards = self._compute_hazards_matrix(I_counts, S_counts)
        self.total_hazard = float(self.hazards.sum())
        if not np.isfinite(self.total_hazard):
            raise ValueError("Total inter-community rate overflows")

    def total_hazard_given(self, I_counts: List[int], S_counts: List[int]) -> float:
        total = float(self._compute_hazards_matrix(I_counts, S_counts).sum())
        if not np.isfinite(total):
            raise ValueError("Total inter-community rate overflows")
        return total

    def sample_transfer(self) -> Tuple[int, int]:
        """Sample a directed transfer; return (-1, -1) when none is enabled."""
        if self.total_hazard <= 0.0:
            return -1, -1
        flat = self.hazards.ravel()
        enabled = np.flatnonzero(flat > 0.0)
        cumulative = np.cumsum(flat[enabled])
        threshold = self.rng.random() * float(cumulative[-1])
        offset = int(np.searchsorted(cumulative, threshold, side="right"))
        index = int(enabled[min(offset, len(enabled) - 1)])
        return divmod(index, self.n)
