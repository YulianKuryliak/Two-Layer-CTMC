import random
from typing import List, Optional, Tuple

import numpy as np


class MacroEngine:
    """
    Macro-scale engine with susceptibility-aware, size-normalized hazards.

    hazard_ij(t) = beta_macro * T * W[i,j] * (I_i / N_i) * (S_j / N_j)
    """

    def __init__(
        self,
        W: np.ndarray,
        beta_macro: float,
        T: float = 1.0,
        community_sizes: Optional[List[int]] = None,
    ):
        self.W = W
        self.beta_macro = beta_macro
        self.T = T
        self.n = W.shape[0]
        self._frac_buffer = np.zeros_like(W, dtype=float)
        self._hazards_buffer = np.zeros_like(W, dtype=float)
        self._flat = np.zeros(W.size, dtype=float)
        self._flat_cumsum = np.zeros(W.size, dtype=float)

        # Sizes N_i of each community i
        if community_sizes is None:
            # Fallback: all communities of size 1 (keeps behaviour sane if sizes are missing)
            self.N = np.ones(self.n, dtype=float)
        else:
            self.N = np.asarray(community_sizes, dtype=float)
            assert self.N.shape[0] == self.n, "community_sizes length must match W.shape[0]"

        self.hazards = np.zeros_like(W, dtype=float)
        self.total_hazard = 0.0

    def _compute_hazards_matrix(self, I_counts: List[int], S_counts: List[int]) -> np.ndarray:
        """
        λ_ij = β T W_ij * (I_i / N_i) * (S_j / N_j)
        """
        I = np.asarray(I_counts, dtype=float).reshape(-1, 1)  # (n,1)
        S = np.asarray(S_counts, dtype=float).reshape(1, -1)  # (1,n)

        N_i = self.N.reshape(-1, 1)  # (n,1)
        N_j = self.N.reshape(1, -1)  # (1,n)

        np.matmul(I / N_i, S / N_j, out=self._frac_buffer)
        np.multiply(self.W, self._frac_buffer, out=self._hazards_buffer)
        self._hazards_buffer *= (self.beta_macro * self.T)
        return self._hazards_buffer

    def update_hazards(self, I_counts: List[int], S_counts: List[int]):
        self.hazards = self._compute_hazards_matrix(I_counts, S_counts)
        self.total_hazard = float(self.hazards.sum())
        flat = self.hazards.ravel()
        np.copyto(self._flat, flat)
        np.cumsum(self._flat, out=self._flat_cumsum)

    def total_hazard_given(self, I_counts: List[int], S_counts: List[int]) -> float:
        return float(self._compute_hazards_matrix(I_counts, S_counts).sum())

    def sample_transfer(self) -> Tuple[int, int]:
        """
        Sample directed edge (i,j) proportional to current hazard_ij.
        Returns (-1, -1) if no macro hazard.
        """
        if self.total_hazard <= 0.0:
            return -1, -1
        thresh = random.random() * self.total_hazard
        idx = int(np.searchsorted(self._flat_cumsum, thresh))
        i, j = divmod(idx, self.n)
        return i, j
