import random
from typing import List, Optional, Tuple

import numpy as np


class MacroEngine:
    """
    Macro-scale engine with susceptibility-aware, size-normalized hazards.

    hazard_ij(t) = beta_macro * T * W[i,j] * infectivity_i * susceptibility_j
    infectivity_i = (I_i / N_i)
    susceptibility_j = (S_j / N_j)
    """

    def __init__(
        self,
        W: np.ndarray,
        beta_macro: float,
        T: float = 1.0,
        community_sizes: Optional[List[int]] = None,
        alphas: Optional[List[float]] = None,
        alpha_mode: str = "off",
        boundary_counts: Optional[List[int]] = None,
        mu2_values: Optional[List[float]] = None,
        correction_mode: str = "off",
        beta_in: float = 0.003,
        beta_out: float = 0.006,
    ):
        self.W = W
        self.beta_macro = beta_macro
        self.T = T
        self.n = W.shape[0]
        # alpha_mode controls whether the spectral exponent alpha_i is applied to
        # infectivity_i = (I_i / N_i) ** alpha_i. "off" (default) keeps the legacy
        # behaviour (exponent 1.0). "spectral" activates the parameters-only
        # correction alpha_i = 1/lambda2_tilde computed from each community's micro graph.
        self.alpha_mode = str(alpha_mode)
        
        # NEW: correction_mode for C2/C12 boundary-availability multipliers
        # "off" (default), "boundary_phi" (C2), "renorm_phi" (C12)
        self.correction_mode = str(correction_mode)
        self.beta_in = float(beta_in)
        self.beta_out = float(beta_out)
        
        self._frac_buffer = np.zeros_like(W, dtype=float)
        self._hazards_buffer = np.zeros_like(W, dtype=float)
        self._flat = np.zeros(W.size, dtype=float)
        self._flat_cumsum = np.zeros(W.size, dtype=float)
        self._flat_dirty = True

        # Sizes N_i of each community i
        if community_sizes is None:
            # Fallback: all communities of size 1 (keeps behaviour sane if sizes are missing)
            self.N = np.ones(self.n, dtype=float)
        else:
            self.N = np.asarray(community_sizes, dtype=float)
            assert self.N.shape[0] == self.n, "community_sizes length must match W.shape[0]"

        if alphas is None:
            self.alphas = np.ones(self.n, dtype=float)
        else:
            self.alphas = np.asarray(alphas, dtype=float)
            assert self.alphas.shape[0] == self.n, "alphas length must match W.shape[0]"
        
        # NEW: boundary descriptors for C2/C12 corrections
        if boundary_counts is None:
            self.boundary_counts = np.zeros(self.n, dtype=float)
        else:
            self.boundary_counts = np.asarray(boundary_counts, dtype=float)
            assert self.boundary_counts.shape[0] == self.n, "boundary_counts length must match W.shape[0]"
        
        if mu2_values is None:
            self.mu2_values = np.ones(self.n, dtype=float)
        else:
            self.mu2_values = np.asarray(mu2_values, dtype=float)
            assert self.mu2_values.shape[0] == self.n, "mu2_values length must match W.shape[0]"

        self.hazards = np.zeros_like(W, dtype=float)
        self.total_hazard = 0.0

    def _compute_hazards_matrix(self, I_counts: List[int], S_counts: List[int]) -> np.ndarray:
        """
        λ_ij = β T W_ij * (I_i / N_i) ** alpha_i * (S_j / N_j)
        
        With optional boundary-availability corrections (C2/C12):
        - correction_mode="boundary_phi": multiply by phi_i (bounded availability)
        - correction_mode="renorm_phi": multiply by renormalized rate phi_i = mu2/(mu2+r)
        """
        I = np.asarray(I_counts, dtype=float).reshape(-1, 1)  # (n,1)
        S = np.asarray(S_counts, dtype=float).reshape(1, -1)  # (1,n)

        N_i = self.N.reshape(-1, 1)  # (n,1)
        N_j = self.N.reshape(1, -1)  # (1,n)

        infectivity = I / N_i
        
        # Apply spectral damping (legacy alpha_mode)
        if self.alpha_mode == "spectral":
            # Parameters-only spectral correction: infectivity_i = (I_i/N_i) ** alpha_i.
            # alpha_i >= 1 for sparser communities (small lambda2_tilde), which
            # damps infectivity (since I/N < 1) and delays macro import, counteracting
            # the premature-arrival bias observed in low-connectivity regimes.
            alpha_col = self.alphas.reshape(-1, 1)  # (n,1)
            infectivity = np.power(infectivity, alpha_col)
        
        # NEW: Apply boundary-availability corrections (C2 or C12)
        if self.correction_mode == "boundary_phi":
            # C2: Bounded multiplier phi_i based on boundary availability and mixing
            # phi_i = min(1, boundary_fraction_i * (1 + mu2_i * I_i/N_i))
            # This is bounded in (0,1] and self-disables to 1 when mu2->1 or boundary_fraction->1
            boundary_frac = self.boundary_counts / np.maximum(self.N, 1.0)
            I_flat = I.flatten()
            N_flat = self.N
            mu2_flat = self.mu2_values
            # Soft saturation: phi starts small when I/N is small, approaches 1 as I/N grows or mu2 is large
            phi = np.minimum(1.0, boundary_frac * (1.0 + mu2_flat * (I_flat / N_flat)))
            phi = phi.reshape(-1, 1)
            infectivity = infectivity * phi
        elif self.correction_mode == "renorm_phi":
            # C12: Renormalized rate phi_i = mu2_i / (mu2_i + r) where r = beta_out/beta_in
            # This captures the competition between internal mixing (~ mu2*beta_in) and export (~ beta_out)
            r = self.beta_out / self.beta_in if self.beta_in > 0 else 1.0
            phi = self.mu2_values / (self.mu2_values + r)
            phi = phi.reshape(-1, 1)
            infectivity = infectivity * phi
        
        susceptibility = S / N_j
        np.matmul(infectivity, susceptibility, out=self._frac_buffer)
        np.multiply(self.W, self._frac_buffer, out=self._hazards_buffer)
        self._hazards_buffer *= (self.beta_macro * self.T)
        return self._hazards_buffer

    def update_hazards(self, I_counts: List[int], S_counts: List[int]):
        self.hazards = self._compute_hazards_matrix(I_counts, S_counts)
        self.total_hazard = float(self.hazards.sum())
        self._flat_dirty = True

    def total_hazard_given(self, I_counts: List[int], S_counts: List[int]) -> float:
        return float(self._compute_hazards_matrix(I_counts, S_counts).sum())

    def sample_transfer(self) -> Tuple[int, int]:
        """
        Sample directed edge (i,j) proportional to current hazard_ij.
        Returns (-1, -1) if no macro hazard.
        """
        if self.total_hazard <= 0.0:
            return -1, -1
        if self._flat_dirty:
            flat = self.hazards.ravel()
            np.copyto(self._flat, flat)
            np.cumsum(self._flat, out=self._flat_cumsum)
            self._flat_dirty = False
        thresh = random.random() * self.total_hazard
        idx = int(np.searchsorted(self._flat_cumsum, thresh))
        i, j = divmod(idx, self.n)
        return i, j
