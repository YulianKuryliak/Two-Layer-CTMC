"""Static graph descriptors used by the Experiment 05 corrections.

This module is intentionally experiment-local.  In particular, none of these
values are recomputed as the epidemic state changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import networkx as nx
import numpy as np


DEFAULT_POWER_TOLERANCE = 1.0e-6
DEFAULT_POWER_MAX_ITERATIONS = 1_000


class SpectralConvergenceError(RuntimeError):
    """Raised when the spectral-radius iteration does not meet its tolerance."""


@dataclass(frozen=True)
class SpectralRadiusResult:
    value: float
    iterations: int
    residual_norm: float
    tolerance: float


@dataclass(frozen=True)
class GraphDescriptors:
    """Descriptors and derived correction parameters for one community."""

    n: int
    m: int
    mean_degree: float
    spectral_radius: float
    average_shortest_path_length: Optional[float]
    tau_find: float
    kappa: float
    degenerate_descriptors: bool
    spectral_iterations: int
    spectral_residual_norm: float

    @property
    def lambda_1(self) -> float:
        """Paper-notation alias for the adjacency spectral radius."""

        return self.spectral_radius

    @property
    def ell(self) -> Optional[float]:
        """Paper-notation alias for average shortest-path length."""

        return self.average_shortest_path_length


def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    if graph.is_directed():
        raise ValueError("Experiment 05 graph descriptors require an undirected graph")
    # NetworkX preserves insertion order.  Sorting would fail for graphs whose
    # node identifiers have heterogeneous, non-orderable types.
    nodes = list(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    if not np.isfinite(adjacency).all():
        raise ValueError("adjacency weights must be finite")
    if np.any(adjacency < 0.0):
        raise ValueError("adjacency weights must be non-negative")
    return adjacency


def spectral_radius_power_iteration(
    graph: nx.Graph,
    *,
    tolerance: float = DEFAULT_POWER_TOLERANCE,
    max_iterations: int = DEFAULT_POWER_MAX_ITERATIONS,
) -> SpectralRadiusResult:
    """Estimate the adjacency spectral radius with deterministic shifted power iteration.

    Iteration is performed on ``A + I``.  The positive shift separates the
    equal-magnitude positive and negative eigenvalues of bipartite graphs.  The
    reported eigenvalue and convergence residual are always evaluated for the
    original adjacency matrix ``A``.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")

    adjacency = _adjacency_matrix(graph)
    n = adjacency.shape[0]
    if n == 0:
        return SpectralRadiusResult(0.0, 0, 0.0, tolerance)

    vector = np.full(n, 1.0 / math.sqrt(n), dtype=float)
    residual = math.inf
    eigenvalue = 0.0

    for iteration in range(1, max_iterations + 1):
        shifted_product = adjacency @ vector + vector
        norm = float(np.linalg.norm(shifted_product))
        if not math.isfinite(norm) or norm == 0.0:
            raise SpectralConvergenceError("shifted power iteration produced an invalid norm")
        vector = shifted_product / norm

        product = adjacency @ vector
        eigenvalue = float(vector @ product)
        residual = float(np.linalg.norm(product - eigenvalue * vector))
        threshold = tolerance * max(1.0, abs(eigenvalue))
        if residual <= threshold:
            # Roundoff can produce a tiny negative Rayleigh quotient for the
            # zero matrix, but the radius of a non-negative adjacency is >= 0.
            value = max(0.0, eigenvalue)
            # Preserve the exact scientific branch at lambda_1 == 1.  A value
            # indistinguishable from one at the requested eigensolver
            # tolerance must not accidentally enable transmission because of
            # a last-bit positive Rayleigh-quotient error.
            if abs(value - 1.0) <= tolerance:
                value = 1.0
            return SpectralRadiusResult(value, iteration, residual, tolerance)

    raise SpectralConvergenceError(
        "shifted power iteration failed to converge in "
        f"{max_iterations} iterations (residual={residual:.12g}, "
        f"eigenvalue={eigenvalue:.12g}, tolerance={tolerance:.12g})"
    )


def compute_graph_descriptors(
    graph: nx.Graph,
    beta_in: float,
    *,
    tolerance: float = DEFAULT_POWER_TOLERANCE,
    max_iterations: int = DEFAULT_POWER_MAX_ITERATIONS,
) -> GraphDescriptors:
    """Compute static descriptors and the Experiment 05 lag/relaxation scales.

    A disconnected graph or a graph with spectral radius at most one is
    degenerate.  Undefined path length is represented by ``None``; no
    replacement descriptor is silently substituted.  Derived parameters are
    set to their no-transmission semantics (``tau_find=inf``, ``kappa=0``).
    """

    if not math.isfinite(beta_in) or beta_in < 0.0:
        raise ValueError("beta_in must be finite and non-negative")

    spectral = spectral_radius_power_iteration(
        graph, tolerance=tolerance, max_iterations=max_iterations
    )
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    mean_degree = 0.0 if n == 0 else 2.0 * m / n

    connected = n > 0 and nx.is_connected(graph)
    path_length = (
        float(nx.average_shortest_path_length(graph, weight=None)) if connected else None
    )
    degenerate = spectral.value <= 1.0 or path_length is None

    if degenerate or beta_in == 0.0:
        tau_find = math.inf
        kappa = 0.0
    else:
        growth_rate = beta_in * (spectral.value - 1.0)
        tau_find = max(0.0, math.log(n) / growth_rate)
        kappa = max(0.0, growth_rate / path_length)

    return GraphDescriptors(
        n=n,
        m=m,
        mean_degree=mean_degree,
        spectral_radius=spectral.value,
        average_shortest_path_length=path_length,
        tau_find=tau_find,
        kappa=kappa,
        degenerate_descriptors=degenerate,
        spectral_iterations=spectral.iterations,
        spectral_residual_norm=spectral.residual_norm,
    )
