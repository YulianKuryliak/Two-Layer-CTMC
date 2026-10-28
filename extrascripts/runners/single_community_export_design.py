"""Deterministic graph and interface construction for the export experiment.

This module deliberately contains no epidemic simulation code.  The runner
uses these builders to make the E0/E1/E2 graph controls described in
``EXPORT_PATH_SPECTRUM_EXPERIMENT_PLAN.md`` reproducible and inspectable.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


N = 100
INTERNAL_DEGREE = 6


def canonical_seed(*parts: object) -> int:
    """Stable integer seed; unlike Python hash(), it is process-independent."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**32)


def graph_hash(graph: nx.Graph, source: int, bridges: Iterable[int]) -> str:
    edges = sorted((min(int(u), int(v)), max(int(u), int(v)),
                    float(data.get("weight", 1.0)))
                   for u, v, data in graph.edges(data=True))
    blob = json.dumps({"nodes": sorted(map(int, graph.nodes())), "edges": edges,
                       "source": int(source), "bridges": sorted(map(int, bridges))},
                      separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _weighted(graph: nx.Graph) -> nx.Graph:
    result = nx.Graph()
    result.add_nodes_from(sorted(graph.nodes()))
    result.add_edges_from((u, v, {"weight": 1.0}) for u, v in graph.edges())
    return result


def e0a_graph(side: str, ell: int, n: int = N, seed: int = 0) -> tuple[nx.Graph, int, list[int], dict[str, Any]]:
    """Corridor plus a side graph attached only to source 0."""
    if side not in {"path", "clique", "random_connected"}:
        raise ValueError(f"unknown E0a side topology: {side}")
    if ell < 1 or ell + 2 > n:
        raise ValueError("invalid corridor length")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((i, i + 1) for i in range(ell))
    side_nodes = list(range(ell + 1, n))
    attachment = side_nodes[0]
    graph.add_edge(0, attachment)
    if side == "path":
        graph.add_edges_from(zip(side_nodes, side_nodes[1:]))
    elif side == "clique":
        graph.add_edges_from((u, v) for pos, u in enumerate(side_nodes) for v in side_nodes[pos + 1:])
    else:
        target_edges = round(6 * len(side_nodes) / 2)
        rng = random.Random(seed)
        for _ in range(10_000):
            candidate = nx.gnm_random_graph(len(side_nodes), target_edges, seed=rng.randrange(2**32))
            if nx.is_connected(candidate):
                graph.add_edges_from((side_nodes[u], side_nodes[v]) for u, v in candidate.edges())
                break
        else:
            raise RuntimeError("could not build connected E0 random side graph")
    return _weighted(graph), 0, [ell], {"block": "E0a", "side": side, "ell": ell}


def e0b_graph(q: int, ell: int = 4, n: int = N) -> tuple[nx.Graph, int, list[int], dict[str, Any]]:
    """q internally vertex-disjoint equal-length paths from source to bridge."""
    if q not in {1, 2, 4}:
        raise ValueError("E0b q must be one of 1, 2, 4")
    needed = 2 + q * (ell - 1)
    if needed >= n:
        raise ValueError("not enough nodes")
    source, bridge = 0, 1
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    next_node = 2
    for _ in range(q):
        path = [source] + list(range(next_node, next_node + ell - 1)) + [bridge]
        next_node += ell - 1
        graph.add_edges_from(zip(path, path[1:]))
    # Keep all unused vertices connected through a branch at s; this does not
    # create a second s--bridge route.
    spare = list(range(next_node, n))
    if spare:
        graph.add_edge(source, spare[0])
        graph.add_edges_from(zip(spare, spare[1:]))
    return _weighted(graph), source, [bridge], {"block": "E0b", "q": q, "ell": ell}


def _connected_random_regular(n: int, d: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    for _ in range(10_000):
        graph = nx.random_regular_graph(d, n, seed=rng.randrange(2**32))
        if nx.is_connected(graph):
            return _weighted(graph)
    raise RuntimeError("failed to sample a connected random regular graph")


def modular_regular_cut(cut_edges: int, n: int = N, d: int = INTERNAL_DEGREE, seed: int = 0) -> tuple[nx.Graph, dict[int, int]]:
    """Two d-regular halves joined by exactly ``cut_edges`` cross edges.

    Each degree-preserving two-edge swap produces two cross edges.  E2 uses
    2, 4, and 16; those values are therefore feasible without degree drift.
    """
    if n % 2 or cut_edges <= 0 or cut_edges % 2:
        raise ValueError("n must be even and cut_edges must be positive/even")
    half = n // 2
    left = _connected_random_regular(half, d, canonical_seed(seed, "left"))
    right = _connected_random_regular(half, d, canonical_seed(seed, "right"))
    graph = nx.disjoint_union(left, right)
    rng = random.Random(canonical_seed(seed, "swaps"))
    swaps = cut_edges // 2
    for _ in range(swaps):
        for _attempt in range(50_000):
            # Earlier swaps remove internal edges, so select from the current
            # graph rather than the original half-graphs.
            left_edges = [(u, v) for u, v in graph.edges() if u < half and v < half]
            right_edges = [(u, v) for u, v in graph.edges() if u >= half and v >= half]
            if not left_edges or not right_edges:
                raise RuntimeError("no remaining internal edge for cut swap")
            a, b = rng.choice(left_edges)
            c, e = rng.choice(right_edges)
            # Current components use distinct nodes and there are no prior
            # cross edges for these selected endpoints in a simple graph.
            if len({a, b, c, e}) < 4 or graph.has_edge(a, c) or graph.has_edge(b, e):
                continue
            graph.remove_edge(a, b)
            graph.remove_edge(c, e)
            graph.add_edge(a, c)
            graph.add_edge(b, e)
            break
        else:
            raise RuntimeError("failed to perform degree-preserving cut swap")
    graph = _weighted(graph)
    if not nx.is_connected(graph) or any(degree != d for _, degree in graph.degree()):
        raise RuntimeError("invalid modular regular graph")
    observed = sum(1 for u, v in graph.edges() if (u < half) != (v < half))
    if observed != cut_edges:
        raise RuntimeError(f"cut construction expected {cut_edges}, observed {observed}")
    return graph, {node: 0 if node < half else 1 for node in graph.nodes()}


def e1_source_graph(family: str, graph_id: int, n: int = N, d: int = INTERNAL_DEGREE) -> tuple[nx.Graph, dict[str, Any]]:
    seed = canonical_seed("E1", family, graph_id, n, d)
    if family == "random_regular":
        graph = _connected_random_regular(n, d, seed)
    elif family == "modular_regular_cut4":
        graph, _ = modular_regular_cut(4, n=n, d=d, seed=seed)
    elif family == "random_connected_gnm":
        target_edges = round(n * d / 2)
        rng = random.Random(seed)
        for _ in range(10_000):
            graph = nx.gnm_random_graph(n, target_edges, seed=rng.randrange(2**32))
            if nx.is_connected(graph):
                graph = _weighted(graph)
                break
        else:
            raise RuntimeError("failed to sample connected G(n,m)")
    else:
        raise ValueError(f"unknown E1 family {family}")
    return graph, {"block": "E1", "family": family, "graph_id": graph_id, "topology_seed": seed}


def _select_k(nodes: list[int], k: int, rng: random.Random) -> list[int] | None:
    return sorted(rng.sample(nodes, k)) if len(nodes) >= k else None


def e1_placements(graph: nx.Graph, source: int, k: int, seed: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Choose low/middle/high distance interfaces, degree matched when needed."""
    distances = nx.single_source_shortest_path_length(graph, source)
    nodes = [node for node in graph if node != source]
    rng = random.Random(seed)
    regular = len({degree for _, degree in graph.degree()}) == 1
    candidate_degree_sets = [None] if regular else sorted({graph.degree(v) for v in nodes})
    best = None
    for degree in candidate_degree_sets:
        pool = nodes if degree is None else [v for v in nodes if graph.degree(v) == degree]
        shells: dict[int, list[int]] = {}
        for v in pool:
            shells.setdefault(distances[v], []).append(v)
        attainable = sorted(distance for distance, values in shells.items() if len(values) >= k)
        if len(attainable) < 3:
            continue
        positions = [attainable[0], attainable[len(attainable) // 2], attainable[-1]]
        if len(set(positions)) < 3:
            continue
        choices = {label: _select_k(shells[distance], k, rng)
                   for label, distance in zip(("near", "middle", "far"), positions, strict=True)}
        if any(value is None for value in choices.values()):
            continue
        spread = positions[-1] - positions[0]
        candidate = (spread, degree, choices, positions)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return {}, {"status": "design_unavailable", "reason": "no degree-matched shells with 3 placements"}
    _, degree, choices, positions = best
    return {key: list(value) for key, value in choices.items()}, {
        "status": "feasible", "degree_match": "regular" if degree is None else [int(degree)] * k,
        "shell_distances": dict(zip(("near", "middle", "far"), positions, strict=True)),
    }


def e2_interfaces_at_distance(
    graph: nx.Graph, partition: dict[int, int], source: int, distance: int, k: int = 5
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Pick both interface sides from one explicit source-distance shell."""
    distances = nx.single_source_shortest_path_length(graph, source)
    source_side = partition[source]
    same = sorted(v for v in graph if v != source and partition[v] == source_side and distances[v] == distance)
    opposite = sorted(v for v in graph if partition[v] != source_side and distances[v] == distance)
    if len(same) >= k and len(opposite) >= k:
        return {"same_side": same[:k], "opposite_side": opposite[:k]}, {
            "status": "feasible", "common_distance": distance, "histogram": [distance] * k,
        }
    return {}, {"status": "design_unavailable", "reason": "insufficient nodes at requested common shell"}


def e2_interfaces(graph: nx.Graph, partition: dict[int, int], source: int, k: int = 5) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Pick same/opposite-side sets from a common exact-distance shell.

    Try D=3, then 2, then 4 as specified by the plan.  The caller compares
    cut levels and retains only blocks with the same selected D.
    """
    for distance in (3, 2, 4):
        interfaces, meta = e2_interfaces_at_distance(graph, partition, source, distance, k)
        if interfaces:
            return interfaces, meta
    return {}, {"status": "design_unavailable", "reason": "no common D shell with at least k nodes per side"}


def shortest_path_counts(graph: nx.Graph, source: int, targets: Iterable[int]) -> dict[int, int]:
    distances = {source: 0}
    counts = {source: 1}
    queue = [source]
    for node in queue:
        for neighbor in graph.neighbors(node):
            candidate = distances[node] + 1
            if neighbor not in distances:
                distances[neighbor] = candidate
                counts[neighbor] = counts[node]
                queue.append(neighbor)
            elif distances[neighbor] == candidate:
                counts[neighbor] += counts[node]
    return {int(target): int(counts.get(target, 0)) for target in targets}


def descriptors(graph: nx.Graph, source: int, bridges: list[int], *, family: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structural descriptors once per graph/interface; no epidemic randomness."""
    n = graph.number_of_nodes()
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    distances = nx.single_source_shortest_path_length(graph, source)
    dvals = np.array([distances[b] for b in bridges], dtype=float)
    # n=100 is small, but sparse solvers avoid setting a dense-eigendecomposition
    # precedent for later extensions.
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), weight="weight", format="csr", dtype=float)
    lap = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    invsqrt = np.zeros(n)
    nonzero = degrees > 0
    invsqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    lnorm = sparse.eye(n, format="csr") - sparse.diags(invsqrt) @ adjacency @ sparse.diags(invsqrt)
    try:
        rho = float(spla.eigsh(adjacency, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0])
        lsmall = np.sort(spla.eigsh(lap, k=2, which="SM", return_eigenvectors=False, tol=1e-10))
        nsmall = np.sort(spla.eigsh(lnorm, k=2, which="SM", return_eigenvectors=False, tol=1e-10))
        lambda2_l, lambda2_norm = float(lsmall[1]), float(nsmall[1])
    except Exception as exc:  # retain failure explicitly; never invent a gap
        rho = lambda2_l = lambda2_norm = math.nan
        solver_error = f"{type(exc).__name__}: {exc}"
    else:
        solver_error = None
    result = {
        "graph_family": family, "n": n, "m": graph.number_of_edges(),
        "realized_mean_degree": float(degrees.mean()), "degree_variance": float(degrees.var()),
        "source": int(source), "source_degree": int(graph.degree(source)),
        "bridges": list(map(int, bridges)), "bridge_degrees": sorted(int(graph.degree(b)) for b in bridges),
        "k": len(bridges), "distance_histogram": sorted(map(int, dvals)),
        "D_min": float(dvals.min()), "D_mean": float(dvals.mean()), "D_median": float(np.median(dvals)),
        "D_max": float(dvals.max()), "D_std": float(dvals.std()),
        "rho_adjacency": rho, "lambda2_laplacian": lambda2_l,
        "lambda2_normalized_laplacian": lambda2_norm, "spectral_solver_error": solver_error,
        "clustering": float(nx.average_clustering(graph)),
        "log1p_shortest_path_counts": {str(v): float(math.log1p(c)) for v, c in shortest_path_counts(graph, source, bridges).items()},
        "graph_hash": graph_hash(graph, source, bridges),
    }
    if n <= 100:
        result["ASPL_est"] = float(nx.average_shortest_path_length(graph))
        result["D_mean_over_ASPL"] = result["D_mean"] / result["ASPL_est"]
    if extra:
        result.update(extra)
    return result


def graph_payload(graph: nx.Graph, source: int, bridges: list[int], metadata: dict[str, Any]) -> dict[str, Any]:
    return {"source_nodes": sorted(map(int, graph.nodes())),
            "source_edges": [[int(u), int(v), float(data.get("weight", 1.0))]
                             for u, v, data in sorted(graph.edges(data=True))],
            "source": int(source), "bridges": list(map(int, bridges)), **metadata}
