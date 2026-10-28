"""Graph-free preparation of the frozen chain TwoScale inputs.

This is deliberately not a full-network graph API. Node-state Graphs are
still created by MicroModel. Only immutable adjacency preparation is replaced;
arbitrary Full/NetworkX workflows use the existing network() fallback.
"""
from array import array
from bisect import bisect_right
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from .topology import NeighborList


class UnsupportedTopology(ValueError):
    """Use the original graph builder for this input's semantics."""


class PreparedMicroTopology:
    """One community's independently owned ordered adjacency arrays."""

    __slots__ = ("_nodes", "_prepared_neighbors", "__weakref__")

    def __init__(self, rows, offset):
        self._nodes = range(offset, offset + len(rows))
        shifted = {}
        for local, source in enumerate(rows):
            row = NeighborList()
            row._nodes = (array("q", source._nodes) if offset == 0 else
                          array("q", (node + offset for node in source._nodes)))
            # Do not share mutable arrays with the cache or another community.
            row._weights = array("d", source._weights)
            shifted[local + offset] = row
        self._prepared_neighbors = shifted

    def nodes(self):
        return self._nodes

    def number_of_nodes(self):
        return len(self._nodes)


def _edge(record, n):
    u, v, weight = record
    # Noncanonical types and implicit extra nodes must retain legacy handling.
    if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n):
        raise UnsupportedTopology("Local node IDs must lie in the frozen node range")
    weight = float(weight)
    if not (weight >= 0 and math.isfinite(weight)):
        raise UnsupportedTopology("Invalid edge weight")
    return u, v, weight


def _append_edge(rows, u, v, weight):
    # Same two ordered appends as MicroModel.add_edge, including self-loops.
    rows[u]._nodes.append(v)
    rows[u]._weights.append(weight)
    rows[v]._nodes.append(u)
    rows[v]._weights.append(weight)


def ordered_rows(n, edges):
    """Match replay of final nx.Graph.edges, including duplicate overwrites.

    Saved NX edge lists normally already have increasing first endpoints,
    u<=v, and no duplicate pairs. That case needs no temporary adjacency dicts.
    Unordered/reversed/duplicate rows use ordered deduplication instead.
    """
    rows = [NeighborList() for _ in range(n)]
    previous = -1
    seen = set()
    for record in edges:
        u, v, weight = _edge(record, n)
        if u < previous or v < u:
            break
        if u != previous:
            previous = u
            seen.clear()
        if v in seen:
            break
        seen.add(v)
        _append_edge(rows, u, v, weight)
    else:
        return tuple(rows)

    # The partial canonical result is discarded; build the final simple graph
    # adjacency in encounter order, with last weight but first insertion slot.
    del rows
    adjacency = [{} for _ in range(n)]
    for record in edges:
        u, v, weight = _edge(record, n)
        adjacency[u][v] = weight
        adjacency[v][u] = weight
    rows = [NeighborList() for _ in range(n)]
    for u, neighbors in enumerate(adjacency):
        for v, weight in neighbors.items():
            if v >= u:
                _append_edge(rows, u, v, weight)
    return tuple(rows)


@lru_cache(maxsize=2)
def _load_template(path_string, expected_sha256, n, file_identity):
    """Bounded, process-local cache of local-node templates, never run state."""
    contents = Path(path_string).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise RuntimeError(f"Frozen graph fingerprint changed: {path_string}")
    payload = json.loads(gzip.decompress(contents))
    return ordered_rows(n, payload["edges"])


def clear_topology_cache():
    _load_template.cache_clear()


def topology_cache_info():
    return _load_template.cache_info()._asdict()


def prepared_chain_network(task, output):
    """Return prepared micro topologies plus unchanged macro metadata, or None.

    Only chain inputs with cross-community interfaces take the fast path.
    The returned metadata Graph has no physical edges: this function must not
    be used for Full-network simulation. Callers use None to select legacy NX.
    """
    if task.get("series") != "chain":
        return None
    sizes = task["community_sizes"]
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        return None
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)
    if offsets[-1] >= 1 << 63:
        return None
    W = np.zeros((len(sizes), len(sizes)))
    boundary = [0] * len(sizes)
    for u, v in task["interface_edges"]:
        if (type(u) is not int or type(v) is not int or
                not (0 <= u < offsets[-1] and 0 <= v < offsets[-1])):
            return None
        a, b = bisect_right(offsets, u) - 1, bisect_right(offsets, v) - 1
        if a == b:
            return None
        # Keep raw interface multiplicity and scalar addition order.
        W[a, b] += 1
        W[b, a] += 1
        boundary[a] += 1
        boundary[b] += 1

    micro, communities = [], []
    task_templates = {}
    try:
        for index, n in enumerate(sizes):
            if task["graph_inputs"] and index < len(task["graph_inputs"]):
                reference = task["graph_inputs"][index]
                if "sha256" not in reference:
                    return None
                path = (Path(output) / reference["snapshot"]).resolve()
                key = (str(path), reference["sha256"], n)
                rows = task_templates.get(key)
                if rows is None:
                    info = path.stat()
                    identity = (info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns)
                    rows = _load_template(*key, identity)
                    task_templates[key] = rows
            else:
                rows = tuple(NeighborList() for _ in range(n))
            micro.append(PreparedMicroTopology(rows, offsets[index]))
            communities.append(list(range(offsets[index], offsets[index + 1])))
    except UnsupportedTopology:
        return None
    metadata = nx.Graph()
    metadata.graph.update(community_sizes=sizes, alphas=[1.] * len(sizes),
                          boundary_counts=boundary, mu2_values=[1.] * len(sizes),
                          spectral_descriptors_computed=False)
    return micro, metadata, W, communities
