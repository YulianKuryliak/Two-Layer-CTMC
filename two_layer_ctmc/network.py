"""Seeded generation of simple community networks."""
import random
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from ._validation import integer, nonnegative, seed_value


def generate_two_scale_network(
    n_communities: int,
    community_size: int,
    inter_links: int,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
) -> Tuple[List[nx.Graph], nx.Graph, np.ndarray]:
    """Return local graphs, their physical union, and cross-community weights.

    Communities have equal sizes and disjoint integer node IDs. Macro topology
    is complete or chain. Local topology is complete, chain, random (independent
    edges), or random_connected (uniform fixed-edge rejection sampling).
    """
    count = integer(n_communities, "n_communities", 1)
    size = integer(community_size, "community_size", 1)
    links = integer(inter_links, "inter_links")
    probability = nonnegative(edge_prob, "edge_prob")
    if probability > 1:
        raise ValueError("edge_prob must be between 0 and 1")
    if macro_graph_type not in {"complete", "chain"}:
        raise ValueError("macro_graph_type must be 'complete' or 'chain'")
    if micro_graph_type not in {"complete", "chain", "random", "random_connected"}:
        raise ValueError("micro_graph_type must be 'complete', 'chain', 'random', or 'random_connected'")
    if links > size * size:
        raise ValueError("inter_links exceeds the possible edges between two communities")
    rng = random.Random(seed_value(seed))
    full = nx.Graph()
    communities = []
    for ci in range(count):
        nodes = list(range(ci * size, (ci + 1) * size))
        local = nx.Graph()
        local.add_nodes_from(nodes)
        if micro_graph_type == "random_connected" and size > 1:
            edges = round(probability * size * (size - 1) / 2)
            if edges < size - 1:
                raise ValueError("edge_prob is too low for a connected graph of this size")
            for _ in range(10_000):
                candidate = nx.gnm_random_graph(size, edges, seed=rng.randrange(1 << 30))
                if nx.is_connected(candidate):
                    local = nx.relabel_nodes(candidate, dict(enumerate(nodes)), copy=True)
                    break
            else:
                raise ValueError("Could not sample a connected graph; increase edge_prob")
        elif micro_graph_type == "chain":
            local.add_edges_from(zip(nodes, nodes[1:]))
        elif micro_graph_type in {"complete", "random"}:
            for index, u in enumerate(nodes):
                for v in nodes[index + 1:]:
                    if micro_graph_type == "complete" or rng.random() < probability:
                        local.add_edge(u, v)
        nx.set_edge_attributes(local, 1.0, "weight")
        full.add_nodes_from(nodes)
        full.add_edges_from(local.edges(data=True))
        communities.append(local)
    W = np.zeros((count, count), dtype=float)
    pairs = ((i, j) for i in range(count) for j in range(i + 1, count)
             if macro_graph_type == "complete" or j == i + 1)
    for i, j in pairs:
        choices = range(size * size) if links == size * size else rng.sample(range(size * size), links)
        for index in choices:
            full.add_edge(i * size + index // size, j * size + index % size, weight=1.0)
        W[i, j] = W[j, i] = float(links)
    return communities, full, W
