import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any


def generate_two_scale_network(
    n_communities: int,
    community_size: int,
    inter_links: int,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
) -> Tuple[List[nx.Graph], nx.Graph, np.ndarray]:
    """
    Generate:
      - micro_graphs: list of n_communities each a micrograph (complete or random)
      - full_graph: union of all communities plus inter-community links
      - W: k-by-k numpy array of inter-community link counts

    Args:
      n_communities: number of communities (k)
      community_size: nodes per community
      inter_links: number of inter-community links per macro edge
      seed: optional random seed
      macro_graph_type: "complete" or "chain"
      micro_graph_type: "complete" or "random"
      edge_prob: probability of edge existence for random micrographs
    Returns:
      micro_graphs, full_graph, W
    """
    if n_communities < 1:
        raise ValueError("n_communities must be >= 1")
    if community_size < 1:
        raise ValueError("community_size must be >= 1")
    if inter_links < 0:
        raise ValueError("inter_links must be >= 0")
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be between 0 and 1")

    macro_graph_type = (macro_graph_type or "").strip().lower()
    if macro_graph_type in {"complete", "clique", "fully_connected"}:
        macro_graph_type = "complete"
    elif macro_graph_type in {"chain", "path", "line"}:
        macro_graph_type = "chain"
    else:
        raise ValueError("macro_graph_type must be 'complete' or 'chain'")

    micro_graph_type = (micro_graph_type or "").strip().lower()
    if micro_graph_type in {"complete", "clique", "fully_connected"}:
        micro_graph_type = "complete"
    elif micro_graph_type in {"random", "erdos_renyi", "erdos-renyi", "er"}:
        micro_graph_type = "random"
    else:
        raise ValueError("micro_graph_type must be 'complete' or 'random'")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    micro_graphs: List[nx.Graph] = []
    communities_nodes: List[List[int]] = []
    full_graph = nx.Graph()
    next_node = 0

    # build isolated communities
    for ci in range(n_communities):
        nodes = list(range(next_node, next_node + community_size))
        communities_nodes.append(nodes)
        Gc = nx.Graph()
        Gc.add_nodes_from(nodes)
        if micro_graph_type == "complete":
            for idx, u in enumerate(nodes):
                for v in nodes[idx + 1:]:
                    Gc.add_edge(u, v, weight=1.0)
        elif micro_graph_type == "random":
            for idx, u in enumerate(nodes):
                for v in nodes[idx + 1:]:
                    if random.random() < edge_prob:
                        Gc.add_edge(u, v, weight=1.0)
        else:
            raise RuntimeError(f"Unsupported micro_graph_type: {micro_graph_type}")
        micro_graphs.append(Gc)
        full_graph.add_nodes_from(nodes)
        full_graph.add_edges_from(Gc.edges(data=True))
        next_node += community_size

    # initialize inter-community weight matrix
    W = np.zeros((n_communities, n_communities), dtype=float)

    # add inter-community edges according to macro topology
    if macro_graph_type == "complete":
        macro_pairs = [
            (i, j)
            for i in range(n_communities)
            for j in range(i + 1, n_communities)
        ]
    elif macro_graph_type == "chain":
        macro_pairs = [(i, i + 1) for i in range(n_communities - 1)]
    else:
        raise RuntimeError(f"Unsupported macro_graph_type: {macro_graph_type}")

    for i, j in macro_pairs:
        nodes_i = communities_nodes[i]
        nodes_j = communities_nodes[j]
        max_possible = len(nodes_i) * len(nodes_j)
        num_links = min(inter_links, max_possible)
        if num_links == 0:
            continue
        if num_links == max_possible:
            choices = range(max_possible)
        else:
            choices = random.sample(range(max_possible), k=num_links)
        size_j = len(nodes_j)
        for idx in choices:
            u = nodes_i[idx // size_j]
            v = nodes_j[idx % size_j]
            full_graph.add_edge(u, v, weight=1.0)
        # record count (undirected)
        W[i, j] = float(num_links)
        W[j, i] = float(num_links)

    return micro_graphs, full_graph, W


# (Existing classes MicroEngine, MicroModel, MacroEngine, Orchestrator remain unchanged...)
