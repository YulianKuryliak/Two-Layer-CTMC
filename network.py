import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any


def generate_two_scale_network(
    n_communities: int,
    community_size: int,
    max_inter_links: int,
    seed: Optional[int] = None
) -> Tuple[List[nx.Graph], nx.Graph, np.ndarray]:
    """
    Generate:
      - micro_graphs: list of n_communities each a complete subgraph of community_size
      - full_graph: union of all communities plus random inter-community links (0 to max_inter_links)
      - W: k×k numpy array of inter-community link counts

    Args:
      n_communities: number of communities (k)
      community_size: nodes per community
      max_inter_links: max number of random links per community-pair
      seed: optional random seed
    Returns:
      micro_graphs, full_graph, W
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    micro_graphs: List[nx.Graph] = []
    communities_nodes: List[List[int]] = []
    full_graph = nx.Graph()
    next_node = 0

    # build isolated complete communities
    for ci in range(n_communities):
        nodes = list(range(next_node, next_node + community_size))
        communities_nodes.append(nodes)
        Gc = nx.Graph()
        Gc.add_nodes_from(nodes)
        # complete intra-community edges
        for u in nodes:
            for v in nodes:
                if u < v:
                    Gc.add_edge(u, v, weight=1.0)
        micro_graphs.append(Gc)
        full_graph.add_nodes_from(nodes)
        full_graph.add_edges_from(Gc.edges(data=True))
        next_node += community_size

    # initialize inter-community weight matrix
    W = np.zeros((n_communities, n_communities), dtype=float)

    # add random inter-community edges
    for i in range(n_communities):
        for j in range(i + 1, n_communities):
            # random number of edges between communities i and j
            num_links = max_inter_links #random.randint(0, max_inter_links)
            for _ in range(num_links):
                u = random.choice(communities_nodes[i])
                v = random.choice(communities_nodes[j])
                full_graph.add_edge(u, v, weight=1.0)
            # record count (undirected)
            W[i, j] = num_links
            W[j, i] = num_links

    return micro_graphs, full_graph, W


# (Existing classes EfficientEpidemicGraph, MicroModel, MacroEngine, Orchestrator remain unchanged...)


