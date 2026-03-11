import random
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np


def generate_two_scale_network(
    n_communities: int,
    community_size: int,
    inter_links: int,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    leaf_count: int = 0,
    leaf_degree: int = 1,
    star_leaf_attachment: str = "random",
) -> Tuple[List[nx.Graph], nx.Graph, np.ndarray]:
    """
    Generate a two-layer network with micro graphs plus inter-community edges.
    Returns (micro_graphs, full_graph, W).
    """
    if n_communities < 1:
        raise ValueError("n_communities must be >= 1")
    if community_size < 1:
        raise ValueError("community_size must be >= 1")
    if inter_links < 0:
        raise ValueError("inter_links must be >= 0")
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be between 0 and 1")
    if leaf_count < 0:
        raise ValueError("leaf_count must be >= 0")
    if leaf_degree < 0:
        raise ValueError("leaf_degree must be >= 0")

    macro_graph_type = (macro_graph_type or "").strip().lower()
    if macro_graph_type in {"complete", "clique", "fully_connected"}:
        macro_graph_type = "complete"
    elif macro_graph_type in {"chain", "path", "line"}:
        macro_graph_type = "chain"
    elif macro_graph_type in {"star", "hub_spoke", "hub-and-spoke", "hubspoke"}:
        macro_graph_type = "star"
    else:
        raise ValueError("macro_graph_type must be 'complete', 'chain', or 'star'")

    star_leaf_attachment = (star_leaf_attachment or "").strip().lower()
    if star_leaf_attachment in {"random", "uniform"}:
        star_leaf_attachment = "random"
    elif star_leaf_attachment in {"node0", "to_zero", "zero", "hub0"}:
        star_leaf_attachment = "node0"
    elif star_leaf_attachment in {"random_nonzero", "random_not_zero", "nonzero"}:
        star_leaf_attachment = "random_nonzero"
    else:
        raise ValueError("star_leaf_attachment must be 'random', 'node0', or 'random_nonzero'")

    micro_graph_type = (micro_graph_type or "").strip().lower()
    if micro_graph_type in {"complete", "clique", "fully_connected"}:
        micro_graph_type = "complete"
    elif micro_graph_type in {"random", "erdos_renyi", "erdos-renyi", "er"}:
        micro_graph_type = "random"
    elif micro_graph_type in {"chain", "path", "line"}:
        micro_graph_type = "chain"
    elif micro_graph_type in {"random_connected", "connected_random", "connected_er", "connected_erdos_renyi"}:
        micro_graph_type = "random_connected"
    elif micro_graph_type in {
        "clique_leaves",
        "clique_with_leaves",
        "clique_plus_leaves",
        "full_with_leaves",
        "full_leaves",
    }:
        micro_graph_type = "clique_leaves"
    else:
        raise ValueError(
            "micro_graph_type must be 'complete', 'random', 'random_connected', 'chain', or 'clique_leaves'"
        )

    if macro_graph_type == "star":
        # In star mode we enforce:
        # - community 0: clique of size `community_size`
        # - communities 1..k-1: single-node leaves
        resolved_clique_size = 0
        resolved_leaf_count = 0
        resolved_leaf_degree = 0
        resolved_community_size = community_size
        community_sizes = [community_size] + [1] * (n_communities - 1)
    else:
        if micro_graph_type == "clique_leaves":
            resolved_clique_size = community_size
            resolved_leaf_count = int(leaf_count)
            resolved_leaf_degree = int(leaf_degree)
            if resolved_leaf_degree > resolved_clique_size:
                raise ValueError("leaf_degree must be <= community_size")
            resolved_community_size = resolved_clique_size + resolved_leaf_count
        else:
            resolved_clique_size = 0
            resolved_leaf_count = 0
            resolved_leaf_degree = 0
            resolved_community_size = community_size
        community_sizes = [resolved_community_size] * n_communities

    rng = random.Random(seed)

    micro_graphs: List[nx.Graph] = []
    communities_nodes: List[List[int]] = []
    full_graph = nx.Graph()
    next_node = 0

    for community_idx in range(n_communities):
        size_this_community = community_sizes[community_idx]
        nodes = list(range(next_node, next_node + size_this_community))
        communities_nodes.append(nodes)
        Gc = nx.Graph()
        Gc.add_nodes_from(nodes)
        if macro_graph_type == "star":
            if community_idx == 0:
                for idx, u in enumerate(nodes):
                    for v in nodes[idx + 1:]:
                        Gc.add_edge(u, v, weight=1.0)
            # leaf communities are single-node by construction, so no intra edges
        elif micro_graph_type == "complete":
            for idx, u in enumerate(nodes):
                for v in nodes[idx + 1:]:
                    Gc.add_edge(u, v, weight=1.0)
        elif micro_graph_type == "random":
            for idx, u in enumerate(nodes):
                for v in nodes[idx + 1:]:
                    if rng.random() < edge_prob:
                        Gc.add_edge(u, v, weight=1.0)
        elif micro_graph_type == "random_connected":
            n = len(nodes)
            if n > 1:
                target_edges = int(round(edge_prob * n * (n - 1) / 2))
                if target_edges < n - 1:
                    raise ValueError("edge_prob too low to create a connected graph with exact average degree")
                max_tries = 10_000
                Gc = None
                for _ in range(max_tries):
                    seed_try = rng.randrange(1 << 30)
                    candidate = nx.gnm_random_graph(n, target_edges, seed=seed_try)
                    if nx.is_connected(candidate):
                        mapping = {i: nodes[i] for i in range(n)}
                        Gc = nx.relabel_nodes(candidate, mapping, copy=True)
                        break
                if Gc is None:
                    raise RuntimeError("Failed to generate a connected random graph; increase edge_prob or max_tries")
                for u, v in Gc.edges():
                    Gc[u][v]["weight"] = 1.0
        elif micro_graph_type == "chain":
            for idx in range(len(nodes) - 1):
                Gc.add_edge(nodes[idx], nodes[idx + 1], weight=1.0)
        elif micro_graph_type == "clique_leaves":
            clique_nodes = nodes[:resolved_clique_size]
            leaf_nodes = nodes[resolved_clique_size:]

            for idx, u in enumerate(clique_nodes):
                for v in clique_nodes[idx + 1:]:
                    Gc.add_edge(u, v, weight=1.0)

            for leaf in leaf_nodes:
                if resolved_leaf_degree == 0:
                    continue
                if resolved_leaf_degree == resolved_clique_size:
                    targets = clique_nodes
                else:
                    targets = rng.sample(clique_nodes, k=resolved_leaf_degree)
                for target in targets:
                    Gc.add_edge(leaf, target, weight=1.0)
        else:
            raise RuntimeError(f"Unsupported micro_graph_type: {micro_graph_type}")
        micro_graphs.append(Gc)
        full_graph.add_nodes_from(nodes)
        full_graph.add_edges_from(Gc.edges(data=True))
        next_node += size_this_community

    W = np.zeros((n_communities, n_communities), dtype=float)

    if macro_graph_type == "complete":
        macro_pairs = [
            (i, j)
            for i in range(n_communities)
            for j in range(i + 1, n_communities)
        ]
    elif macro_graph_type == "chain":
        macro_pairs = [(i, i + 1) for i in range(n_communities - 1)]
    elif macro_graph_type == "star":
        # Community 0 is the central hub (clique-community on macro level),
        # all remaining communities are leaves connected only to hub.
        macro_pairs = [(0, j) for j in range(1, n_communities)]
    else:
        raise RuntimeError(f"Unsupported macro_graph_type: {macro_graph_type}")

    for i, j in macro_pairs:
        nodes_i = communities_nodes[i]
        nodes_j = communities_nodes[j]

        if (
            macro_graph_type == "star"
            and i == 0
            and len(nodes_j) == 1
            and len(nodes_i) >= 1
            and star_leaf_attachment in {"node0", "random_nonzero"}
        ):
            leaf_node = nodes_j[0]
            hub_node_zero = nodes_i[0]

            if star_leaf_attachment == "node0":
                if inter_links > 0:
                    full_graph.add_edge(hub_node_zero, leaf_node, weight=1.0)
                    num_links = 1
                else:
                    num_links = 0
            else:
                eligible_hub_nodes = [u for u in nodes_i if u != hub_node_zero]
                if inter_links > 0 and not eligible_hub_nodes:
                    raise ValueError(
                        "star_leaf_attachment='random_nonzero' requires clique size >= 2 "
                        "(community_size must be >= 2)"
                    )
                num_links = min(inter_links, len(eligible_hub_nodes))
                if num_links > 0:
                    if num_links == len(eligible_hub_nodes):
                        targets = eligible_hub_nodes
                    else:
                        targets = rng.sample(eligible_hub_nodes, k=num_links)
                    for u in targets:
                        full_graph.add_edge(u, leaf_node, weight=1.0)

            W[i, j] = float(num_links)
            W[j, i] = float(num_links)
            continue

        max_possible = len(nodes_i) * len(nodes_j)
        num_links = min(inter_links, max_possible)
        if num_links == 0:
            continue
        if num_links == max_possible:
            choices = range(max_possible)
        else:
            choices = rng.sample(range(max_possible), k=num_links)
        size_j = len(nodes_j)
        for idx in choices:
            u = nodes_i[idx // size_j]
            v = nodes_j[idx % size_j]
            full_graph.add_edge(u, v, weight=1.0)
        W[i, j] = float(num_links)
        W[j, i] = float(num_links)

    return micro_graphs, full_graph, W


__all__ = ["generate_two_scale_network"]
