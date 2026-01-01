import json
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple


def community_nodes_from_micrographs(micro_graphs: Sequence[nx.Graph]) -> List[List[int]]:
    return [list(G.nodes()) for G in micro_graphs]


def node_to_community(micro_graphs: Sequence[nx.Graph]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, G in enumerate(micro_graphs):
        for n in G.nodes():
            mapping[n] = idx
    return mapping


def build_macro_graph_from_W(W: np.ndarray) -> nx.Graph:
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w)
    return G


def _macro_positions(
    n: int,
    macro_graph: Optional[nx.Graph],
    layout: str,
    scale: float,
    seed: Optional[int],
) -> Dict[int, Tuple[float, float]]:
    layout = (layout or "spring").strip().lower()
    if layout == "line":
        return {i: (i * scale, 0.0) for i in range(n)}
    if layout == "circular":
        base = nx.circular_layout(range(n), scale=scale)
        return {i: (float(base[i][0]), float(base[i][1])) for i in range(n)}
    if layout == "spring":
        G = macro_graph if macro_graph is not None else nx.complete_graph(n)
        base = nx.spring_layout(G, seed=seed, scale=scale)
        return {i: (float(base[i][0]), float(base[i][1])) for i in range(n)}
    raise ValueError("macro_layout must be 'spring', 'circular', or 'line'")


def _micro_positions(
    G: nx.Graph,
    layout: str,
    scale: float,
    seed: Optional[int],
) -> Dict[int, Tuple[float, float]]:
    layout = (layout or "spring").strip().lower()
    if layout == "circular":
        base = nx.circular_layout(G, scale=scale)
    elif layout == "spring":
        base = nx.spring_layout(G, seed=seed, scale=scale)
    else:
        raise ValueError("micro_layout must be 'spring' or 'circular'")
    return {n: (float(base[n][0]), float(base[n][1])) for n in G.nodes()}


def two_scale_layout(
    micro_graphs: Sequence[nx.Graph],
    macro_graph: Optional[nx.Graph] = None,
    macro_layout: str = "spring",
    micro_layout: str = "spring",
    macro_scale: float = 6.0,
    micro_scale: float = 1.2,
    seed: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute a two-level layout by placing each community around a macro position.
    Returns dict[node -> (x, y)] for full graph drawing.
    """
    n_comm = len(micro_graphs)
    macro_pos = _macro_positions(n_comm, macro_graph, macro_layout, macro_scale, seed)

    pos: Dict[int, Tuple[float, float]] = {}
    for idx, Gc in enumerate(micro_graphs):
        local = _micro_positions(Gc, micro_layout, micro_scale, seed)
        cx, cy = macro_pos[idx]
        for node, (x, y) in local.items():
            pos[node] = (x + cx, y + cy)
    return pos


def _split_edges(
    full_graph: nx.Graph,
    node_comm: Dict[int, int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    intra: List[Tuple[int, int]] = []
    inter: List[Tuple[int, int]] = []
    for u, v in full_graph.edges():
        if node_comm.get(u) == node_comm.get(v):
            intra.append((u, v))
        else:
            inter.append((u, v))
    return intra, inter


def draw_two_scale_network(
    micro_graphs: Sequence[nx.Graph],
    full_graph: nx.Graph,
    W: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    macro_layout: str = "spring",
    micro_layout: str = "spring",
    macro_scale: float = 6.0,
    micro_scale: float = 1.2,
    node_size: float = 40.0,
    intra_edge_alpha: float = 0.35,
    inter_edge_alpha: float = 0.7,
    inter_edge_width: float = 1.2,
    show_macro_nodes: bool = False,
    seed: Optional[int] = None,
) -> plt.Axes:
    """
    Draw a full two-scale network with community coloring and edge separation.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    node_comm = node_to_community(micro_graphs)
    macro_graph = build_macro_graph_from_W(W) if W is not None else None
    pos = two_scale_layout(
        micro_graphs=micro_graphs,
        macro_graph=macro_graph,
        macro_layout=macro_layout,
        micro_layout=micro_layout,
        macro_scale=macro_scale,
        micro_scale=micro_scale,
        seed=seed,
    )

    intra_edges, inter_edges = _split_edges(full_graph, node_comm)
    n_comm = len(micro_graphs)
    cmap = plt.get_cmap("tab20")
    node_colors = [cmap(node_comm[n] % 20) for n in full_graph.nodes()]

    nx.draw_networkx_nodes(
        full_graph,
        pos,
        node_size=node_size,
        node_color=node_colors,
        ax=ax,
    )
    if intra_edges:
        nx.draw_networkx_edges(
            full_graph,
            pos,
            edgelist=intra_edges,
            edge_color="gray",
            alpha=intra_edge_alpha,
            ax=ax,
        )
    if inter_edges:
        nx.draw_networkx_edges(
            full_graph,
            pos,
            edgelist=inter_edges,
            edge_color="black",
            alpha=inter_edge_alpha,
            width=inter_edge_width,
            ax=ax,
        )

    if show_macro_nodes:
        macro_pos = _macro_positions(n_comm, macro_graph, macro_layout, macro_scale, seed)
        nx.draw_networkx_nodes(
            macro_graph if macro_graph is not None else nx.Graph(),
            macro_pos,
            node_size=node_size * 3.0,
            node_color="white",
            edgecolors="black",
            linewidths=1.0,
            ax=ax,
        )

    ax.set_axis_off()
    return ax


def draw_macro_graph(
    W: np.ndarray,
    ax: Optional[plt.Axes] = None,
    layout: str = "spring",
    node_size: float = 300.0,
    edge_width_scale: float = 0.6,
    show_weights: bool = True,
    seed: Optional[int] = None,
) -> plt.Axes:
    """
    Draw the macro graph defined by W.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    G = build_macro_graph_from_W(W)
    pos = _macro_positions(G.number_of_nodes(), G, layout, scale=2.5, seed=seed)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [edge_width_scale * w for w in weights]
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#cccccc", ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#333333", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    if show_weights:
        labels = {(u, v): f"{G[u][v]['weight']:.0f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=9, ax=ax)

    ax.set_axis_off()
    return ax


if __name__ == "__main__":
    from network import generate_two_scale_network

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    net_cfg = cfg["network"]
    save_plot = True

    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=int(net_cfg["communities"]),
        community_size=int(net_cfg["community_size"]),
        inter_links=int(net_cfg["inter_links"]),
        seed=int(net_cfg["seed"]),
        macro_graph_type=str(net_cfg["macro_graph_type"]),
        micro_graph_type=str(net_cfg["micro_graph_type"]),
        edge_prob=float(net_cfg["edge_prob"]),
    )

    draw_two_scale_network(
        micro_graphs,
        full_graph,
        W=W,
        show_macro_nodes=False,
        seed=int(net_cfg["seed"]),
    )
    if save_plot:
        out_dir = "plots"
        os.makedirs(out_dir, exist_ok=True)
        edge_prob_str = str(net_cfg["edge_prob"]).replace(".", "p")
        filename = (
            "network_"
            f"k{net_cfg['communities']}_"
            f"n{net_cfg['community_size']}_"
            f"inter{net_cfg['inter_links']}_"
            f"macro{net_cfg['macro_graph_type']}_"
            f"micro{net_cfg['micro_graph_type']}_"
            f"p{edge_prob_str}.png"
        )
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
