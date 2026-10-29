"""The user API: network construction and two in-memory simulation methods."""
from dataclasses import dataclass
import random
from numbers import Integral
from typing import List, Optional, Union

import networkx as nx
import numpy as np

from ._validation import graph, integer, matrix, model_id, nonnegative, positive, seed_value
from .network import generate_two_scale_network
from .orchestrators.micro import MicroOrchestrator
from .orchestrators.micromacro import MicroMacroOrchestrator
from .types import MicroMacroSimulationResult, MicroSimulationResult


@dataclass(frozen=True)
class TwoLayerNetwork:
    """Physical graph, its disjoint local graphs, and weighted interface matrix.

    Custom inputs must be simple undirected graphs with integer node IDs,
    no self-loops, finite nonnegative weights and nonempty communities. Keep
    topology and weights fixed while a simulation runs. Input node attributes
    are not used as epidemic state and are never overwritten.
    """
    micro_graphs: List[nx.Graph]
    full_graph: nx.Graph
    inter_matrix: np.ndarray
    community_nodes: List[List[int]]

    def __post_init__(self):
        self.validate()

    def validate(self):
        graph(self.full_graph, "full_graph")
        if not self.micro_graphs or len(self.micro_graphs) != len(self.community_nodes):
            raise ValueError("Provide one local graph per community")
        flattened = [node for nodes in self.community_nodes for node in nodes]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(self.full_graph):
            raise ValueError("community_nodes must partition the full graph without overlap")
        W = matrix(self.inter_matrix, len(self.micro_graphs))
        membership = {node: ci for ci, nodes in enumerate(self.community_nodes) for node in nodes}
        interfaces = np.zeros_like(W)
        intra_counts = [0] * len(self.micro_graphs)
        for u, v, data in self.full_graph.edges(data=True):
            i, j = membership[u], membership[v]
            if i == j:
                intra_counts[i] += 1
            else:
                weight = float(data.get("weight", 1.0))
                interfaces[i, j] += weight
                interfaces[j, i] += weight
        if not np.allclose(W, interfaces, rtol=1e-12, atol=1e-12):
            raise ValueError("inter_matrix must equal cross-community edge-weight totals")
        for ci, (local, nodes) in enumerate(zip(self.micro_graphs, self.community_nodes)):
            graph(local, f"micro_graphs[{ci}]")
            if set(local) != set(nodes) or local.number_of_edges() != intra_counts[ci]:
                raise ValueError("Each local graph must be the corresponding induced subgraph")
            for u, v, data in local.edges(data=True):
                if not self.full_graph.has_edge(u, v) or not np.isclose(
                    float(data.get("weight", 1.0)),
                    float(self.full_graph[u][v].get("weight", 1.0)), rtol=1e-12, atol=1e-12,
                ):
                    raise ValueError("Local edges and weights must match full_graph")


def build_two_layer_network(
    *, n_communities: int = 2, community_size: int = 50, inter_links: int = 1,
    seed: Optional[int] = None, macro_graph_type: str = "complete",
    micro_graph_type: str = "complete", edge_prob: float = 0.1,
) -> TwoLayerNetwork:
    """Build equal-sized communities with complete/chain macro connectivity.

    Local topologies: complete, chain, random, random_connected. edge_prob is
    used only for random topologies. Each macro link has inter_links distinct
    physical edges of weight one. The seed is local and leaves global RNG alone.
    """
    local, full, W = generate_two_scale_network(
        n_communities, community_size, inter_links, seed,
        macro_graph_type, micro_graph_type, edge_prob,
    )
    return TwoLayerNetwork(local, full, W, [list(G) for G in local])


def _network(network, n_communities, community_size, inter_links, seed,
             macro_graph_type, micro_graph_type, edge_prob):
    if network is None:
        return build_two_layer_network(
            n_communities=n_communities, community_size=community_size,
            inter_links=inter_links, seed=seed, macro_graph_type=macro_graph_type,
            micro_graph_type=micro_graph_type, edge_prob=edge_prob,
        )
    if not isinstance(network, TwoLayerNetwork):
        raise ValueError("network must be a TwoLayerNetwork")
    network.validate()
    return network


def _initial_node(value):
    if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)):
        raise ValueError("initial_node must be an integer or None")
    return value


def simulate_micro(
    *, beta: float, gamma: float, T_end: float, dt_out: float,
    model: Union[int, str] = "SIR", network: Optional[TwoLayerNetwork] = None,
    n_communities: int = 2, community_size: int = 50, inter_links: int = 1,
    seed: Optional[int] = None, macro_graph_type: str = "complete",
    micro_graph_type: str = "complete", edge_prob: float = 0.1,
    initial_node: Optional[int] = None,
) -> MicroSimulationResult:
    """Full Gillespie SIR/SIS. Return rows, typed events, and the seed node.

    rows are (community, time, S, I, R), sampled at dt_out plus T_end.
    Events at an output time are included in that time's counts. Only events
    at or before T_end are applied. No files or global RNG state are changed.
    With an explicit network, network-generation parameters are ignored.
    """
    beta, gamma = nonnegative(beta, "beta"), nonnegative(gamma, "gamma")
    T_end, dt_out = positive(T_end, "T_end"), positive(dt_out, "dt_out")
    model, seed = model_id(model), seed_value(seed)
    initial_node = _initial_node(initial_node)
    net = _network(network, n_communities, community_size, inter_links, seed,
                   macro_graph_type, micro_graph_type, edge_prob)
    orchestrator = MicroOrchestrator(
        net.full_graph, net.community_nodes, beta, gamma, model, rng=random.Random(seed),
    )
    return orchestrator.run(T_end=T_end, dt_out=dt_out, initial_node=initial_node)


def simulate_micromacro(
    *, beta_micro: float, gamma: float, tau_micro: float, T_end: float,
    beta_macro: Optional[float] = None, macro_T: float = 1.0,
    model: Union[int, str] = "SIR", network: Optional[TwoLayerNetwork] = None,
    n_communities: int = 2, community_size: int = 50, inter_links: int = 1,
    seed: Optional[int] = None, macro_graph_type: str = "complete",
    micro_graph_type: str = "complete", edge_prob: float = 0.1,
    initial_community: int = 0, initial_node: Optional[int] = None,
) -> MicroMacroSimulationResult:
    """Approximate TwoScale SIR/SIS with midpoint synchronization.

    Return (times, total_infected, community_logs, events). Each community log
    contains times/S/I/R sampled on the shared synchronization/transfer times.
    Macro hazard is beta_macro * macro_T * W_ij * I_i/N_i * S_j/N_j;
    imported infections choose a susceptible destination node uniformly.
    tau_micro controls synchronization error; this is not Full Gillespie.
    Random streams are private to each call. Input graphs are not modified.
    """
    beta_micro = nonnegative(beta_micro, "beta_micro")
    beta_macro = beta_micro if beta_macro is None else nonnegative(beta_macro, "beta_macro")
    gamma, macro_T = nonnegative(gamma, "gamma"), positive(macro_T, "macro_T")
    tau_micro, T_end = positive(tau_micro, "tau_micro"), positive(T_end, "T_end")
    model, seed = model_id(model), seed_value(seed)
    initial_community = integer(initial_community, "initial_community")
    initial_node = _initial_node(initial_node)
    net = _network(network, n_communities, community_size, inter_links, seed,
                   macro_graph_type, micro_graph_type, edge_prob)
    if initial_community >= len(net.micro_graphs):
        raise ValueError("initial_community is out of range")
    if initial_node is not None and initial_node not in net.micro_graphs[initial_community]:
        raise ValueError("initial_node must belong to initial_community")
    orchestrator = MicroMacroOrchestrator(
        W=net.inter_matrix, micro_graphs=net.micro_graphs, beta_micro=beta_micro,
        gamma=gamma, beta_macro=beta_macro, tau_micro=tau_micro, T_end=T_end,
        macro_T=macro_T, model=model, rng=random.Random(seed),
    )
    local = orchestrator.micro_models[initial_community]
    if initial_node is None:
        initial_node = local.rng.choice(list(local.G))
    local._infect_node(initial_node)
    orchestrator.event_log.append({
        "time": 0.0, "wait_time": 0.0, "event_type": "seed", "mode": "seed",
        "community": initial_community, "node": int(initial_node), "src": None,
    })
    return orchestrator.run()
