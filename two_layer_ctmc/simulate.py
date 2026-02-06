from dataclasses import dataclass
from typing import List, Optional, Union

import networkx as nx
import numpy as np

from .network import generate_two_scale_network
from .simulators import MicroMacroSimulator, MicroSimulator
from .types import MicroMacroSimulationResult, MicroSimulationResult

ModelLike = Union[int, str]


@dataclass(frozen=True)
class TwoLayerNetwork:
    micro_graphs: List[nx.Graph]
    full_graph: nx.Graph
    inter_matrix: np.ndarray
    community_nodes: List[List[int]]


def normalize_model(model: ModelLike) -> int:
    if isinstance(model, int):
        if model in (1, 2):
            return model
        raise ValueError("model must be 1 (SIS) or 2 (SIR)")
    if isinstance(model, str):
        key = model.strip().lower()
        if key in {"1", "sis"}:
            return 1
        if key in {"2", "sir"}:
            return 2
    raise ValueError("model must be 1/2 or SIS/SIR")


def build_two_layer_network(
    *,
    n_communities: int = 2,
    community_size: int = 50,
    inter_links: int = 1,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
) -> TwoLayerNetwork:
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=n_communities,
        community_size=community_size,
        inter_links=inter_links,
        seed=seed,
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
    )
    community_nodes = [list(G.nodes()) for G in micro_graphs]
    return TwoLayerNetwork(
        micro_graphs=micro_graphs,
        full_graph=full_graph,
        inter_matrix=W,
        community_nodes=community_nodes,
    )


def simulate_micro(
    *,
    beta: float,
    gamma: float,
    T_end: float,
    dt_out: float,
    model: ModelLike = "SIR",
    network: Optional[TwoLayerNetwork] = None,
    n_communities: int = 2,
    community_size: int = 50,
    inter_links: int = 1,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    initial_node: Optional[int] = None,
) -> MicroSimulationResult:
    """
    Run an in-memory microscopic simulation. If network is provided, network
    parameters are ignored.
    """
    if network is None:
        network = build_two_layer_network(
            n_communities=n_communities,
            community_size=community_size,
            inter_links=inter_links,
            seed=seed,
            macro_graph_type=macro_graph_type,
            micro_graph_type=micro_graph_type,
            edge_prob=edge_prob,
        )

    model_id = normalize_model(model)
    sim = MicroSimulator(
        full_graph=network.full_graph,
        comm_nodes=network.community_nodes,
        infection_rate=beta,
        recovery_rate=gamma,
        model=model_id,
    )
    return sim.run(
        T_end=T_end,
        dt_out=dt_out,
        initial_node=initial_node,
        seed=seed,
    )


def simulate_micromacro(
    *,
    beta_micro: float,
    gamma: float,
    tau_micro: float,
    T_end: float,
    beta_macro: Optional[float] = None,
    macro_T: float = 1.0,
    model: ModelLike = "SIR",
    network: Optional[TwoLayerNetwork] = None,
    n_communities: int = 2,
    community_size: int = 50,
    inter_links: int = 1,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    initial_community: int = 0,
    initial_node: Optional[int] = None,
) -> MicroMacroSimulationResult:
    """
    Run an in-memory micro/macro simulation. If network is provided, network
    parameters are ignored.
    """
    if network is None:
        network = build_two_layer_network(
            n_communities=n_communities,
            community_size=community_size,
            inter_links=inter_links,
            seed=seed,
            macro_graph_type=macro_graph_type,
            micro_graph_type=micro_graph_type,
            edge_prob=edge_prob,
        )

    model_id = normalize_model(model)
    beta_macro = beta_micro if beta_macro is None else beta_macro
    sim = MicroMacroSimulator(
        W=network.inter_matrix,
        micro_graphs=network.micro_graphs,
        beta_micro=beta_micro,
        gamma=gamma,
        beta_macro=beta_macro,
        tau_micro=tau_micro,
        T_end=T_end,
        macro_T=macro_T,
        model=model_id,
        full_graph=network.full_graph,
    )
    return sim.run(
        seed=seed,
        initial_community=initial_community,
        initial_node=initial_node,
    )


__all__ = [
    "TwoLayerNetwork",
    "build_two_layer_network",
    "simulate_micro",
    "simulate_micromacro",
    "normalize_model",
]
