import random
from typing import List, Optional

import networkx as nx

from ..orchestrators.micro import MicroOrchestrator
from ..types import MicroSimulationResult


class MicroSimulator:
    """
    User-facing Micro simulator. Builds a micro orchestrator per run and
    returns in-memory results.
    """

    def __init__(
        self,
        full_graph: nx.Graph,
        comm_nodes: List[List[int]],
        infection_rate: float,
        recovery_rate: float,
        model: int = 2,
    ):
        self.full_graph = full_graph
        self.comm_nodes = comm_nodes
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.model = model

    def run(
        self,
        T_end: float,
        dt_out: float,
        initial_node: Optional[int] = None,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> MicroSimulationResult:
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both")
        if rng is None:
            rng = random.Random(seed)

        orchestrator = MicroOrchestrator(
            full_graph=self.full_graph,
            comm_nodes=self.comm_nodes,
            infection_rate=self.infection_rate,
            recovery_rate=self.recovery_rate,
            model=self.model,
            rng=rng,
        )
        return orchestrator.run(T_end=T_end, dt_out=dt_out, initial_node=initial_node)


__all__ = ["MicroSimulator"]
