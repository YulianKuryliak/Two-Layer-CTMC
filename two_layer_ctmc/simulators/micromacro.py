import random
from typing import List, Optional

import networkx as nx
import numpy as np

from ..orchestrators.micromacro import MicroMacroOrchestrator
from ..types import MicroMacroSimulationResult


class MicroMacroSimulator:
    """
    User-facing MicroMacro simulator. Creates an orchestrator per run
    and returns in-memory results.
    """

    def __init__(
        self,
        W: np.ndarray,
        micro_graphs: List[nx.Graph],
        beta_micro: float,
        gamma: float,
        beta_macro: float,
        tau_micro: float,
        T_end: float,
        macro_T: float = 1.0,
        model: int = 2,
        full_graph: Optional[nx.Graph] = None,
    ):
        self.W = W
        self.micro_graphs = micro_graphs
        self.beta_micro = beta_micro
        self.gamma = gamma
        self.beta_macro = beta_macro
        self.tau_micro = tau_micro
        self.T_end = T_end
        self.macro_T = macro_T
        self.model = model
        self.full_graph = full_graph

    def run(
        self,
        seed: Optional[int] = None,
        initial_community: int = 0,
        initial_node: Optional[int] = None,
    ) -> MicroMacroSimulationResult:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        orch = MicroMacroOrchestrator(
            W=self.W,
            micro_graphs=self.micro_graphs,
            beta_micro=self.beta_micro,
            gamma=self.gamma,
            beta_macro=self.beta_macro,
            tau_micro=self.tau_micro,
            T_end=self.T_end,
            macro_T=self.macro_T,
            model=self.model,
            full_graph=self.full_graph,
            layout_seed=seed if seed is not None else 0,
        )

        if not orch.micro_models:
            return ([], [], {}, [])
        if not (0 <= initial_community < len(orch.micro_models)):
            raise ValueError("initial_community is out of range")

        if initial_node is None:
            orch.micro_models[initial_community]._infect_node()
        else:
            orch.micro_models[initial_community]._infect_node(initial_node)

        return orch.run()


__all__ = ["MicroMacroSimulator"]
