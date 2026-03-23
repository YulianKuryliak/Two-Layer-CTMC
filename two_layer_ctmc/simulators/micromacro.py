import random
import time
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
        verbose_steps: bool = False,
        phase_timing: bool = False,
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
        self.verbose_steps = bool(verbose_steps)
        self.phase_timing = bool(phase_timing)

        self.community_sizes: Optional[List[int]] = None
        self.alphas: Optional[List[float]] = None
        if full_graph is not None:
            cs = full_graph.graph.get("community_sizes")
            al = full_graph.graph.get("alphas")
            if isinstance(cs, list) and isinstance(al, list):
                self.community_sizes = [int(x) for x in cs]
                self.alphas = [float(x) for x in al]

    def run(
        self,
        seed: Optional[int] = None,
        initial_community: int = 0,
        initial_node: Optional[int] = None,
    ) -> MicroMacroSimulationResult:
        t_init_start = time.perf_counter()
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
            community_sizes=self.community_sizes,
            alphas=self.alphas,
            full_graph=self.full_graph,
            verbose_steps=self.verbose_steps,
        )
        t_init_done = time.perf_counter()
        if self.phase_timing:
            print(f"[timing] orchestrator_init={t_init_done - t_init_start:.3f}s")

        if not orch.micro_models:
            return ([], [], {}, [])
        if not (0 <= initial_community < len(orch.micro_models)):
            raise ValueError("initial_community is out of range")

        seeded_node: Optional[int] = initial_node
        if initial_node is None:
            before = set(orch.micro_models[initial_community].infected_nodes)
            orch.micro_models[initial_community]._infect_node()
            after = set(orch.micro_models[initial_community].infected_nodes)
            delta = list(after - before)
            seeded_node = int(delta[0]) if delta else None
        else:
            orch.micro_models[initial_community]._infect_node(initial_node)

        orch.event_log.append(
            {
                "time": 0.0,
                "wait_time": 0.0,
                "event_type": "seed",
                "mode": "seed",
                "community": int(initial_community),
                "node": seeded_node,
                "src": None,
            }
        )

        t_loop_start = time.perf_counter()
        result = orch.run()
        t_loop_done = time.perf_counter()
        if self.phase_timing:
            print(f"[timing] main_loop={t_loop_done - t_loop_start:.3f}s")
        return result


__all__ = ["MicroMacroSimulator"]
