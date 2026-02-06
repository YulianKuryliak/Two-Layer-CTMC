import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..engines.micro_engine import MicroEngine
from ..types import InfectionEvent, MicroRunRow, MicroSimulationResult


class MicroOrchestrator:
    """
    Orchestrates a single microscopic simulation on a full graph
    and returns per-community time series without any I/O.
    """

    def __init__(
        self,
        full_graph: nx.Graph,
        comm_nodes: List[List[int]],
        infection_rate: float,
        recovery_rate: float,
        model: int = 2,
        rng: Optional[random.Random] = None,
    ):
        self.full_graph = full_graph
        self.comm_nodes = comm_nodes
        self.rng = random.Random() if rng is None else rng

        self.sim = MicroEngine(
            infection_rate=infection_rate,
            recovery_rate=recovery_rate,
            model=model,
            track_counts=False,
            cache_neighbors=False,
            cache_events=False,
            rng=self.rng,
        )
        for n in full_graph.nodes():
            self.sim.add_node(n)
        for u, v, d in full_graph.edges(data=True):
            self.sim.add_edge(u, v, weight=d.get("weight", 1.0))

        self._node_to_comm: Dict[int, int] = {}
        for ci, nodes in enumerate(comm_nodes):
            for n in nodes:
                self._node_to_comm[n] = ci

    @staticmethod
    def _counts_SIR_per_community(
        sim: MicroEngine,
        comm_nodes: List[List[int]],
    ) -> List[Tuple[int, int, int]]:
        infected_set = set(sim.infected_nodes)
        recovered_set = {n for n, a in sim.G.nodes(data=True) if a.get("recovered", False)}
        out: List[Tuple[int, int, int]] = []
        for nodes in comm_nodes:
            I = sum(1 for n in nodes if n in infected_set)
            R = sum(1 for n in nodes if n in recovered_set)
            S = len(nodes) - I - R
            out.append((S, I, R))
        return out

    def seed_infection(self, initial_node: Optional[int] = None) -> Optional[int]:
        if initial_node is None:
            nodes = list(self.full_graph.nodes())
            if not nodes:
                return None
            initial_node = self.rng.choice(nodes)
        self.sim._infect_node(initial_node)
        return initial_node

    def run(
        self,
        T_end: float,
        dt_out: float,
        initial_node: Optional[int] = None,
    ) -> MicroSimulationResult:
        if not self.sim.infected_nodes:
            initial_node = self.seed_infection(initial_node)

        current_counts = self._counts_SIR_per_community(self.sim, self.comm_nodes)
        infection_events: List[InfectionEvent] = []
        if initial_node is not None and current_counts:
            seed_comm = self._node_to_comm.get(initial_node, 0)
            if 0 <= seed_comm < len(current_counts):
                infection_events.append((0.0, seed_comm, "intra", current_counts[seed_comm][1]))

        rows: List[MicroRunRow] = []
        t = 0.0
        t_next_out = 0.0
        EPS = 1e-9

        while t < T_end:
            dt, event = self.sim.simulate_step(return_details=True)

            if dt == float("inf"):
                # hold last state
                while t_next_out <= T_end + EPS:
                    if t_next_out >= t - EPS:
                        for ci, (S, I, R) in enumerate(current_counts):
                            rows.append((ci, float(t_next_out), S, I, R))
                    t_next_out += dt_out
                break

            t_event = t + dt
            pending_infection: Optional[Tuple[float, int, str]] = None

            if event is not None and event[0] == "infection":
                _, src, dst = event
                dst_comm = self._node_to_comm.get(int(dst), 0)
                src_comm = self._node_to_comm.get(int(src), dst_comm)
                inf_type = "intra" if src_comm == dst_comm else "inter"
                pending_infection = (t_event, dst_comm, inf_type)

            # sample & hold until event
            while t_next_out < t_event and t_next_out <= T_end + EPS:
                if t_next_out >= t - EPS:
                    for ci, (S, I, R) in enumerate(current_counts):
                        rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out

            # update time and counts
            t = t_event
            current_counts = self._counts_SIR_per_community(self.sim, self.comm_nodes)

            if pending_infection is not None and current_counts:
                t_inf, dst_comm, inf_type = pending_infection
                if 0 <= dst_comm < len(current_counts):
                    infection_events.append((t_inf, dst_comm, inf_type, current_counts[dst_comm][1]))

        # finish to T_end
        while t_next_out <= T_end + EPS:
            for ci, (S, I, R) in enumerate(current_counts):
                rows.append((ci, float(t_next_out), S, I, R))
            t_next_out += dt_out

        return {
            "rows": rows,
            "infection_events": infection_events,
            "initial_node": initial_node,
        }


__all__ = ["MicroOrchestrator"]
