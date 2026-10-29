"""Full-network Gillespie simulation with a sampled community output grid."""
import random
from typing import List, Optional

import networkx as nx

from .._validation import positive
from ..engines.micro_engine import MicroEngine
from ..types import MicroSimulationResult


class MicroOrchestrator:
    def __init__(self, full_graph: nx.Graph, comm_nodes: List[List[int]],
                 infection_rate: float, recovery_rate: float, model: int = 2,
                 rng: Optional[random.Random] = None):
        self.full_graph = full_graph
        self.comm_nodes = comm_nodes
        self.rng = random.Random() if rng is None else rng
        self.sim = MicroEngine(infection_rate, recovery_rate, model, rng=self.rng)
        self.sim.bind_static_graph(full_graph)
        flattened = [n for nodes in comm_nodes for n in nodes]
        if not comm_nodes or any(not nodes for nodes in comm_nodes):
            raise ValueError("Communities must be nonempty")
        if len(flattened) != len(set(flattened)) or set(flattened) != set(full_graph):
            raise ValueError("Communities must partition the full graph")
        self.node_to_community = {n: ci for ci, nodes in enumerate(comm_nodes) for n in nodes}

    def run(self, T_end: float, dt_out: float,
            initial_node: Optional[int] = None) -> MicroSimulationResult:
        horizon = positive(T_end, "T_end")
        step = positive(dt_out, "dt_out")
        if initial_node is None:
            initial_node = self.rng.choice(list(self.full_graph))
        if initial_node not in self.full_graph:
            raise ValueError("initial_node must belong to the full graph")
        self.sim._infect_node(initial_node)
        counts = [[len(nodes), 0, 0] for nodes in self.comm_nodes]
        seed_comm = self.node_to_community[initial_node]
        counts[seed_comm][0] -= 1
        counts[seed_comm][1] += 1
        events = [{"time": 0.0, "wait_time": 0.0, "event_type": "seed",
                   "mode": "seed", "community": seed_comm, "node": int(initial_node), "src": None}]
        rows = []
        output_index = 0
        next_output = 0.0
        current_time = 0.0

        def emit(time):
            for ci, (S, I, R) in enumerate(counts):
                rows.append((ci, float(time), S, I, R))

        while current_time < horizon:
            wait, event = self.sim._sample_next_event()
            event_time = current_time + wait
            if event_time > horizon:
                break
            if event is None:
                raise RuntimeError("Positive event rate produced no enabled event")
            if event_time <= current_time:
                raise ValueError("Event time cannot advance at these rates and time precision")
            while next_output < event_time:
                emit(next_output)
                output_index += 1
                next_output = output_index * step
            kind, node, source = self.sim._apply_event(event)
            ci = self.node_to_community[node]
            if kind == "infection":
                counts[ci][0] -= 1
                counts[ci][1] += 1
            else:
                counts[ci][1] -= 1
                counts[ci][0 if self.sim.model == 1 else 2] += 1
            events.append({"time": event_time, "wait_time": wait, "event_type": kind,
                           "mode": "micro", "community": ci, "node": int(node), "src": source})
            current_time = event_time
        while next_output < horizon:
            emit(next_output)
            output_index += 1
            next_output = output_index * step
        emit(horizon)
        return {"rows": rows, "events": events, "initial_node": int(initial_node)}
