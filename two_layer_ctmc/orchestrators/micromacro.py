"""Midpoint synchronization of local CTMCs and aggregate imports."""
import math
import random
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .._validation import graph, matrix, model_id, nonnegative, positive
from ..engines.macro_engine import MacroEngine
from ..engines.micro_model import MicroModel
from ..types import MicroMacroSimulationResult


class MicroMacroOrchestrator:
    """Run the basic finite-step TwoScale approximation with private RNG streams."""

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
        rng: Optional[random.Random] = None,
    ):
        self.tau_micro = positive(tau_micro, "tau_micro")
        self.T_end = positive(T_end, "T_end")
        beta_micro = nonnegative(beta_micro, "beta_micro")
        gamma = nonnegative(gamma, "gamma")
        model = model_id(model)
        if not micro_graphs:
            raise ValueError("Provide at least one community graph")
        W = matrix(W, len(micro_graphs))
        seen = set()
        for index, community in enumerate(micro_graphs):
            graph(community, f"micro_graphs[{index}]")
            if seen.intersection(community.nodes()):
                raise ValueError("Community node sets must be disjoint")
            seen.update(community.nodes())
        self.rng = random.Random() if rng is None else rng
        self.micro_models: List[MicroModel] = []
        for community in micro_graphs:
            local_rng = random.Random(self.rng.getrandbits(128))
            local = MicroModel(beta_micro, gamma, model=model, rng=local_rng)
            for node in community.nodes():
                local.add_node(node)
            for source, destination, data in community.edges(data=True):
                local.add_edge(source, destination, weight=data.get("weight", 1.0))
            self.micro_models.append(local)
        self.macro = MacroEngine(
            W, beta_macro, macro_T,
            community_sizes=[community.number_of_nodes() for community in micro_graphs],
            rng=self.rng,
        )
        self.times: List[float] = []
        self.I_total: List[int] = []
        self.logs = {
            index: {"times": [], "S": [], "I": [], "R": []}
            for index in range(len(self.micro_models))
        }
        self.event_log: List[dict] = []

    def _counts_arrays(self) -> Tuple[List[int], List[int], List[int]]:
        counts = [model.count_states() for model in self.micro_models]
        return tuple([count[index] for count in counts] for index in range(3))

    def _snapshot(self, time: float) -> None:
        self.times.append(float(time))
        total = 0
        for index, model in enumerate(self.micro_models):
            S, I, R = model.count_states()
            total += I
            log = self.logs[index]
            log["times"].append(float(time))
            log["S"].append(S)
            log["I"].append(I)
            log["R"].append(R)
        self.I_total.append(total)

    def _advance(self, horizon: float):
        susceptible, infected, pending = [], [], []
        for index, model in enumerate(self.micro_models):
            S, I, _, events = model.simulate_until(horizon)
            susceptible.append(S)
            infected.append(I)
            pending.extend((index, *event) for event in events)
        return susceptible, infected, pending

    def _log_micro_events(self, pending, last_event_time: float) -> float:
        for community, event_time, kind, node, source in sorted(pending, key=lambda item: item[1]):
            if event_time < last_event_time:
                raise RuntimeError("Local events are not chronological")
            self.event_log.append({
                "time": event_time,
                "wait_time": event_time - last_event_time,
                "event_type": kind,
                "mode": "micro",
                "community": community,
                "node": node,
                "src": source,
            })
            last_event_time = event_time
        return last_event_time

    def _threshold(self) -> float:
        # random.Random can return zero; an exponential threshold must be finite.
        uniform = self.rng.random()
        while uniform == 0.0:
            uniform = self.rng.random()
        return -math.log(uniform)

    def run(self) -> MicroMacroSimulationResult:
        """Advance to the requested horizon and return dense community snapshots."""
        if self.times:
            raise ValueError("Create a new orchestrator for each run")
        time = 0.0
        last_event_time = 0.0
        susceptible, infected, _ = self._counts_arrays()
        self.macro.update_hazards(infected, susceptible)
        self._snapshot(0.0)
        threshold = self._threshold()
        accumulated = 0.0
        probes = [model.clone() for model in self.micro_models]

        while time < self.T_end:
            grid_end = min(time + self.tau_micro, self.T_end)
            if grid_end <= time:
                raise ValueError("tau_micro is too small to advance floating-point time")
            while time < grid_end:
                duration = grid_end - time
                midpoint = time + 0.5 * duration
                for probe, model in zip(probes, self.micro_models):
                    probe.refresh_from(model)
                    probe.simulate_until(midpoint, record_events=False)
                states = [probe.count_states() for probe in probes]
                S_mid = [state[0] for state in states]
                I_mid = [state[1] for state in states]
                hazard_mid = self.macro.total_hazard_given(I_mid, S_mid)

                if hazard_mid > 0.0 and accumulated + hazard_mid * duration >= threshold:
                    event_time = time + (threshold - accumulated) / hazard_mid
                    if event_time <= time or event_time > grid_end:
                        raise ValueError("Inter-community rates cannot resolve event time; reduce rates or time step")
                    susceptible, infected, pending = self._advance(event_time)
                    self.macro.update_hazards(infected, susceptible)
                    last_event_time = self._log_micro_events(pending, last_event_time)
                    source, destination = self.macro.sample_transfer()
                    if source >= 0 and destination >= 0:
                        node = self.micro_models[destination].import_infection(at_time=event_time)
                        if node is not None:
                            susceptible[destination] -= 1
                            infected[destination] += 1
                        self.macro.update_hazards(infected, susceptible)
                        self.event_log.append({
                            "time": event_time,
                            "wait_time": event_time - last_event_time,
                            "event_type": "transfer",
                            "mode": "macro",
                            "community": destination,
                            "node": node,
                            "src": source,
                        })
                        last_event_time = event_time
                    self._snapshot(event_time)
                    time = event_time
                    threshold = self._threshold()
                    accumulated = 0.0
                    continue

                susceptible, infected, pending = self._advance(grid_end)
                self.macro.update_hazards(infected, susceptible)
                last_event_time = self._log_micro_events(pending, last_event_time)
                self._snapshot(grid_end)
                accumulated += hazard_mid * duration
                time = grid_end
                break

        return self.times, self.I_total, self.logs, self.event_log
