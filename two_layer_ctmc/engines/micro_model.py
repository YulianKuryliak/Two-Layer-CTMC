import random
from typing import List, Optional, Tuple

import networkx as nx

from .micro_engine import MicroEngine


class MicroModel(MicroEngine):
    """
    Micro-scale community simulator with horizon-safe advancing.
    """

    def __init__(
        self,
        infection_rate: float,
        recovery_rate: float,
        model: int = 2,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            infection_rate,
            recovery_rate,
            model,
            track_counts=True,
            cache_neighbors=True,
            cache_events=True,
            rng=rng,
        )
        self.current_time = 0.0

    def simulate_until(self, t_end: float) -> Tuple[int, int, int, List[Tuple[float, str, int, Optional[int]]]]:
        """
        Advance time up to t_end, applying ONLY events whose occurrence time <= t_end.
        Returns (S, I, R, events) where each event is (time, etype, node, src_if_infection_else_None).
        """
        events: List[Tuple[float, str, int, Optional[int]]] = []

        while self.current_time < t_end:
            wait_time, ev, rng_before, rng_after = self._ensure_next_event()

            # No events possible -> stop at horizon
            if wait_time == float("inf") or ev is None:
                self._invalidate_cache()
                break

            # Next event beyond horizon -> restore RNG and stop
            if self.current_time + wait_time > t_end:
                self._set_rng_state(rng_before)
                break

            # Apply the event
            self._set_rng_state(rng_after)
            self.current_time += wait_time
            etype, node, src = self._apply_event(ev)
            if etype != "none" and node is not None:
                events.append((self.current_time, etype, node, src))
            self._invalidate_cache()

        S, I, R = self.count_states()
        return S, I, R, events

    # --- lightweight cloning for midpoint probing (no RNG leakage) ---
    def clone(self) -> "MicroModel":
        m = MicroModel(self.infection_rate, self.recovery_rate, self.model, rng=self.rng)
        m.current_time = self.current_time
        # copy node states but avoid duplicating static topology (edges handled via _neighbors)
        m.G = nx.Graph()
        m.G.add_nodes_from(self.G.nodes(data=True))
        m.infected_nodes = list(self.infected_nodes)
        m.total_infection_rate = self.total_infection_rate
        m.total_recovery_rate = self.total_recovery_rate
        m.S_count = self.S_count
        m.I_count = self.I_count
        m.R_count = self.R_count
        m._neighbors = self._neighbors  # topology is static; share reference
        m._invalidate_cache()
        return m
