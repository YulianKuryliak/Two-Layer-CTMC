import math
import random
from typing import List, Optional, Tuple

import networkx as nx

from .micro_engine import MicroEngine
from .._validation import nonnegative


class MicroModel(MicroEngine):
    """
    Micro-scale community simulator with horizon-safe advancing.

    G stores node states only; static weighted topology lives in _neighbors.
    Use the original input graph when graph-level edge queries are needed.
    """

    def __init__(
        self,
        infection_rate: float,
        recovery_rate: float,
        model: int = 2,
        rng: Optional[random.Random] = None,
    ):
        # A community owns its stream. Disposable midpoint probes must never
        # consume the parent's or another community's random numbers.
        if rng is None or rng is random:
            rng = random.Random()
        super().__init__(
            infection_rate,
            recovery_rate,
            model,
            track_counts=True,
            cache_neighbors=True,
            cache_events=True,
            rng=rng,
            compact_neighbors=True,
        )
        self.current_time = 0.0
        self._pending_rate: Optional[float] = None
        self._pending_origin: Optional[float] = None
        self._remaining_intensity_budget: Optional[float] = None

    def _invalidate_cache(self) -> None:
        super()._invalidate_cache()
        self._pending_rate = None
        self._pending_origin = None
        self._remaining_intensity_budget = None

    def _ensure_next_event(self):
        if self._cached_wait is None:
            rate = self.total_infection_rate + self.total_recovery_rate
            if not math.isfinite(rate):
                raise ValueError("Total event rate overflows; reduce rates or weights")
            if rate <= 0.0 or not self.infected_nodes:
                # A zero hazard consumes no integrated intensity. Retain any
                # outstanding threshold in case a future import reactivates it.
                return float("inf"), None, None, None
            before = self._get_rng_state()
            wait = None
            if self._remaining_intensity_budget is not None:
                wait = self._remaining_intensity_budget / rate
            wait, event = self._sample_next_event(waiting_time=wait)
            self._cached_wait, self._cached_event = wait, event
            self._cached_rng_before = before
            self._cached_rng_after = self._get_rng_state()
            self._pending_rate = rate
            self._pending_origin = self.current_time
            self._remaining_intensity_budget = None
        return (self._cached_wait, self._cached_event,
                self._cached_rng_before, self._cached_rng_after)

    def import_infection(self, *, at_time: Optional[float] = None) -> Optional[int]:
        """Import after advancing to its time; retain the unused SSA budget.

        If a_old was constant since the pending proposal, its remaining
        intensity is a_old * (t_old - t_import). The new wait is this quantity
        divided by a_new; only the event mark is sampled again. An inactive
        community with no pending threshold starts a fresh exponential clock.
        """
        when = self.current_time if at_time is None else float(at_time)
        if not math.isfinite(when) or when < self.current_time:
            raise ValueError("Import time must be finite and not precede local time")
        if not any(not d["infected"] and not d["recovered"] for _, d in self.G.nodes(data=True)):
            return None
        # Materialize any active clock before moving its time origin, including
        # consecutive imports invoked directly through this API.
        if self._cached_wait is None:
            self._ensure_next_event()
        budget = self._remaining_intensity_budget
        if self._cached_wait is not None and self._pending_origin is not None:
            remaining = self._pending_origin + self._cached_wait - when
            tolerance = 1e-12 * max(1.0, abs(when), abs(self._pending_origin + self._cached_wait))
            if remaining < -tolerance:
                raise ValueError("Advance local events to the import time before importing")
            budget = self._pending_rate * max(0.0, remaining)
        self.current_time = when
        node = super().import_infection()
        # _infect_node invalidates the old mark and updates all local rates.
        # Restore only the unspent threshold, never the consumed RNG state.
        self._remaining_intensity_budget = budget
        # Anchor the updated rate and proposal at the import time immediately.
        # This also accounts for elapsed intensity before a subsequent import.
        self._ensure_next_event()
        return node

    def add_edge(self, node1: int, node2: int, weight: float):
        """Register topology once, retaining the original neighbor/RNG order."""
        w = nonnegative(weight, "edge weight")
        if node1 == node2:
            raise ValueError("Self-loops are not supported")
        if node1 not in self.G or node2 not in self.G:
            raise ValueError("Add both nodes before adding an edge")
        if self._neighbors is None:
            self._neighbors = {}
        self._append_neighbor(node1, node2, w)
        self._append_neighbor(node2, node1, w)
        if self._cached_wait is not None:
            self._invalidate_cache()

    def simulate_until(
        self, t_end: float, *, record_events: bool = True
    ) -> Tuple[int, int, int, List[Tuple[float, str, int, Optional[int]]]]:
        """
        Advance time up to t_end, applying ONLY events whose occurrence time <= t_end.
        Returns (S, I, R, events) where each event is (time, etype, node, src_if_infection_else_None).
        With record_events=False, events is empty; advancing, RNG and cache
        behavior are identical. This is useful for disposable midpoint probes.
        """
        events: List[Tuple[float, str, int, Optional[int]]] = []
        t_end = nonnegative(t_end, "t_end")
        if t_end < self.current_time:
            raise ValueError("t_end must not precede the local event time")

        while self.current_time < t_end:
            wait_time, ev, rng_before, rng_after = self._ensure_next_event()

            # No events possible -> stop at horizon
            if wait_time == float("inf") or ev is None:
                break

            # The proposal has consumed its random numbers even when it is
            # beyond the horizon. Keep it cached, without rewinding the stream:
            # an intervening import must use fresh, unconditioned randomness.
            if self.current_time + wait_time > t_end:
                break

            if wait_time > 0 and self.current_time + wait_time <= self.current_time:
                raise ValueError("Event time cannot advance at these rates and time precision")
            # Apply the event
            self.current_time += wait_time
            etype, node, src = self._apply_event(ev)
            if record_events and etype != "none" and node is not None:
                events.append((self.current_time, etype, node, src))
            self._invalidate_cache()

        S, I, R = self.count_states()
        return S, I, R, events

    # --- lightweight cloning for midpoint probing (no RNG leakage) ---
    def clone(self) -> "MicroModel":
        copied_rng = random.Random()
        copied_rng.setstate(self.rng.getstate())
        m = MicroModel(self.infection_rate, self.recovery_rate, self.model, rng=copied_rng)
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
        m._copy_pending_event_from(self)
        return m

    def _copy_pending_event_from(self, src: "MicroModel") -> None:
        # Event tuples and random.Random states are immutable. Copying the
        # pending proposal keeps probe and real trajectory consistent until
        # their states diverge, without consuming a second proposal.
        self._cached_wait = src._cached_wait
        self._cached_event = src._cached_event
        self._cached_rng_before = src._cached_rng_before
        self._cached_rng_after = src._cached_rng_after
        self._pending_rate = src._pending_rate
        self._pending_origin = src._pending_origin
        self._remaining_intensity_budget = src._remaining_intensity_budget

    def refresh_from(self, src: "MicroModel") -> None:
        self.current_time = src.current_time
        self.infected_nodes = list(src.infected_nodes)
        self.total_infection_rate = src.total_infection_rate
        self.total_recovery_rate = src.total_recovery_rate
        self.S_count = src.S_count
        self.I_count = src.I_count
        self.R_count = src.R_count
        self._neighbors = src._neighbors
        for node in self.G.nodes():
            dst = self.G.nodes[node]
            srcd = src.G.nodes[node]
            dst["infected"] = srcd["infected"]
            dst["recovered"] = srcd["recovered"]
            dst["sum_of_weights_i"] = srcd["sum_of_weights_i"]
        self.rng.setstate(src.rng.getstate())
        self._copy_pending_event_from(src)
