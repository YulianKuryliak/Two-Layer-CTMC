import math
import random
from typing import Any, List, Optional, Tuple

import networkx as nx


class MicroEngine:
    """
    Event-driven SIR/SIS on a static graph via Gillespie SSA.

    Optional knobs:
      - cache_neighbors: keep adjacency list with weights for faster iteration
      - track_counts: maintain S/I/R counters
      - cache_events: cache sampled event + RNG states for horizon-safe stepping
      - rng: inject RNG with random/uniform/expovariate/getstate/setstate
    """

    def __init__(
        self,
        infection_rate: float = 0.1,
        recovery_rate: float = 0.0,
        model: int = 2,
        *,
        track_counts: bool = False,
        cache_neighbors: bool = False,
        cache_events: bool = False,
        rng: Optional[random.Random] = None,
    ):
        self.G = nx.Graph()
        self.model = int(model)  # 1 = SIS, 2 = SIR
        self.infection_rate = float(infection_rate)
        self.recovery_rate = float(recovery_rate)
        self.infected_nodes: List[int] = []
        self.total_infection_rate = 0.0
        self.total_recovery_rate = 0.0

        self.track_counts = bool(track_counts)
        self.cache_neighbors = bool(cache_neighbors)
        self.cache_events = bool(cache_events)
        self.rng = random if rng is None else rng

        self.S_count = 0
        self.I_count = 0
        self.R_count = 0

        self._neighbors: Optional[dict[int, List[Tuple[int, float]]]] = (
            {} if self.cache_neighbors else None
        )
        self._cached_wait: Optional[float] = None
        self._cached_event: Optional[Tuple[str, Any]] = None
        self._cached_rng_before = None
        self._cached_rng_after = None

    # ---------- graph building ----------

    def add_node(self, node_id: int):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)
        if self.cache_neighbors:
            if self._neighbors is None:
                self._neighbors = {}
            self._neighbors[node_id] = []
        if self.track_counts:
            self.S_count += 1

    def add_edge(self, node1: int, node2: int, weight: float):
        if not (weight >= 0 and math.isfinite(weight)):
            raise ValueError("Edge weight must be non-negative and finite")
        w = float(weight)
        self.G.add_edge(node1, node2, weight=w)
        if self.cache_neighbors:
            if self._neighbors is None:
                self._neighbors = {}
            self._neighbors.setdefault(node1, []).append((node2, w))
            self._neighbors.setdefault(node2, []).append((node1, w))
        self._invalidate_cache()

    # ---------- RNG helpers ----------

    def _get_rng_state(self):
        if hasattr(self.rng, "getstate"):
            return self.rng.getstate()
        return None

    def _set_rng_state(self, state) -> None:
        if state is None:
            return
        if hasattr(self.rng, "setstate"):
            self.rng.setstate(state)

    # ---------- caching ----------

    def _invalidate_cache(self):
        if not self.cache_events:
            return
        self._cached_wait = None
        self._cached_event = None
        self._cached_rng_before = None
        self._cached_rng_after = None

    def _ensure_next_event(self):
        if self.cache_events:
            if self._cached_wait is None:
                rng_before = self._get_rng_state()
                wait_time, ev = self._sample_next_event()
                rng_after = self._get_rng_state()
                self._cached_wait = wait_time
                self._cached_event = ev
                self._cached_rng_before = rng_before
                self._cached_rng_after = rng_after
            return (
                self._cached_wait,
                self._cached_event,
                self._cached_rng_before,
                self._cached_rng_after,
            )

        rng_before = self._get_rng_state()
        wait_time, ev = self._sample_next_event()
        rng_after = self._get_rng_state()
        return wait_time, ev, rng_before, rng_after

    # ---------- neighbor access ----------

    def _neighbors_with_weights(self, node: int) -> List[Tuple[int, float]]:
        if self.cache_neighbors:
            if self._neighbors is None:
                return []
            return self._neighbors.get(node, [])
        return [(nbr, self.G[node][nbr]["weight"]) for nbr in self.G.neighbors(node)]

    def _susceptible_neighbors(self, node: int) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for nbr, w in self._neighbors_with_weights(node):
            nbrd = self.G.nodes[nbr]
            if (not nbrd["infected"]) and (not nbrd["recovered"]):
                out.append((nbr, w))
        return out

    def _choose_weighted_neighbor(self, neighbors: List[Tuple[int, float]]) -> Optional[int]:
        if not neighbors:
            return None
        wsum = sum(w for _, w in neighbors)
        if wsum <= 0.0:
            return None
        target = self.rng.uniform(0.0, wsum)
        cum = 0.0
        for n, w in neighbors:
            cum += w
            if cum > target:
                return n
        return None

    def _choose_infected_source(self) -> Optional[int]:
        if not self.infected_nodes or self.total_infection_rate <= 0.0:
            return None
        target = self.rng.uniform(0.0, self.total_infection_rate)
        cum = 0.0
        for node in self.infected_nodes:
            cum += self.infection_rate * self.G.nodes[node]["sum_of_weights_i"]
            if cum > target:
                return node
        return None

    # ---------- core state changes ----------

    def _infect_node(self, node: Optional[int] = None):
        if node is None:
            if self.G.number_of_nodes() == 0:
                return
            node = self.rng.choice(list(self.G.nodes))

        nd = self.G.nodes[node]
        if nd["infected"] or nd["recovered"]:
            return

        nd["infected"] = True
        nd["recovered"] = False
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        if self.track_counts:
            self.S_count -= 1
            self.I_count += 1

        s = 0.0
        for nbr, w in self._neighbors_with_weights(node):
            nbrd = self.G.nodes[nbr]
            if (not nbrd["infected"]) and (not nbrd["recovered"]):
                s += w
                self.total_infection_rate += self.infection_rate * w
            elif nbrd["infected"] and nbr != node:
                new_sum = nbrd["sum_of_weights_i"] - w
                if -1e-12 < new_sum < 0:
                    new_sum = 0.0
                nbrd["sum_of_weights_i"] = new_sum
                self.total_infection_rate -= self.infection_rate * w

        if -1e-12 < s < 0:
            s = 0.0
        nd["sum_of_weights_i"] = s

        if self.total_infection_rate < 0.0:
            self.total_infection_rate = 0.0
        self._invalidate_cache()

    def _recover_node(self, node: int):
        if not self.G.nodes[node]["infected"]:
            return

        node_sum = self.G.nodes[node]["sum_of_weights_i"]
        if node_sum:
            self.total_infection_rate -= self.infection_rate * node_sum
            if self.total_infection_rate < 0.0:
                self.total_infection_rate = 0.0

        if node in self.infected_nodes:
            self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.track_counts:
            self.I_count -= 1

        if self.model == 1:  # SIS
            for nbr, w in self._neighbors_with_weights(node):
                if self.G.nodes[nbr]["infected"]:
                    new_sum = self.G.nodes[nbr]["sum_of_weights_i"] + w
                    if -1e-12 < new_sum < 0:
                        new_sum = 0.0
                    self.G.nodes[nbr]["sum_of_weights_i"] = new_sum
                    self.total_infection_rate += self.infection_rate * w

            self.G.nodes[node]["infected"] = False
            self.G.nodes[node]["recovered"] = False
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
            if self.track_counts:
                self.S_count += 1

        else:  # SIR
            self.G.nodes[node]["infected"] = False
            self.G.nodes[node]["recovered"] = True
            self.G.nodes[node]["sum_of_weights_i"] = 0.0
            if self.track_counts:
                self.R_count += 1

        if self.total_infection_rate < 0.0:
            self.total_infection_rate = 0.0
        self._invalidate_cache()

    def _infect_neighbor(self, src: int) -> Optional[int]:
        neighbors = self._susceptible_neighbors(src)
        chosen = self._choose_weighted_neighbor(neighbors)
        if chosen is None:
            return None
        self._infect_node(chosen)
        return chosen

    # ---------- Gillespie sampling ----------

    def _sample_next_event(self) -> Tuple[float, Optional[Tuple[str, Any]]]:
        if not self.infected_nodes:
            return float("inf"), None

        total_rate = self.total_infection_rate + self.total_recovery_rate
        if total_rate < 1e-12:
            return float("inf"), None

        wait_time = self.rng.expovariate(total_rate)

        if self.rng.random() < (self.total_infection_rate / total_rate):
            src = self._choose_infected_source()
            if src is None:
                return wait_time, None
            neighbors = self._susceptible_neighbors(src)
            dst = self._choose_weighted_neighbor(neighbors)
            if dst is None:
                return wait_time, None
            return wait_time, ("infection", src, dst)

        target = self.rng.uniform(0.0, self.total_recovery_rate)
        cum = 0.0
        chosen = None
        for node in self.infected_nodes:
            cum += self.recovery_rate
            if cum > target:
                chosen = node
                break
        if chosen is None:
            return wait_time, None
        return wait_time, ("recovery", chosen)

    def _apply_event(self, event: Tuple[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
        etype = event[0]
        if etype == "infection":
            if len(event) < 3:
                return "none", None, None
            _, src, dst = event
            if dst is None:
                return "none", None, None
            self._infect_node(dst)
            return "infection", dst, src
        if etype == "recovery":
            node = event[1]
            if node is None:
                return "none", None, None
            self._recover_node(node)
            return "recovery", node, None
        return "none", None, None

    def simulate_step(self, return_details: bool = False):
        wait_time, ev = self._sample_next_event()
        if wait_time == float("inf") or ev is None:
            if return_details:
                return float("inf"), None
            return float("inf"), None, None

        etype, node, src = self._apply_event(ev)
        if return_details:
            if etype == "infection":
                return wait_time, ("infection", src, node)
            if etype == "recovery":
                return wait_time, ("recovery", node, None)
            return wait_time, None
        return wait_time, etype, node

    # ---------- convenience ----------

    def count_states(self) -> Tuple[int, int, int]:
        if self.track_counts:
            return self.S_count, self.I_count, self.R_count
        s = i = r = 0
        for _, data in self.G.nodes(data=True):
            if data.get("infected"):
                i += 1
            elif data.get("recovered"):
                r += 1
            else:
                s += 1
        return s, i, r

    def import_infection(self) -> Optional[int]:
        susceptibles = [
            n for n, d in self.G.nodes(data=True)
            if (not d["infected"]) and (not d["recovered"])
        ]
        if not susceptibles:
            return None
        node = self.rng.choice(susceptibles)
        self._infect_node(node)
        return node
