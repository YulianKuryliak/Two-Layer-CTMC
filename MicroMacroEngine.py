import math
import random
from typing import Any, List, Optional, Tuple

import networkx as nx
import numpy as np

from MicroEngine import MicroEngine


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


class MacroEngine:
    """
    Macro-scale engine with susceptibility-aware, size-normalized hazards.

    hazard_ij(t) = beta_macro * T * W[i,j] * (I_i / N_i) * (S_j / N_j)
    """

    def __init__(
        self,
        W: np.ndarray,
        beta_macro: float,
        T: float = 1.0,
        community_sizes: Optional[List[int]] = None,
    ):
        self.W = W
        self.beta_macro = beta_macro
        self.T = T
        self.n = W.shape[0]
        self._frac_buffer = np.zeros_like(W, dtype=float)
        self._hazards_buffer = np.zeros_like(W, dtype=float)
        self._flat = np.zeros(W.size, dtype=float)
        self._flat_cumsum = np.zeros(W.size, dtype=float)

        # Sizes N_i of each community i
        if community_sizes is None:
            # Fallback: all communities of size 1 (keeps behaviour sane if sizes are missing)
            self.N = np.ones(self.n, dtype=float)
        else:
            self.N = np.asarray(community_sizes, dtype=float)
            assert self.N.shape[0] == self.n, "community_sizes length must match W.shape[0]"

        self.hazards = np.zeros_like(W, dtype=float)
        self.total_hazard = 0.0

    def _compute_hazards_matrix(self, I_counts: List[int], S_counts: List[int]) -> np.ndarray:
        """
        λ_ij = β T W_ij * (I_i / N_i) * (S_j / N_j)
        """
        I = np.asarray(I_counts, dtype=float).reshape(-1, 1)  # (n,1)
        S = np.asarray(S_counts, dtype=float).reshape(1, -1)  # (1,n)

        N_i = self.N.reshape(-1, 1)  # (n,1)
        N_j = self.N.reshape(1, -1)  # (1,n)

        np.matmul(I / N_i, S / N_j, out=self._frac_buffer)
        np.multiply(self.W, self._frac_buffer, out=self._hazards_buffer)
        self._hazards_buffer *= (self.beta_macro * self.T)
        return self._hazards_buffer

    def update_hazards(self, I_counts: List[int], S_counts: List[int]):
        self.hazards = self._compute_hazards_matrix(I_counts, S_counts)
        self.total_hazard = float(self.hazards.sum())
        flat = self.hazards.ravel()
        np.copyto(self._flat, flat)
        np.cumsum(self._flat, out=self._flat_cumsum)

    def total_hazard_given(self, I_counts: List[int], S_counts: List[int]) -> float:
        return float(self._compute_hazards_matrix(I_counts, S_counts).sum())

    def sample_transfer(self) -> Tuple[int, int]:
        """
        Sample directed edge (i,j) proportional to current hazard_ij.
        Returns (-1, -1) if no macro hazard.
        """
        if self.total_hazard <= 0.0:
            return -1, -1
        thresh = random.random() * self.total_hazard
        idx = int(np.searchsorted(self._flat_cumsum, thresh))
        i, j = divmod(idx, self.n)
        return i, j


class Orchestrator:
    """
    Orchestrates micro and macro layers (susceptibility-aware hazards).
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
        layout_seed: Optional[int] = 0,
    ):
        self.micro_models: List[MicroModel] = []
        for G in micro_graphs:
            m = MicroModel(beta_micro, gamma, model=model)
            for node in G.nodes():
                m.add_node(node)
            for u, v, data in G.edges(data=True):
                m.add_edge(u, v, weight=data.get("weight", 1.0))
            self.micro_models.append(m)

        # Community sizes for normalized macro hazards
        community_sizes = [G.number_of_nodes() for G in micro_graphs]

        self.macro = MacroEngine(
            W=W,
            beta_macro=beta_macro,
            T=macro_T,
            community_sizes=community_sizes,
        )
        self.tau_micro = float(tau_micro)
        self.T_end = float(T_end)

        self.times: List[float] = []
        self.I_total: List[int] = []
        self.logs = {i: {"times": [], "S": [], "I": [], "R": []} for i in range(len(self.micro_models))}
        self.event_log: List[dict] = []

        if full_graph is not None:
            self.full_graph = full_graph.copy()
        else:
            self.full_graph = nx.Graph()
            for m in self.micro_models:
                self.full_graph.add_nodes_from(m.G.nodes())
                self.full_graph.add_edges_from(m.G.edges())

        self._node_to_comm: dict[int, int] = {}
        for idx, m in enumerate(self.micro_models):
            for n in m.G.nodes():
                self._node_to_comm[n] = idx

        # fixed layout (kept, but plotting calls removed for data collection use-case)
        self._pos = nx.spring_layout(self.full_graph, seed=layout_seed)

    # ----- helpers -----

    def _gather_state_per_community(self):
        state_per_community = []
        for m in self.micro_models:
            S_nodes = [n for n, d in m.G.nodes(data=True) if (not d["infected"]) and (not d["recovered"])]
            I_nodes = [n for n, d in m.G.nodes(data=True) if d["infected"]]
            R_nodes = [n for n, d in m.G.nodes(data=True) if d["recovered"]]
            state_per_community.append({"S": S_nodes, "I": I_nodes, "R": R_nodes})
        return state_per_community

    def _counts_arrays(self) -> Tuple[List[int], List[int], List[int]]:
        S_arr, I_arr, R_arr = [], [], []
        for m in self.micro_models:
            S, I, R = m.count_states()
            S_arr.append(S)
            I_arr.append(I)
            R_arr.append(R)
        return S_arr, I_arr, R_arr

    # --- logging with correct hazard snapshot (batch) ---
    def _log_micro_events_batch(
        self,
        hazard_matrix: np.ndarray,
        pending: List[Tuple[int, float, str, int, Optional[int]]],  # (community, ev_time, etype, node, src)
        last_event_time: float,
    ) -> float:
        total_hazard_snapshot = float(hazard_matrix.sum())
        hazard_snapshot = hazard_matrix.copy()
        states_snapshot = self._gather_state_per_community()
        for community, ev_time, etype, node, src in pending:
            self.event_log.append(
                {
                    "time": ev_time,
                    "wait_time": ev_time - last_event_time,
                    "event_type": etype,
                    "mode": "micro",
                    "community": community,
                    "node": node,
                    "src": src,
                    "hazard_matrix": hazard_snapshot,
                    "total_hazard": total_hazard_snapshot,
                    "states": states_snapshot,
                }
            )
            last_event_time = ev_time
        return last_event_time

    # ----- main loop -----

    def run(self) -> Tuple[List[float], List[int], Any, List[dict]]:
        t = 0.0
        last_event_time = 0.0

        # Initialize hazards with susceptibility-aware macro
        S0, I0, R0 = self._counts_arrays()
        self.macro.update_hazards(I0, S0)

        # >>> Initial snapshot at t=0.0 (needed for reliable resampling) <<<
        self.times.append(0.0)
        self.I_total.append(sum(I0))
        for idx, (S, I, R) in enumerate(zip(S0, I0, R0)):
            self.logs[idx]["times"].append(0.0)
            self.logs[idx]["S"].append(S)
            self.logs[idx]["I"].append(I)
            self.logs[idx]["R"].append(R)
        # <<< end initial snapshot >>>

        # Gillespie integral threshold for macro events
        thresh_int = -math.log(random.random())
        int_accum = 0.0

        while t < self.T_end:
            dt = min(self.tau_micro, self.T_end - t)
            t_mid = t + 0.5 * dt

            # --- midpoint hazard via micro snapshots (no RNG leakage) ---
            rng_state = random.getstate()
            clones = [m.clone() for m in self.micro_models]
            for clone in clones:
                clone.simulate_until(t_mid)
            S_mid = [clone.count_states()[0] for clone in clones]
            I_mid = [clone.count_states()[1] for clone in clones]
            random.setstate(rng_state)

            total_hazard_mid = self.macro.total_hazard_given(I_mid, S_mid)

            if total_hazard_mid <= 0.0:
                t_next = t + dt
                pending_micro: List[Tuple[int, float, str, int, Optional[int]]] = []
                S_list: List[int] = []
                I_list: List[int] = []
                for idx, m in enumerate(self.micro_models):
                    S, I, R, events = m.simulate_until(t_next)
                    S_list.append(S)
                    I_list.append(I)
                    for ev_time, ev_type, node, src in events:
                        pending_micro.append((idx, ev_time, ev_type, node, src))

                    # per-community SIR time series
                    self.logs[idx]["times"].append(t_next)
                    self.logs[idx]["S"].append(S)
                    self.logs[idx]["I"].append(I)
                    self.logs[idx]["R"].append(R)

                # global series
                self.times.append(t_next)
                self.I_total.append(sum(I_list))

                # recompute hazards at t_next (snapshot)
                self.macro.update_hazards(I_list, S_list)

                # Now log buffered micro events with the t_next snapshot
                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time,
                )

                t = t_next
                continue

            # midpoint rectangle accumulation: check if threshold crosses in this [t, t+dt]
            if int_accum + total_hazard_mid * dt >= thresh_int:
                # A MACRO EVENT occurs inside the interval at t_event
                t_event = t + (thresh_int - int_accum) / total_hazard_mid

                # advance real micros to t_event; buffer events
                pending_micro: List[Tuple[int, float, str, int, Optional[int]]] = []
                S_event: List[int] = []
                I_event: List[int] = []
                for idx, micro_model in enumerate(self.micro_models):
                    S, I, R, events = micro_model.simulate_until(t_event)
                    S_event.append(S)
                    I_event.append(I)
                    for ev_time, ev_type, node, src in events:
                        pending_micro.append((idx, ev_time, ev_type, node, src))

                # recompute hazards at t_event (snapshot)
                self.macro.update_hazards(I_event, S_event)

                # log the buffered micro events using the t_event snapshot
                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time,
                )

                # sample and apply the macro transfer at t_event
                i, j = self.macro.sample_transfer()
                if j != i:
                    self.micro_models[j].current_time = t_event
                    node = self.micro_models[j].import_infection()
                    # After import, update hazards again to reflect new state
                    S_post, I_post, _ = self._counts_arrays()
                    self.macro.update_hazards(I_post, S_post)

                    self.event_log.append(
                        {
                            "time": t_event,
                            "wait_time": t_event - last_event_time,
                            "event_type": "transfer",
                            "mode": "macro",
                            "community": j,
                            "src": i,
                            "node": node,
                            "hazard_matrix": self.macro.hazards.copy(),
                            "total_hazard": self.macro.total_hazard,
                            "states": self._gather_state_per_community(),
                        }
                    )
                    last_event_time = t_event

                # Log state at t_event
                self.times.append(t_event)
                self.I_total.append(sum(m.count_states()[1] for m in self.micro_models))
                for idx, m in enumerate(self.micro_models):
                    S, I, R = m.count_states()
                    self.logs[idx]["times"].append(t_event)
                    self.logs[idx]["S"].append(S)
                    self.logs[idx]["I"].append(I)
                    self.logs[idx]["R"].append(R)

                # advance time & reset integral
                t = t_event
                # draw fresh macro threshold and reset integral accumulator
                thresh_int = -math.log(random.random())
                int_accum = 0.0

            else:
                # NO MACRO EVENT in this interval: advance micros to t+dt; buffer events
                t_next = t + dt
                pending_micro: List[Tuple[int, float, str, int, Optional[int]]] = []
                S_list: List[int] = []
                I_list: List[int] = []
                for idx, m in enumerate(self.micro_models):
                    S, I, R, events = m.simulate_until(t_next)
                    S_list.append(S)
                    I_list.append(I)
                    for ev_time, ev_type, node, src in events:
                        pending_micro.append((idx, ev_time, ev_type, node, src))

                    # per-community SIR time series
                    self.logs[idx]["times"].append(t_next)
                    self.logs[idx]["S"].append(S)
                    self.logs[idx]["I"].append(I)
                    self.logs[idx]["R"].append(R)

                self.times.append(t_next)
                self.I_total.append(sum(I_list))

                # accumulate hazard integral with midpoint rectangle
                int_accum += total_hazard_mid * dt

                # recompute hazards at t_next and then log buffered micro events
                self.macro.update_hazards(I_list, S_list)

                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time,
                )

                t = t_next

        return self.times, self.I_total, self.logs, self.event_log


__all__ = ["MicroModel", "MacroEngine", "Orchestrator"]
