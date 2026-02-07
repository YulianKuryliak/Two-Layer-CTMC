import math
import random
from typing import Any, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..engines.macro_engine import MacroEngine
from ..engines.micro_model import MicroModel


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
        alphas = self._compute_alphas(micro_graphs)

        self.macro = MacroEngine(
            W=W,
            beta_macro=beta_macro,
            T=macro_T,
            community_sizes=community_sizes,
            alphas=alphas,
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
    @staticmethod
    def _compute_alphas(micro_graphs: List[nx.Graph]) -> List[float]:
        alphas = []
        for G in micro_graphs:
            n = G.number_of_nodes()
            if n < 2:
                alphas.append(1.0)
                continue
            A = nx.to_numpy_array(G, weight="weight", dtype=float)
            deg = A.sum(axis=1)
            inv_sqrt_deg = np.zeros_like(deg)
            nonzero = deg > 0
            inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(deg[nonzero])
            D_inv_sqrt = np.diag(inv_sqrt_deg)
            L_norm = np.eye(n) - (D_inv_sqrt @ A @ D_inv_sqrt)
            eigvals = np.linalg.eigvalsh(L_norm)
            if eigvals.size < 2:
                alphas.append(1.0)
                continue
            lambda2_tilde = float(np.sort(eigvals)[-2])
            if lambda2_tilde <= 0.0:
                alphas.append(1.0)
                continue
            alphas.append(1.0 / lambda2_tilde)
        return alphas

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


MicroMacroOrchestrator = Orchestrator

__all__ = ["Orchestrator", "MicroMacroOrchestrator"]
