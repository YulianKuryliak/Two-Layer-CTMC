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

    dense_logs=False retains community change-point logs; their values are
    held constant until the next logged time. Global times and event logs
    are unaffected. The default retains the original dense output API.
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
        community_sizes: Optional[List[int]] = None,
        alphas: Optional[List[float]] = None,
        full_graph: Optional[nx.Graph] = None,
        verbose_steps: bool = False,
        alpha_mode: str = "off",
        correction_mode: str = "off",
        dense_logs: bool = True,
        stop_at_absorption: bool = False,
        micro_rngs: Optional[List[random.Random]] = None,
    ):
        if micro_rngs is not None and len(micro_rngs) != len(micro_graphs):
            raise ValueError("Provide one private RNG per community")
        if micro_rngs is not None and len({id(rng) for rng in micro_rngs}) != len(micro_rngs):
            raise ValueError("Community RNGs must be distinct objects")
        self.micro_models: List[MicroModel] = []
        for community, G in enumerate(micro_graphs):
            rng = None if micro_rngs is None else micro_rngs[community]
            m = MicroModel(beta_micro, gamma, model=model, rng=rng)
            for node in G.nodes():
                m.add_node(node)
            for u, v, data in G.edges(data=True):
                m.add_edge(u, v, weight=data.get("weight", 1.0))
            self.micro_models.append(m)

        # Community sizes / alphas for normalized macro hazards.
        if community_sizes is None:
            community_sizes = [G.number_of_nodes() for G in micro_graphs]
        if alphas is None:
            alphas = self._compute_alphas(micro_graphs)

        # NEW: Extract boundary descriptors for error-reduction corrections
        boundary_counts: Optional[List[int]] = None
        mu2_values: Optional[List[float]] = None
        if full_graph is not None:
            bc = full_graph.graph.get("boundary_counts")
            mu2 = full_graph.graph.get("mu2_values")
            if isinstance(bc, list):
                boundary_counts = [int(x) for x in bc]
            if isinstance(mu2, list):
                mu2_values = [float(x) for x in mu2]

        self.macro = MacroEngine(
            W=W,
            beta_macro=beta_macro,
            T=macro_T,
            community_sizes=community_sizes,
            alphas=alphas,
            alpha_mode=alpha_mode,
            boundary_counts=boundary_counts,
            mu2_values=mu2_values,
            correction_mode=correction_mode,
            beta_in=beta_micro,
            beta_out=beta_macro,
        )
        self.tau_micro = float(tau_micro)
        self.T_end = float(T_end)
        self.verbose_steps = bool(verbose_steps)
        self.dense_logs = bool(dense_logs)
        self.stop_at_absorption = bool(stop_at_absorption)
        self.run_metadata: Optional[dict] = None

        self.times: List[float] = []
        self.I_total: List[int] = []
        self.logs = {i: {"times": [], "S": [], "I": [], "R": []} for i in range(len(self.micro_models))}
        self.event_log: List[dict] = []

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
            eigvals = np.sort(np.linalg.eigvalsh(L_norm))
            if eigvals.size < 2:
                alphas.append(1.0)
                continue
            # Spectral gap = second-smallest eigenvalue (Fiedler value); large
            # gap (dense) -> alpha ~ 1, small gap (sparse) -> alpha > 1 damps
            # the over-estimated early macro export. Damping-only clamp.
            mu2 = float(eigvals[1])
            if mu2 <= 0.0:
                alphas.append(1.0)
                continue
            alphas.append(max(1.0, min(1.0 / mu2, 6.0)))
        return alphas

    def _counts_arrays(self) -> Tuple[List[int], List[int], List[int]]:
        S_arr, I_arr, R_arr = [], [], []
        for m in self.micro_models:
            S, I, R = m.count_states()
            S_arr.append(S)
            I_arr.append(I)
            R_arr.append(R)
        return S_arr, I_arr, R_arr

    def _append_comm_log_if_changed(self, comm: int, time: float, S: int, I: int, R: int) -> None:
        log = self.logs[comm]
        if log["times"] and log["S"][-1] == S and log["I"][-1] == I and log["R"][-1] == R:
            return
        log["times"].append(time)
        log["S"].append(S)
        log["I"].append(I)
        log["R"].append(R)

    def _natural_stop_reason(self, S_arr: List[int], I_arr: List[int]) -> Optional[str]:
        """Return a reason only for an absorbing/no-event global state."""
        if sum(I_arr) == 0:
            return "no_infected_nodes"
        if all(model.recovery_rate == 0.0 for model in self.micro_models) and sum(S_arr) == 0:
            return "all_nodes_infected"
        if (
            self.macro.total_hazard <= 0.0
            and all(
                model.total_infection_rate + model.total_recovery_rate < 1e-12
                for model in self.micro_models
            )
        ):
            return "no_enabled_events"
        return None

    def _set_run_metadata(
        self, stop_time: float, stop_reason: str, S_arr: List[int], I_arr: List[int], R_arr: List[int]
    ) -> None:
        self.run_metadata = {
            "stop_time": float(stop_time),
            "stop_reason": stop_reason,
            "final_state_by_community": [
                {"community": idx, "S": int(S), "I": int(I), "R": int(R)}
                for idx, (S, I, R) in enumerate(zip(S_arr, I_arr, R_arr))
            ],
        }

    def _densify_logs_from_global_times(self) -> None:
        if not self.times:
            return
        dense_times = list(self.times)
        for comm, sparse in self.logs.items():
            sparse_times = sparse["times"]
            sparse_S = sparse["S"]
            sparse_I = sparse["I"]
            sparse_R = sparse["R"]
            if not sparse_times:
                self.logs[comm] = {
                    "times": dense_times.copy(),
                    "S": [0] * len(dense_times),
                    "I": [0] * len(dense_times),
                    "R": [0] * len(dense_times),
                }
                continue

            idx = 0
            cur_S = sparse_S[0]
            cur_I = sparse_I[0]
            cur_R = sparse_R[0]
            n_sparse = len(sparse_times)

            out_S: List[int] = []
            out_I: List[int] = []
            out_R: List[int] = []
            for t in dense_times:
                while idx + 1 < n_sparse and sparse_times[idx + 1] <= t:
                    idx += 1
                    cur_S = sparse_S[idx]
                    cur_I = sparse_I[idx]
                    cur_R = sparse_R[idx]
                out_S.append(cur_S)
                out_I.append(cur_I)
                out_R.append(cur_R)

            self.logs[comm] = {
                "times": dense_times.copy(),
                "S": out_S,
                "I": out_I,
                "R": out_R,
            }

    # --- logging with correct hazard snapshot (batch) ---
    def _log_micro_events_batch(
        self,
        hazard_matrix: np.ndarray,
        pending: List[Tuple[int, float, str, int, Optional[int]]],  # (community, ev_time, etype, node, src)
        last_event_time: float,
    ) -> float:
        if not pending:
            return last_event_time
        total_hazard_snapshot = float(hazard_matrix.sum())
        for community, ev_time, etype, node, src in sorted(pending, key=lambda ev: ev[1]):
            self.event_log.append(
                {
                    "time": ev_time,
                    "wait_time": ev_time - last_event_time,
                    "event_type": etype,
                    "mode": "micro",
                    "community": community,
                    "node": node,
                    "src": src,
                    "total_hazard": total_hazard_snapshot,
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
            self._append_comm_log_if_changed(idx, 0.0, S, I, R)
        # <<< end initial snapshot >>>

        if self.stop_at_absorption:
            reason = self._natural_stop_reason(S0, I0)
            if reason is not None:
                self._set_run_metadata(0.0, reason, S0, I0, R0)
                if self.dense_logs:
                    self._densify_logs_from_global_times()
                return self.times, self.I_total, self.logs, self.event_log

        # Gillespie integral threshold for macro events
        thresh_int = -math.log(random.random())
        int_accum = 0.0
        clones = [m.clone() for m in self.micro_models]

        while t < self.T_end and self.run_metadata is None:
            t_grid = min(t + self.tau_micro, self.T_end)

            while t < t_grid:
                dt = t_grid - t
                t_mid = t + 0.5 * dt

                if self.verbose_steps:
                    print("t = {:.4f}, next dt = {:.4f}, t_mid = {:.4f}".format(t, dt, t_mid))

                # --- midpoint hazard via micro snapshots (no RNG leakage) ---
                rng_state = random.getstate()
                for clone, source in zip(clones, self.micro_models):
                    clone.refresh_from(source)
                    clone.simulate_until(t_mid, record_events=False)
                mid_states = [clone.count_states() for clone in clones]
                S_mid = [state[0] for state in mid_states]
                I_mid = [state[1] for state in mid_states]
                random.setstate(rng_state)

                total_hazard_mid = self.macro.total_hazard_given(I_mid, S_mid)

                if total_hazard_mid <= 0.0:
                    pending_micro: List[Tuple[int, float, str, int, Optional[int]]] = []
                    S_list: List[int] = []
                    I_list: List[int] = []
                    for idx, m in enumerate(self.micro_models):
                        S, I, R, events = m.simulate_until(t_grid)
                        S_list.append(S)
                        I_list.append(I)
                        for ev_time, ev_type, node, src in events:
                            pending_micro.append((idx, ev_time, ev_type, node, src))
                        self._append_comm_log_if_changed(idx, t_grid, S, I, R)

                    self.times.append(t_grid)
                    self.I_total.append(sum(I_list))
                    self.macro.update_hazards(I_list, S_list)
                    last_event_time = self._log_micro_events_batch(
                        hazard_matrix=self.macro.hazards,
                        pending=pending_micro,
                        last_event_time=last_event_time,
                    )
                    t = t_grid
                    if self.stop_at_absorption:
                        S_actual, I_actual, R_actual = self._counts_arrays()
                        reason = self._natural_stop_reason(S_actual, I_actual)
                        if reason is not None:
                            self._set_run_metadata(t, reason, S_actual, I_actual, R_actual)
                            break
                    break

                # midpoint rectangle accumulation: check if threshold crosses in [t, t_grid]
                if int_accum + total_hazard_mid * dt >= thresh_int:
                    # A MACRO EVENT occurs inside the current sub-interval at t_event
                    t_event = t + (thresh_int - int_accum) / total_hazard_mid

                    if self.verbose_steps:
                        print(">>> MACRO EVENT at t = {:.4f} <<<".format(t_event))

                    pending_micro: List[Tuple[int, float, str, int, Optional[int]]] = []
                    S_event: List[int] = []
                    I_event: List[int] = []
                    for idx, micro_model in enumerate(self.micro_models):
                        S, I, R, events = micro_model.simulate_until(t_event)
                        S_event.append(S)
                        I_event.append(I)
                        for ev_time, ev_type, node, src in events:
                            pending_micro.append((idx, ev_time, ev_type, node, src))

                    self.macro.update_hazards(I_event, S_event)
                    last_event_time = self._log_micro_events_batch(
                        hazard_matrix=self.macro.hazards,
                        pending=pending_micro,
                        last_event_time=last_event_time,
                    )

                    # sample and apply the macro transfer at t_event
                    hazard_before = self.macro.total_hazard
                    i, j = self.macro.sample_transfer()
                    if j != i:
                        node = self.micro_models[j].import_infection(at_time=t_event)
                        S_post = list(S_event)
                        I_post = list(I_event)
                        if node is not None:
                            S_post[j] -= 1
                            I_post[j] += 1
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
                                "successful_import": node is not None,
                                "destination_S_before": S_event[j],
                                "destination_I_before": I_event[j],
                                "repeat_import": node is not None and I_event[j] > 0,
                                "total_hazard": hazard_before,
                                "total_hazard_after": self.macro.total_hazard,
                            }
                        )
                        last_event_time = t_event

                    self.times.append(t_event)
                    self.I_total.append(sum(m.count_states()[1] for m in self.micro_models))
                    for idx, m in enumerate(self.micro_models):
                        S, I, R = m.count_states()
                        self._append_comm_log_if_changed(idx, t_event, S, I, R)

                    t = t_event
                    if self.stop_at_absorption:
                        S_actual, I_actual, R_actual = self._counts_arrays()
                        reason = self._natural_stop_reason(S_actual, I_actual)
                        if reason is not None:
                            self._set_run_metadata(t, reason, S_actual, I_actual, R_actual)
                            break
                    thresh_int = -math.log(random.random())
                    int_accum = 0.0
                    continue

                # NO MACRO EVENT before grid boundary: advance micros to t_grid
                pending_micro = []
                S_list = []
                I_list = []
                for idx, m in enumerate(self.micro_models):
                    S, I, R, events = m.simulate_until(t_grid)
                    S_list.append(S)
                    I_list.append(I)
                    for ev_time, ev_type, node, src in events:
                        pending_micro.append((idx, ev_time, ev_type, node, src))
                    self._append_comm_log_if_changed(idx, t_grid, S, I, R)

                self.times.append(t_grid)
                self.I_total.append(sum(I_list))
                int_accum += total_hazard_mid * dt
                self.macro.update_hazards(I_list, S_list)
                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time,
                )
                t = t_grid
                if self.stop_at_absorption:
                    S_actual, I_actual, R_actual = self._counts_arrays()
                    reason = self._natural_stop_reason(S_actual, I_actual)
                    if reason is not None:
                        self._set_run_metadata(t, reason, S_actual, I_actual, R_actual)
                break

        if self.stop_at_absorption and self.run_metadata is None:
            S_final, I_final, R_final = self._counts_arrays()
            self._set_run_metadata(t, "max_horizon", S_final, I_final, R_final)
        if self.dense_logs:
            self._densify_logs_from_global_times()
        return self.times, self.I_total, self.logs, self.event_log


MicroMacroOrchestrator = Orchestrator

__all__ = ["Orchestrator", "MicroMacroOrchestrator"]
