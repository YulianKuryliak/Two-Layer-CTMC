import os
import csv
import math
import random
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Any
import json
from pathlib import Path

from network import generate_two_scale_network
from sim_db import log_run


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_like: str) -> Path:
    normalized = str(path_like).replace("\\", "/")
    path = Path(normalized).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


def load_config(path: str = "config.json") -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Low-level SSA (micro layer)
# ---------------------------

class EfficientEpidemicGraph:
    """
    Base event-driven SIR/SIS simulator on a static network using the Gillespie algorithm.

    Tracks:
      - self.infected_nodes: list of infected node ids
      - per infected node i: nodes[i]['sum_of_weights_i'] = sum_{j susceptible} w_ij
      - self.total_infection_rate = beta * sum_i sum_of_weights_i over infected i
      - self.total_recovery_rate  = gamma * (# infected)
    """
    def __init__(self, infection_rate: float = 0.1, recovery_rate: float = 0.0, model: int = 2):
        self.G = nx.Graph()
        self.model = model                 # 1 = SIS, 2 = SIR
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.infected_nodes: List[int] = []
        self.total_infection_rate = 0.0
        self.total_recovery_rate = 0.0
        self.S_count = 0
        self.I_count = 0
        self.R_count = 0
        self._neighbors: dict[int, List[Tuple[int, float]]] = {}
        self._cached_wait: Optional[float] = None
        self._cached_event: Optional[Tuple[str, Any]] = None
        self._cached_rng_before = None
        self._cached_rng_after = None

    def add_node(self, node_id: int):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)
        self._neighbors[node_id] = []
        self.S_count += 1

    def add_edge(self, node1: int, node2: int, weight: float):
        if not (weight >= 0 and math.isfinite(weight)):
            raise ValueError("Edge weight must be non-negative and finite")
        self.G.add_edge(node1, node2, weight=weight)
        self._neighbors[node1].append((node2, weight))
        self._neighbors[node2].append((node1, weight))
        # topology change invalidates any cached event
        self._invalidate_cache()

    # -------------------------
    # Infection & Recovery core
    # -------------------------

    def _infect_node(self, node: int = None):
        """
        Turn 'node' from susceptible into infected and update hazard bookkeeping.
        """
        if node is None:
            node = random.choice(list(self.G.nodes))

        if self.G.nodes[node]['infected'] or self.G.nodes[node]['recovered']:
            return  # already infected or recovered: no-op

        self.G.nodes[node]['infected'] = True
        self.G.nodes[node]['recovered'] = False
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        self.S_count -= 1
        self.I_count += 1

        s = 0.0
        for nbr, w in self._neighbors[node]:
            nbr_data = self.G.nodes[nbr]
            if (not nbr_data['infected']) and (not nbr_data['recovered']):
                s += w
                self.total_infection_rate += w * self.infection_rate
            elif nbr_data['infected'] and nbr != node:
                # the infected neighbor loses this susceptible edge (node switched to I)
                new_sum = self.G.nodes[nbr]['sum_of_weights_i'] - w
                if -1e-12 < new_sum < 0:  # clamp tiny negatives
                    new_sum = 0.0
                self.G.nodes[nbr]['sum_of_weights_i'] = new_sum
                self.total_infection_rate -= w * self.infection_rate

        # set the new infected node's sum of susceptible neighbor weights
        if -1e-12 < s < 0:
            s = 0.0
        self.G.nodes[node]['sum_of_weights_i'] = s
        self._invalidate_cache()

    def _recover_node(self, node: int):
        """
        Recover 'node' from infected state.
        SIS: node becomes susceptible again; infected neighbors gain a susceptible edge.
        SIR: node becomes recovered; no neighbor gains susceptibility.
        """
        if not self.G.nodes[node]['infected']:
            return

        # Remove node's own infection hazard
        node_sum = self.G.nodes[node]['sum_of_weights_i']
        if node_sum != 0.0:
            self.total_infection_rate -= self.infection_rate * node_sum
        if self.total_infection_rate < 0.0:
            self.total_infection_rate = 0.0



        # Remove from infected set
        if node in self.infected_nodes:
            self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        self.I_count -= 1

        if self.model == 1:  # SIS
            # Each INFECTED neighbor gains a susceptible edge to 'node'
            for nbr, w in self._neighbors[node]:
                if self.G.nodes[nbr]['infected']:
                    new_sum = self.G.nodes[nbr]['sum_of_weights_i'] + w
                    if -1e-12 < new_sum < 0:
                        new_sum = 0.0
                    self.G.nodes[nbr]['sum_of_weights_i'] = new_sum
                    self.total_infection_rate += w * self.infection_rate

            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['recovered'] = False
            self.G.nodes[node]['sum_of_weights_i'] = 0.0
            self.S_count += 1

        else:  # SIR
            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['recovered'] = True
            self.G.nodes[node]['sum_of_weights_i'] = 0.0
            self.R_count += 1

        self._invalidate_cache()

    # ---------------------------------
    # Gillespie: sample vs. apply split
    # ---------------------------------

    def _sample_next_event(self) -> Tuple[float, Optional[Tuple[str, Any]]]:
        """
        SAMPLE the next event WITHOUT modifying graph state.

        Returns:
            (wait_time, event)
            where event is:
              ('infection', src_infected, dst_susceptible) OR ('recovery', node)
        """
        if not self.infected_nodes:
            return float('inf'), None

        total_rate = self.total_infection_rate + self.total_recovery_rate
        if total_rate < 1e-12:
            return float('inf'), None

        # wait_time = 1/total_rate

        wait_time = random.expovariate(total_rate)

        # Infection vs recovery
        if random.random() < (self.total_infection_rate / total_rate):
            # pick infected source with prob ~ beta * sum_of_weights_i
            target = random.uniform(0.0, self.total_infection_rate)
            cum = 0.0
            chosen_src = None
            for node in self.infected_nodes:
                cum += self.G.nodes[node]['sum_of_weights_i'] * self.infection_rate
                if cum > target:
                    chosen_src = node
                    break
            if chosen_src is None:
                return wait_time, None

            # pick susceptible neighbor of chosen_src with prob ~ w
            neighbors = [n for n, _ in self._neighbors[chosen_src]
                         if (not self.G.nodes[n]['infected']) and (not self.G.nodes[n]['recovered'])]
            if not neighbors:
                return wait_time, None

            weight_map = dict(self._neighbors[chosen_src])
            weights = np.array([weight_map[n] for n in neighbors], dtype=float)
            wsum = float(weights.sum())
            if wsum <= 0.0:
                return wait_time, None
            target2 = random.uniform(0.0, wsum)
            cum2 = 0.0
            chosen_dst = None
            for w, n in zip(weights, neighbors):
                cum2 += w
                if cum2 > target2:
                    chosen_dst = n
                    break
            if chosen_dst is None:
                return wait_time, None

            return wait_time, ('infection', chosen_src, chosen_dst)

        else:
            target = random.uniform(0.0, self.total_recovery_rate)
            cum = 0.0
            chosen = None
            for node in self.infected_nodes:
                cum += self.recovery_rate
                if cum > target:
                    chosen = node
                    break
            if chosen is None:
                return wait_time, None
            return wait_time, ('recovery', chosen)

    def _apply_event(self, event: Tuple[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
        """
        APPLY a previously sampled event to the graph.

        Returns:
            (etype, node_for_logging, src_if_infection_else_None)
        """
        etype = event[0]
        if etype == 'infection':
            _, src, dst = event
            self._infect_node(dst)
            return 'infection', dst, src
        elif etype == 'recovery':
            _, node = event
            self._recover_node(node)
            return 'recovery', node, None
        else:
            return 'none', None, None

    def simulate_step(self) -> Tuple[float, Optional[str], Optional[int]]:
        """
        Backwards-compatible single step.
        """
        wait_time, ev = self._sample_next_event()
        if wait_time == float('inf') or ev is None:
            return float('inf'), None, None
        etype, node, _ = self._apply_event(ev)
        return wait_time, etype, node


class MicroModel(EfficientEpidemicGraph):
    """
    Micro-scale community simulator with horizon-safe advancing.
    """
    def __init__(self, infection_rate: float, recovery_rate: float, model: int = 2):
        super().__init__(infection_rate, recovery_rate, model)
        self.current_time = 0.0

    def _invalidate_cache(self):
        self._cached_wait = None
        self._cached_event = None
        self._cached_rng_before = None
        self._cached_rng_after = None

    def _ensure_next_event(self):
        """
        Cache the next event sampling (wait_time, event, rng_before, rng_after).
        Preserves RNG sequence by storing pre/post RNG states.
        """
        if self._cached_wait is None:
            rng_before = random.getstate()
            wait_time, ev = self._sample_next_event()
            rng_after = random.getstate()
            self._cached_wait = wait_time
            self._cached_event = ev
            self._cached_rng_before = rng_before
            self._cached_rng_after = rng_after
        return self._cached_wait, self._cached_event, self._cached_rng_before, self._cached_rng_after

    def simulate_until(self, t_end: float) -> Tuple[int, int, int, List[Tuple[float, str, int, Optional[int]]]]:
        """
        Advance time up to t_end, applying ONLY events whose occurrence time <= t_end.
        Returns (S, I, R, events) where each event is (time, etype, node, src_if_infection_else_None).
        """
        events: List[Tuple[float, str, int, Optional[int]]] = []

        while self.current_time < t_end:
            wait_time, ev, rng_before, rng_after = self._ensure_next_event()

            # No events possible -> stop at horizon
            if wait_time == float('inf') or ev is None:
                self._invalidate_cache()
                break

            # Next event beyond horizon -> restore RNG and stop
            if self.current_time + wait_time > t_end:
                random.setstate(rng_before)
                break

            # Apply the event
            random.setstate(rng_after)
            self.current_time += wait_time
            etype, node, src = self._apply_event(ev)
            if etype != 'none' and node is not None:
                events.append((self.current_time, etype, node, src))
            self._invalidate_cache()

        # if self.current_time < t_end:
        #     self.current_time = t_end

        return self.S_count, self.I_count, self.R_count, events

    def count_states(self) -> Tuple[int, int, int]:
        return self.S_count, self.I_count, self.R_count

    def import_infection(self) -> Optional[int]:
        """
        Introduce one new infection at random among susceptibles.
        """
        susceptibles = [n for n, d in self.G.nodes(data=True)
                        if (not d['infected']) and (not d['recovered'])]
        if not susceptibles:
            return None
        node = random.choice(susceptibles)
        self._infect_node(node)
        self._invalidate_cache()
        return node

    # --- lightweight cloning for midpoint probing (no RNG leakage) ---
    def clone(self) -> "MicroModel":
        m = MicroModel(self.infection_rate, self.recovery_rate, self.model)
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


# ---------------------------
# Macro layer (susceptibility-aware, normalized)
# ---------------------------

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
        I = np.asarray(I_counts, dtype=float).reshape(-1, 1)   # (n,1)
        S = np.asarray(S_counts, dtype=float).reshape(1, -1)   # (1,n)

        N_i = self.N.reshape(-1, 1)  # (n,1)
        N_j = self.N.reshape(1, -1)  # (1,n)

        # frac = (I / N_i) @ (S / N_j)
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


# ---------------------------
# Orchestrator
# ---------------------------

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
                m.add_edge(u, v, weight=data.get('weight', 1.0))
            self.micro_models.append(m)

        # Community sizes for normalized macro hazards
        community_sizes = [G.number_of_nodes() for G in micro_graphs]

        self.macro = MacroEngine(
            W=W,
            beta_macro=beta_macro,
            T=macro_T,
            community_sizes=community_sizes
        )
        self.tau_micro = float(tau_micro)
        self.T_end = float(T_end)

        self.times: List[float] = []
        self.I_total: List[int] = []
        self.logs = {i: {'times': [], 'S': [], 'I': [], 'R': []} for i in range(len(self.micro_models))}
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
            S_nodes = [n for n, d in m.G.nodes(data=True) if (not d['infected']) and (not d['recovered'])]
            I_nodes = [n for n, d in m.G.nodes(data=True) if d['infected']]
            R_nodes = [n for n, d in m.G.nodes(data=True) if d['recovered']]
            state_per_community.append({'S': S_nodes, 'I': I_nodes, 'R': R_nodes})
        return state_per_community

    def _counts_arrays(self) -> Tuple[List[int], List[int], List[int]]:
        S_arr, I_arr, R_arr = [], [], []
        for m in self.micro_models:
            S, I, R = m.count_states()
            S_arr.append(S); I_arr.append(I); R_arr.append(R)
        return S_arr, I_arr, R_arr

    # --- logging with correct hazard snapshot (batch) ---
    def _log_micro_events_batch(
        self,
        hazard_matrix: np.ndarray,
        pending: List[Tuple[int, float, str, int, Optional[int]]],  # (community, ev_time, etype, node, src)
        last_event_time: float
    ) -> float:
        total_hazard_snapshot = float(hazard_matrix.sum())
        hazard_snapshot = hazard_matrix.copy()
        states_snapshot = self._gather_state_per_community()
        for community, ev_time, etype, node, src in pending:
            self.event_log.append({
                'time':         ev_time,
                'wait_time':    ev_time - last_event_time,
                'event_type':   etype,
                'mode':         'micro',
                'community':    community,
                'node':         node,
                'src':          src,
                'hazard_matrix': hazard_snapshot,
                'total_hazard':  total_hazard_snapshot,
                'states':       states_snapshot
            })
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
            self.logs[idx]['times'].append(0.0)
            self.logs[idx]['S'].append(S)
            self.logs[idx]['I'].append(I)
            self.logs[idx]['R'].append(R)
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
                    self.logs[idx]['times'].append(t_next)
                    self.logs[idx]['S'].append(S)
                    self.logs[idx]['I'].append(I)
                    self.logs[idx]['R'].append(R)

                # global series
                self.times.append(t_next)
                self.I_total.append(sum(I_list))

                # recompute hazards at t_next (snapshot)
                self.macro.update_hazards(I_list, S_list)

                # Now log buffered micro events with the t_next snapshot
                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time
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
                    last_event_time=last_event_time
                )

                # sample and apply the macro transfer at t_event
                i, j = self.macro.sample_transfer()
                if j != i:
                    self.micro_models[j].current_time = t_event
                    node = self.micro_models[j].import_infection()
                    # After import, update hazards again to reflect new state
                    S_post, I_post, _ = self._counts_arrays()
                    self.macro.update_hazards(I_post, S_post)

                    self.event_log.append({
                        'time':       t_event,
                        'wait_time':  t_event - last_event_time,
                        'event_type': 'transfer',
                        'mode':       'macro',
                        'community':  j,
                        'src':        i,
                        'node':       node,
                        'hazard_matrix': self.macro.hazards.copy(),
                        'total_hazard':  self.macro.total_hazard,
                        'states':     self._gather_state_per_community()
                    })
                    last_event_time = t_event

                # Log state at t_event
                self.times.append(t_event)
                self.I_total.append(sum(m.count_states()[1] for m in self.micro_models))
                for idx, m in enumerate(self.micro_models):
                    S, I, R = m.count_states()
                    self.logs[idx]['times'].append(t_event)
                    self.logs[idx]['S'].append(S)
                    self.logs[idx]['I'].append(I)
                    self.logs[idx]['R'].append(R)

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
                    self.logs[idx]['times'].append(t_next)
                    self.logs[idx]['S'].append(S)
                    self.logs[idx]['I'].append(I)
                    self.logs[idx]['R'].append(R)

                self.times.append(t_next)
                self.I_total.append(sum(I_list))

                # accumulate hazard integral with midpoint rectangle
                int_accum += total_hazard_mid * dt

                # recompute hazards at t_next and then log buffered micro events
                self.macro.update_hazards(I_list, S_list)

                last_event_time = self._log_micro_events_batch(
                    hazard_matrix=self.macro.hazards,
                    pending=pending_micro,
                    last_event_time=last_event_time
                )

                t = t_next

        return self.times, self.I_total, self.logs, self.event_log


# ---------------------------
# Experiments (DATA EXPORT)
# ---------------------------

def _export_discrete_grid_csv(
    logs_per_comm: dict,
    k: int,
    tau_micro: float,
    T_end: float,
    csv_path: str
):
    """
    Export **only** the discrete grid t = tau, 2*tau, ..., floor(T_end/tau)*tau.
    Between grid points nothing is saved.
    Uses carry-forward of the last state at or before each grid point.
    """
    csv_path = resolve_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the grid robustly against floating point drift
    n_steps = int(math.floor(T_end / tau_micro + 1e-12))
    grid = np.array([(i + 1) * tau_micro for i in range(n_steps)], dtype=float)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['community', 'time', 'S', 'I', 'R'])

        for community in range(k):
            times = np.asarray(logs_per_comm[community]['times'], dtype=float)
            Ss    = np.asarray(logs_per_comm[community]['S'],    dtype=int)
            Is    = np.asarray(logs_per_comm[community]['I'],    dtype=int)
            Rs    = np.asarray(logs_per_comm[community]['R'],    dtype=int)

            # ensure ascending
            order = np.argsort(times)
            times, Ss, Is, Rs = times[order], Ss[order], Is[order], Rs[order]

            # For each t_out on the grid, pick the last snapshot with time <= t_out
            idxs = np.searchsorted(times, grid, side='right') - 1
            # clip (because we recorded t=0, idxs should be >= 0)
            idxs = np.clip(idxs, 0, len(times) - 1)

            for t_out, idx in zip(grid, idxs):
                s, i, r = int(Ss[idx]), int(Is[idx]), int(Rs[idx])
                # write as exact float; optionally round for neatness
                writer.writerow([community, float(t_out), s, i, r])


def run_experiments(
    runs: int = 100,
    output_dir: str = "data/Simulations_MicroMacro_hazard_updated",
    k: int = 2,
    size: int = 50,
    inter_links: int = 1,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    base_seed: int = 42,
    beta: float = 0.01,
    gamma: float = 0.0,
    tau_micro: float = 1.0,
    T_end: float = 100.0,
    macro_T: float = 1.0,
    model: int = 2,
):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- generate ONE fixed two-scale network for all runs ----
    net_seed = base_seed  # можна окремо параметризувати, але логіка проста: одна мережа
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=k,
        community_size=size,
        inter_links=inter_links,
        seed=net_seed,
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
    )

    for run_id in range(1, runs + 1):
        seed = base_seed + run_id
        random.seed(seed)
        np.random.seed(seed)

        # 2) run two-scale simulator on the same network
        orch = Orchestrator(
            W=W,
            micro_graphs=micro_graphs,
            beta_micro=beta,
            gamma=gamma,
            beta_macro=beta,
            tau_micro=tau_micro,
            T_end=T_end,
            macro_T=macro_T,
            model=model,
            full_graph=full_graph,
            layout_seed=seed
        )
        # seed a single infection (uniform susceptible) in community 0 by default
        orch.micro_models[0]._infect_node()

        print(f"\n--- Run {run_id}/{runs} starts (tau={tau_micro}) ---")
        _, _, logs_2s, _ = orch.run()
        print(f"--- Run {run_id}/{runs} ends ---\n")

        # 3) export **ONLY** discrete grid t = τ, 2τ, ..., floor(T_end/τ)*τ
        csv_path = output_dir / f"{run_id}.csv"
        _export_discrete_grid_csv(
            logs_per_comm=logs_2s,
            k=k,
            tau_micro=tau_micro,
            T_end=T_end,
            csv_path=csv_path
        )
        print(f"Run {run_id}/{runs} → {csv_path}")


if __name__ == "__main__":
    cfg = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    sim_variant = cfg["micromacro_v2"]

    k = int(net_cfg["communities"])
    size = int(net_cfg["community_size"])
    inter_links = int(net_cfg["inter_links"])
    macro_graph_type = str(net_cfg["macro_graph_type"])
    micro_graph_type = str(net_cfg["micro_graph_type"])
    edge_prob = float(net_cfg["edge_prob"])
    base_seed = int(sim_common["base_seed"])
    beta = float(virus_cfg["beta"])
    gamma = float(virus_cfg["gamma"])
    tau_micro = float(sim_cfg["tau_micro"])
    T_end = float(sim_common["T_end"])
    macro_T = float(sim_cfg["macro_T"])
    model = int(virus_cfg["model"])
    runs = int(sim_common["n_runs"])
    output_dir = sim_variant["out_folder"]

    run_experiments(
        runs=runs,
        output_dir=output_dir,
        k=k,             # number of communities
        size=size,        # size of one community
        inter_links=inter_links,    # number of links between communities
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
        base_seed=base_seed,
        beta=beta,      #infection rate
        gamma=gamma,       #recovery rate
        tau_micro=tau_micro,   # <-- set your step here; e.g., 0.1
        T_end=T_end,     # time end
        macro_T=macro_T,       # time step
        model=model,         # 1 - SIS 2 - SIR
    )

    log_run(
        simulator="MicroMacro_v2",
        sim_version="1.0.3",
        network_params=net_cfg,
        virus_params=virus_cfg,
        sim_params={
            "n_runs": runs,
            "base_seed": base_seed,
            "T_end": T_end,
            "tau_micro": tau_micro,
            "macro_T": macro_T,
            "out_folder": output_dir,
            "k": k,
            "size": size,
            "inter_links": inter_links,
        },
        output_path=output_dir,
    )


