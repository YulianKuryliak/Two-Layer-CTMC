import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..engines.micro_engine import MicroEngine
from ..types import (
    CommunityTiming,
    InfectionEvent,
    MicroRunRow,
    MicroSimulationResult,
    TransmissionEvent,
)


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
        if not full_graph.is_directed() and not full_graph.is_multigraph():
            self.sim.bind_static_graph(full_graph)
        else:
            # Retain legacy normalization for non-simple input graphs.
            for n in full_graph.nodes():
                self.sim.add_node(n)
            for u, v, d in full_graph.edges(data=True):
                self.sim.add_edge(u, v, weight=d.get("weight", 1.0))

        self._node_to_comm: Dict[int, int] = {}
        for ci, nodes in enumerate(comm_nodes):
            for n in nodes:
                self._node_to_comm[n] = ci
        self._bridge_nodes: Dict[int, set[int]] = {ci: set() for ci in range(len(comm_nodes))}
        self._external_neighbor_comms: Dict[int, set[int]] = {}
        for u, v in self.full_graph.edges():
            cu = self._node_to_comm.get(int(u))
            cv = self._node_to_comm.get(int(v))
            if cu is None or cv is None or cu == cv:
                continue
            self._bridge_nodes[cu].add(int(u))
            self._bridge_nodes[cv].add(int(v))
            self._external_neighbor_comms.setdefault(int(u), set()).add(int(cv))
            self._external_neighbor_comms.setdefault(int(v), set()).add(int(cu))

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

    @staticmethod
    def _build_community_timings(
        first_infection: List[Optional[float]],
        first_bridge: List[Optional[float]],
        first_export: List[Optional[float]],
        first_import: List[Optional[float]],
        first_import_source: List[Optional[int]],
    ) -> List[CommunityTiming]:
        rows: List[CommunityTiming] = []
        for community in range(len(first_infection)):
            t0 = first_infection[community]
            t_bridge = first_bridge[community]
            t_export = first_export[community]
            t_import = first_import[community]
            import_source = first_import_source[community]
            rows.append(
                {
                    "community": community,
                    "first_infection_time": t0,
                    "first_bridge_infection_time": t_bridge,
                    "first_export_time": t_export,
                    "first_import_time": t_import,
                    "first_import_source_community": import_source,
                    "time_to_bridge": (t_bridge - t0) if (t0 is not None and t_bridge is not None) else None,
                    "time_to_export": (t_export - t0) if (t0 is not None and t_export is not None) else None,
                    "time_to_import": (t_import - t0) if (t0 is not None and t_import is not None) else None,
                }
            )
        return rows

    def run(
        self,
        T_end: float,
        dt_out: float,
        initial_node: Optional[int] = None,
        stop_at_absorption: bool = False,
    ) -> MicroSimulationResult:
        if not self.sim.infected_nodes:
            initial_node = self.seed_infection(initial_node)

        current_counts = self._counts_SIR_per_community(self.sim, self.comm_nodes)
        # Only one node changes per event.  Keep its last reported state so
        # no-op events also leave counts unchanged.  Membership lists preserve
        # the recount semantics even for overlapping/repeated community nodes.
        count_memberships: Dict[int, List[int]] = {}
        for ci, nodes in enumerate(self.comm_nodes):
            for node in nodes:
                count_memberships.setdefault(node, []).append(ci)
        reported_states = {
            node: (int(data["infected"]), int(data["recovered"]))
            for node, data in self.sim.G.nodes(data=True)
        }
        n_comm = len(current_counts)
        infection_events: List[InfectionEvent] = []
        transmission_events: List[TransmissionEvent] = []
        first_infection: List[Optional[float]] = [None] * n_comm
        first_bridge: List[Optional[float]] = [None] * n_comm
        first_export: List[Optional[float]] = [None] * n_comm
        first_import: List[Optional[float]] = [None] * n_comm
        first_import_source: List[Optional[int]] = [None] * n_comm
        if initial_node is not None and current_counts:
            seed_comm = self._node_to_comm.get(initial_node, 0)
            if 0 <= seed_comm < len(current_counts):
                infection_events.append((0.0, seed_comm, "intra", current_counts[seed_comm][1]))
                transmission_events.append(
                    {
                        "time": 0.0,
                        "kind": "seed",
                        "src_node": None,
                        "dst_node": int(initial_node),
                        "src_community": None,
                        "dst_community": int(seed_comm),
                    }
                )
                first_infection[seed_comm] = 0.0
                if int(initial_node) in self._bridge_nodes.get(seed_comm, set()):
                    first_bridge[seed_comm] = 0.0

        rows: List[MicroRunRow] = []
        t = 0.0
        t_next_out = 0.0
        EPS = 1e-9
        stop_metadata = None

        def final_metadata(reason: str):
            return {
                "stop_time": float(t),
                "stop_reason": reason,
                "final_state_by_community": [
                    {"community": ci, "S": S, "I": I, "R": R}
                    for ci, (S, I, R) in enumerate(current_counts)
                ],
            }

        def natural_stop_reason() -> Optional[str]:
            total_S = sum(S for S, _, _ in current_counts)
            total_I = sum(I for _, I, _ in current_counts)
            if total_I == 0:
                return "no_infected_nodes"
            # gamma=0 makes the all-infected state absorbing for either SIS
            # or SIR implementation. With recovery it is not absorbing.
            if self.sim.recovery_rate == 0.0 and total_S == 0:
                return "all_nodes_infected"
            if self.sim.total_infection_rate + self.sim.total_recovery_rate < 1e-12:
                return "no_enabled_events"
            return None

        if stop_at_absorption:
            reason = natural_stop_reason()
            if reason is not None:
                for ci, (S, I, R) in enumerate(current_counts):
                    rows.append((ci, 0.0, S, I, R))
                stop_metadata = final_metadata(reason)

        while stop_metadata is None and t < T_end:
            if stop_at_absorption:
                # The historical public path applies the sampled event before
                # checking the horizon. This opt-in path instead restores RNG
                # when the event lies beyond T_end, so no post-horizon state is
                # materialized merely to fill an output plateau.
                rng_before = self.sim._get_rng_state()
                dt, raw_event = self.sim._sample_next_event()
                if dt == float("inf") or raw_event is None:
                    event = None
                elif t + dt > T_end:
                    self.sim._set_rng_state(rng_before)
                    stop_metadata = final_metadata("max_horizon")
                    break
                else:
                    etype, node, src = self.sim._apply_event(raw_event)
                    if etype == "infection":
                        event = ("infection", src, node)
                    elif etype == "recovery":
                        event = ("recovery", node, None)
                    else:
                        event = None
            else:
                dt, event = self.sim.simulate_step(return_details=True)

            if dt == float("inf"):
                if stop_at_absorption:
                    stop_metadata = final_metadata(natural_stop_reason() or "no_enabled_events")
                    break
                # hold last state
                while t_next_out <= T_end + EPS:
                    if t_next_out >= t - EPS:
                        for ci, (S, I, R) in enumerate(current_counts):
                            rows.append((ci, float(t_next_out), S, I, R))
                    t_next_out += dt_out
                break

            t_event = t + dt
            pending_infection: Optional[Tuple[float, int, int, int, int, str]] = None

            if event is not None and event[0] == "infection":
                _, src, dst = event
                dst_comm = self._node_to_comm.get(int(dst), 0)
                src_comm = self._node_to_comm.get(int(src), dst_comm)
                inf_type = "intra" if src_comm == dst_comm else "inter"
                pending_infection = (t_event, int(src), int(dst), int(src_comm), int(dst_comm), inf_type)

            # sample & hold until event
            while t_next_out < t_event and t_next_out <= T_end + EPS:
                if t_next_out >= t - EPS:
                    for ci, (S, I, R) in enumerate(current_counts):
                        rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out

            # update time and counts
            t = t_event
            if event is not None and event[0] in {"infection", "recovery"}:
                changed_node = event[2] if event[0] == "infection" else event[1]
                data = self.sim.G.nodes[changed_node]
                new_state = (int(data["infected"]), int(data["recovered"]))
                old_I, old_R = reported_states[changed_node]
                delta_I, delta_R = new_state[0] - old_I, new_state[1] - old_R
                for ci in count_memberships.get(changed_node, ()):
                    S, I, R = current_counts[ci]
                    current_counts[ci] = (S - delta_I - delta_R, I + delta_I, R + delta_R)
                reported_states[changed_node] = new_state

            if pending_infection is not None and current_counts:
                t_inf, src, dst, src_comm, dst_comm, inf_type = pending_infection
                if 0 <= dst_comm < len(current_counts):
                    infection_events.append((t_inf, dst_comm, inf_type, current_counts[dst_comm][1]))
                    transmission_events.append(
                        {
                            "time": t_inf,
                            "kind": inf_type,
                            "src_node": src,
                            "dst_node": dst,
                            "src_community": src_comm,
                            "dst_community": dst_comm,
                        }
                    )
                    if first_infection[dst_comm] is None:
                        first_infection[dst_comm] = t_inf
                    if src_comm != dst_comm:
                        if 0 <= src_comm < len(current_counts) and first_export[src_comm] is None:
                            first_export[src_comm] = t_inf
                        if first_import[dst_comm] is None:
                            first_import[dst_comm] = t_inf
                            first_import_source[dst_comm] = src_comm

                    if first_bridge[dst_comm] is None and dst in self._bridge_nodes.get(dst_comm, set()):
                        entry_src = first_import_source[dst_comm]
                        ext_targets = self._external_neighbor_comms.get(dst, set())
                        if entry_src is None or any(c != entry_src for c in ext_targets):
                            first_bridge[dst_comm] = t_inf

            if stop_at_absorption:
                reason = natural_stop_reason()
                if reason is not None:
                    stop_metadata = final_metadata(reason)
                    break

        if not stop_at_absorption:
            # Historical API: materialize a held output plateau through T_end.
            while t_next_out <= T_end + EPS:
                for ci, (S, I, R) in enumerate(current_counts):
                    rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out
        elif stop_metadata is None:
            stop_metadata = final_metadata("max_horizon")

        community_timings = self._build_community_timings(
            first_infection=first_infection,
            first_bridge=first_bridge,
            first_export=first_export,
            first_import=first_import,
            first_import_source=first_import_source,
        )
        bridge_nodes = {ci: sorted(nodes) for ci, nodes in self._bridge_nodes.items()}

        result = {
            "rows": rows,
            "infection_events": infection_events,
            "initial_node": initial_node,
            "transmission_events": transmission_events,
            "community_timings": community_timings,
            "bridge_nodes": bridge_nodes,
        }
        if stop_at_absorption:
            result["run_metadata"] = stop_metadata
        return result


__all__ = ["MicroOrchestrator"]
