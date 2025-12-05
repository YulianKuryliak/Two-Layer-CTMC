import os
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import networkx as nx

from network import generate_two_scale_network  # your network.py



# =========================
# Event-driven SIR/SIS core
# =========================
class EfficientEpidemicGraph:
    """
    Base event-driven SIR/SIS simulator on a static network using Gillespie algorithm.
    model: 1=SIS, 2=SIR
    """
    def __init__(self, infection_rate: float = 0.1, recovery_rate: float = 0.0, model: int = 2):
        self.G = nx.Graph()
        self.model = model
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.infected_nodes: List[int] = []
        self.total_infection_rate = 0.0
        self.total_recovery_rate = 0.0

    def add_node(self, node_id: int):
        self.G.add_node(node_id, infected=False, recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1: int, node2: int, weight: float):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self) -> float:
        """Perform one event and return waiting time."""
        if not self.infected_nodes:
            return float('inf')
        total_rate = self.total_infection_rate + self.total_recovery_rate
        if total_rate < 1e-8:
            return float('inf')

        # wait_time = random.expovariate(total_rate)

        wait_time = 1 / total_rate 

        # choose event type
        if random.random() < (self.total_infection_rate / total_rate):
            target = random.uniform(0, self.total_infection_rate)
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.G.nodes[node]['sum_of_weights_i']
                if cum > target:
                    self._infect_neighbor(node)
                    break
        else:
            target = random.uniform(0, self.total_recovery_rate)
            cum = 0.0
            for node in list(self.infected_nodes):
                cum += self.recovery_rate
                if cum > target:
                    self._recover_node(node)
                    break
        return wait_time

    def _infect_neighbor(self, node: int):
        neighbors = [n for n in self.G.neighbors(node)
                     if not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if not neighbors:
            return
        weights = np.array([self.G[node][n]['weight'] for n in neighbors], dtype=float)
        target = random.uniform(0, float(weights.sum()))
        cum = 0.0
        for w, n in zip(weights, neighbors):
            cum += w
            if cum > target:
                self._infect_node(n)
                break

    def _infect_node(self, node: int):
        if self.G.nodes[node]['infected'] or self.G.nodes[node]['recovered']:
            return
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        # update infection pressures
        for nbr in self.G.neighbors(node):
            w = self.G[node][nbr]['weight']
            if not self.G.nodes[nbr]['infected'] and not self.G.nodes[nbr]['recovered']:
                self.G.nodes[node]['sum_of_weights_i'] += w
                self.total_infection_rate += w * self.infection_rate
            elif self.G.nodes[nbr]['infected'] and nbr != node:
                self.G.nodes[nbr]['sum_of_weights_i'] -= w
                self.total_infection_rate -= w * self.infection_rate

    def _recover_node(self, node: int):
        if node not in self.infected_nodes:
            return
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.model == 1:  # SIS
            for nbr in self.G.neighbors(node):
                w = self.G[node][nbr]['weight']
                if not self.G.nodes[nbr]['infected'] and not self.G.nodes[nbr]['recovered']:
                    self.G.nodes[nbr]['sum_of_weights_i'] += w
                    self.total_infection_rate += w * self.infection_rate
            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['sum_of_weights_i'] = 0.0
        else:  # SIR
            self.G.nodes[node]['infected'] = False
            self.G.nodes[node]['recovered'] = True
            self.G.nodes[node]['sum_of_weights_i'] = 0.0


# =========================================
# Sampling & CSV saving (per-community rows)
# =========================================
def _counts_SIR_per_community(sim: EfficientEpidemicGraph, comm_nodes: List[List[int]]) -> List[Tuple[int,int,int]]:
    """
    Returns a list of (S, I, R) per community in the same order as comm_nodes.
    """
    infected_set = set(sim.infected_nodes)
    recovered_set = {n for n, a in sim.G.nodes(data=True) if a.get("recovered", False)}
    out = []
    for nodes in comm_nodes:
        I = sum(1 for n in nodes if n in infected_set)
        R = sum(1 for n in nodes if n in recovered_set)
        S = len(nodes) - I - R
        out.append((S, I, R))
    return out


def run_and_save_uniform_csv_per_community(
    full_graph: nx.Graph,
    comm_nodes: List[List[int]],          # node IDs per community
    beta: float,
    gamma: float,
    initial_node: int,
    T_end: float,
    dt_out: float,
    out_csv_path: str,
    seed: Optional[int] = None,
    model: int = 2,  # 1=SIS, 2=SIR
) -> pd.DataFrame:
    """
    Event-driven sim; record at uniform grid t = 0, dt_out, 2*dt_out, ..., T_end.
    Writes rows per community at each t:
      columns: community,time,S,I,R
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Build sim on the full graph
    sim = EfficientEpidemicGraph(infection_rate=beta, recovery_rate=gamma, model=model)
    for n in full_graph.nodes():
        sim.add_node(n)
    for u, v, d in full_graph.edges(data=True):
        sim.add_edge(u, v, weight=d.get("weight", 1.0))

    # Seed infection
    sim._infect_node(initial_node)

    rows: List[Tuple[int, float, int, int, int]] = []  # (community, time, S, I, R)

    # === snapshot at t = 0.0 ===
    per_comm_0 = _counts_SIR_per_community(sim, comm_nodes)
    for ci, (S0, I0, R0) in enumerate(per_comm_0):
        rows.append((ci, 0.0, S0, I0, R0))

    # main loop: step events, emit rows at uniform grid
    t = 0.0
    t_next_out = dt_out

    while t < T_end and t_next_out <= T_end:
        dt = sim.simulate_step()
        if dt == float("inf"):
            # no more events → repeat current state on remaining grid points
            while t_next_out <= T_end + 1e-12:
                per_comm = _counts_SIR_per_community(sim, comm_nodes)
                for ci, (S, I, R) in enumerate(per_comm):
                    rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out
            break

        t += dt

        while t_next_out <= T_end and t_next_out <= t + 1e-12:
            per_comm = _counts_SIR_per_community(sim, comm_nodes)
            for ci, (S, I, R) in enumerate(per_comm):
                rows.append((ci, float(t_next_out), S, I, R))
            t_next_out += dt_out

    # safety: fill any remaining grid points
    while t_next_out <= T_end + 1e-12:
        per_comm = _counts_SIR_per_community(sim, comm_nodes)
        for ci, (S, I, R) in enumerate(per_comm):
            rows.append((ci, float(t_next_out), S, I, R))
        t_next_out += dt_out

    # Build DF and enforce ordering: community then time
    df = pd.DataFrame(rows, columns=["community", "time", "S", "I", "R"])
    df = df[df["time"].between(0.0 - 1e-9, T_end + 1e-9)].copy()  # include 0.0
    df.sort_values(["community", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Expect rows = (#communities) * (T_end/dt_out + 1)  (the +1 is for t=0)
    n_times = int(round(T_end / dt_out)) + 1
    expected = len(comm_nodes) * n_times
    assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"

    df.to_csv(out_csv_path, index=False)
    return df


def run_many_uniform_and_save_per_community(
    full_graph: nx.Graph,
    comm_nodes: List[List[int]],
    beta: float,
    gamma: float,
    T_end: float,
    dt_out: float,
    n_sims: int,
    out_folder: str,
    seeds: Optional[List[int]] = None,
    model: int = 2,
):
    os.makedirs(out_folder, exist_ok=True)
    node_list = list(full_graph.nodes)

    for i in range(n_sims):
        seed = None if seeds is None else seeds[i]
        initial_node = random.sample(node_list, 1)[0]  # uniform random node
        out_csv = os.path.join(out_folder, f"{i+1}.csv")

        run_and_save_uniform_csv_per_community(
            full_graph=full_graph,
            comm_nodes=comm_nodes,
            beta=beta,
            gamma=gamma,
            initial_node=initial_node,
            T_end=T_end,
            dt_out=dt_out,
            out_csv_path=out_csv,
            seed=seed,
            model=model,
        )


# ==================
# Run with YOUR params
# ==================
if __name__ == "__main__":
    # --- network params (your network.py signature) ---
    COMMUNITIES = 5          # n_communities
    NODES_PER_COMM = 100       # community_size
    MAX_INTER_LINKS = 5      # max_inter_links
    SEED = 42

    # --- epidemic params ---
    BETA = 0.004
    GAMMA = 0.0
    MODEL = 2                # 1=SIS, 2=SIR

    # --- simulation params ---
    T_END = 100
    DT_OUT = 1.0             # can be 1.0, 0.5, 0.1, etc.
    N_SIMS = 100
    OUT_FOLDER = r"data\Simulations_Solid"

    # Build network using your generator
    micro_graphs, full_graph, _ = generate_two_scale_network(
        n_communities=COMMUNITIES,
        community_size=NODES_PER_COMM,
        max_inter_links=MAX_INTER_LINKS,
        seed=SEED,
    )
    # Extract node lists per community from your micro_graphs
    comm_nodes = [list(Gc.nodes()) for Gc in micro_graphs]

    # Run N_SIMS → each CSV has (#communities * (T_END/DT_OUT + 1)) rows (t = 0, DT_OUT, …, T_END)
    run_many_uniform_and_save_per_community(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        beta=BETA,
        gamma=GAMMA,
        T_end=T_END,
        dt_out=DT_OUT,
        n_sims=N_SIMS,
        out_folder=OUT_FOLDER,
        seeds=[1000 + i for i in range(N_SIMS)],
        model=MODEL,
    )

    print(f"Done. Wrote {N_SIMS} CSV files (per-community rows, times 0..{T_END} step {DT_OUT}) to: {OUT_FOLDER}")
