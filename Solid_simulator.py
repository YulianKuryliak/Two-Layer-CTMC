import os
import random
from typing import List, Tuple, Optional, Dict
import json
from sim_db import log_run

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

        wait_time = random.expovariate(total_rate)

        # wait_time = 1 / total_rate 

        # choose event type
        if random.random() < (self.total_infection_rate / total_rate):
            target = random.uniform(0, self.total_infection_rate)
            cum = 0.0
            for node in self.infected_nodes:
                cum += self.G.nodes[node]['sum_of_weights_i']
                if cum > target:
                    self._infect_neighbor(node)
                    # print(node)
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


import pandas as pd
import networkx as nx
import random
import numpy as np
from typing import List, Tuple, Optional

def run_and_save_uniform_csv_per_community(
    full_graph: nx.Graph,
    comm_nodes: List[List[int]],
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
    Executes an event-driven Gillespie simulation and records the S/I/R state 
    at uniform time intervals (t=0, dt, 2dt, ...).

    This function correctly handles the continuous-time nature of the simulation
    by maintaining the previous state for all time points falling within the 
    waiting time between events (Sample-and-Hold).

    Args:
        full_graph (nx.Graph): The complete network topology.
        comm_nodes (List[List[int]]): List of lists, where each sublist contains node IDs for a specific community.
        beta (float): Infection rate per link.
        gamma (float): Recovery rate per node.
        initial_node (int): ID of the seed node (patient zero).
        T_end (float): Total simulation duration.
        dt_out (float): Time step for data recording (sampling resolution).
        out_csv_path (str): File path to save the resulting CSV.
        seed (Optional[int]): Random seed for reproducibility.
        model (int): 1 for SIS, 2 for SIR.

    Returns:
        pd.DataFrame: A DataFrame sorted by community and time with columns [community, time, S, I, R].
    """
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize Simulator
    sim = EfficientEpidemicGraph(infection_rate=beta, recovery_rate=gamma, model=model)
    for n in full_graph.nodes():
        sim.add_node(n)
    for u, v, d in full_graph.edges(data=True):
        sim.add_edge(u, v, weight=d.get("weight", 1.0))

    sim._infect_node(initial_node)

    rows: List[Tuple[int, float, int, int, int]] = []
    
    t = 0.0
    t_next_out = 0.0
    EPS = 1e-9

    # Cache initial state
    current_counts = _counts_SIR_per_community(sim, comm_nodes)

    while t < T_end:
        # Calculate wait time (dt) and execute the event (state changes immediately after this call)
        dt = sim.simulate_step()

        # Handle Deadlock: If no events are possible (everyone recovered or isolated)
        if dt == float("inf"):
            while t_next_out <= T_end + EPS:
                if t_next_out >= t - EPS:
                    for ci, (S, I, R) in enumerate(current_counts):
                        rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out
            break

        t_event = t + dt

        # Fill timeline with OLD state (state is constant between t and t_event)
        while t_next_out < t_event:
            if t_next_out > T_end + EPS:
                break
            
            # Only record if we haven't already recorded this time point
            if t_next_out >= t - EPS:
                for ci, (S, I, R) in enumerate(current_counts):
                    rows.append((ci, float(t_next_out), S, I, R))
            
            t_next_out += dt_out

        # Advance time and update state cache for the next iteration
        t = t_event
        current_counts = _counts_SIR_per_community(sim, comm_nodes)

    # Final cleanup: Ensure the end of the simulation is recorded
    while t_next_out <= T_end + EPS:
        for ci, (S, I, R) in enumerate(current_counts):
            rows.append((ci, float(t_next_out), S, I, R))
        t_next_out += dt_out

    # Format Output
    df = pd.DataFrame(rows, columns=["community", "time", "S", "I", "R"])
    df = df[df["time"].between(0.0 - EPS, T_end + EPS)].copy()
    
    df.sort_values(["community", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

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
def load_config(path: str = "config.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cfg = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_cfg = cfg["solid"]
    sim_common = cfg["simulation"]

    # --- network params (your network.py signature) ---
    COMMUNITIES = int(net_cfg["communities"])          # n_communities
    NODES_PER_COMM = int(net_cfg["community_size"])    # community_size
    MAX_INTER_LINKS = int(net_cfg["max_inter_links"])  # max_inter_links
    SEED = int(net_cfg["seed"])

    # --- epidemic params ---
    BETA = float(virus_cfg["beta"])
    GAMMA = float(virus_cfg["gamma"])
    MODEL = int(virus_cfg["model"])                # 1=SIS, 2=SIR

    # --- simulation params ---
    T_END = float(sim_common["T_end"])
    DT_OUT = float(sim_cfg["dt_out"])             # can be 1.0, 0.5, 0.1, etc.
    N_SIMS = int(sim_common["n_runs"])
    OUT_FOLDER = sim_cfg["out_folder"]
    BASE_SEED = int(sim_common["base_seed"])

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
        seeds=[BASE_SEED + i for i in range(N_SIMS)],
        model=MODEL,
    )

    print(f"Done. Wrote {N_SIMS} CSV files (per-community rows, times 0..{T_END} step {DT_OUT}) to: {OUT_FOLDER}")

    log_run(
        simulator="Solid",
        sim_version="1.0.1",
        network_params=net_cfg,
        virus_params=virus_cfg,
        sim_params={
            "T_end": T_END,
            "dt_out": DT_OUT,
            "n_runs": N_SIMS,
            "base_seed": BASE_SEED,
            "out_folder": OUT_FOLDER,
        },
        output_path=OUT_FOLDER,
    )
