import os
import random
from typing import List, Tuple, Optional, Dict
import json
from sim_db import log_run

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

    def simulate_step(self) -> Tuple[float, Optional[Tuple[str, int, Optional[int]]]]:
        """Perform one event and return (waiting time, event)."""
        if not self.infected_nodes:
            return float('inf'), None
        total_rate = self.total_infection_rate + self.total_recovery_rate
        if total_rate < 1e-8:
            return float('inf'), None

        wait_time = random.expovariate(total_rate)

        # wait_time = 1 / total_rate 

        # choose event type
        if random.random() < (self.total_infection_rate / total_rate):
            target = random.uniform(0, self.total_infection_rate)
            cum = 0.0
            src = None
            dst = None
            for node in self.infected_nodes:
                cum += self.G.nodes[node]['sum_of_weights_i']
                if cum > target:
                    src = node
                    dst = self._infect_neighbor(node)
                    # print(node)
                    break
            if src is None or dst is None:
                return wait_time, None
            return wait_time, ("infection", src, dst)
        else:
            target = random.uniform(0, self.total_recovery_rate)
            cum = 0.0
            chosen = None
            for node in list(self.infected_nodes):
                cum += self.recovery_rate
                if cum > target:
                    chosen = node
                    self._recover_node(node)
                    break
            if chosen is None:
                return wait_time, None
            return wait_time, ("recovery", chosen, None)
        return wait_time, None

    def _infect_neighbor(self, node: int) -> Optional[int]:
        neighbors = [n for n in self.G.neighbors(node)
                     if not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if not neighbors:
            return None
        weights = np.array([self.G[node][n]['weight'] for n in neighbors], dtype=float)
        target = random.uniform(0, float(weights.sum()))
        cum = 0.0
        for w, n in zip(weights, neighbors):
            cum += w
            if cum > target:
                self._infect_node(n)
                return n
        return None

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


def _plot_network_state(
    full_graph: nx.Graph,
    infected_nodes: List[int],
    pos: Dict[int, Tuple[float, float]],
    out_path: str,
    show: bool = False,
):
    infected_set = set(infected_nodes)
    node_colors = ["red" if n in infected_set else "green" for n in full_graph.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(full_graph, pos, edge_color="#999999", alpha=0.5, width=0.8)
    nx.draw_networkx_nodes(full_graph, pos, node_color=node_colors, node_size=80)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    


def _print_community_counts(
    t_event: float,
    counts: List[Tuple[int, int, int]],
):
    counts_str = ", ".join([f"c{idx}: S={s} I={i} R={r}" for idx, (s, i, r) in enumerate(counts)])
    print(f"[Micro] t={t_event:.6f} {counts_str}")


def _plot_all_communities_curves(
    df: pd.DataFrame,
    infection_events: List[Tuple[float, int, str, int]],
    out_path: str,
    show: bool = False,
):
    marker_map = {"inter": "x", "intra": "^"}
    cmap = plt.get_cmap("tab10")

    comm_series: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for comm_id, grp in df.groupby("community"):
        t = grp["time"].to_numpy()
        I = grp["I"].to_numpy()
        comm_series[int(comm_id)] = (t, I)

    plt.figure(figsize=(9, 5))
    for comm_id, (t, I) in comm_series.items():
        color = cmap(comm_id % 10)
        plt.plot(t, I, color=color, linewidth=1.8, label=f"Community {comm_id}")

    for t_event, comm_id, inf_type, I_count in infection_events:
        if comm_id not in comm_series:
            continue
        marker = marker_map.get(inf_type, "^")
        plt.scatter(t_event, I_count, marker=marker, color="red", s=50, alpha=0.8, zorder=5)

    marker_handles = [
        Line2D([], [], color="black", marker="x", linestyle="None", label="intercommunity infection"),
        Line2D([], [], color="black", marker="^", linestyle="None", label="intracommunity infection"),
    ]
    line_handles, line_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=line_handles + marker_handles,
        labels=line_labels + [h.get_label() for h in marker_handles],
        fontsize=8,
        ncol=2,
        loc="upper right",
    )

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title("Community I(t) with infection events")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


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
    plot_first_infection: bool = False,
    plot_folder: str = "plots",
    plot_layout_seed: Optional[int] = None,
    plot_all_communities: bool = False,
    all_communities_folder: str = "plots/micro_all_communities",
    plot_show: bool = False,
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

    node_to_comm: Dict[int, int] = {}
    for ci, nodes in enumerate(comm_nodes):
        for n in nodes:
            node_to_comm[n] = ci

    current_counts = _counts_SIR_per_community(sim, comm_nodes)
    infection_events: List[Tuple[float, int, str, int]] = []
    seed_comm = node_to_comm.get(initial_node, 0)
    infection_events.append((0.0, seed_comm, "intra", current_counts[seed_comm][1]))

    plot_pos = None
    first_infected_comms: set[int] = {seed_comm}
    if plot_first_infection:
        os.makedirs(plot_folder, exist_ok=True)
        plot_pos = nx.spring_layout(full_graph, seed=plot_layout_seed)
        _print_community_counts(0.0, current_counts)
        run_tag = os.path.splitext(os.path.basename(out_csv_path))[0]
        out_path = os.path.join(plot_folder, f"{run_tag}_comm{seed_comm}_t0.png")
        _plot_network_state(full_graph, sim.infected_nodes, plot_pos, out_path, show=plot_show)

    rows: List[Tuple[int, float, int, int, int]] = []
    
    t = 0.0
    t_next_out = 0.0
    EPS = 1e-9

    while t < T_end:
        # Calculate wait time (dt) and execute the event (state changes immediately after this call)
        dt, event = sim.simulate_step()

        # Handle Deadlock: If no events are possible (everyone recovered or isolated)
        if dt == float("inf"):
            while t_next_out <= T_end + EPS:
                if t_next_out >= t - EPS:
                    for ci, (S, I, R) in enumerate(current_counts):
                        rows.append((ci, float(t_next_out), S, I, R))
                t_next_out += dt_out
            break

        t_event = t + dt
        new_infected_comms: List[int] = []
        pending_infection: Optional[Tuple[float, int, str]] = None
        if event is not None and event[0] == "infection":
            _, src, dst = event
            dst_comm = node_to_comm.get(dst, 0)
            src_comm = node_to_comm.get(src, dst_comm)
            inf_type = "intra" if src_comm == dst_comm else "inter"
            pending_infection = (t_event, dst_comm, inf_type)
            if plot_first_infection and dst_comm not in first_infected_comms:
                new_infected_comms.append(dst_comm)

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
        if pending_infection is not None:
            t_inf, dst_comm, inf_type = pending_infection
            infection_events.append((t_inf, dst_comm, inf_type, current_counts[dst_comm][1]))
        if plot_first_infection and new_infected_comms:
            _print_community_counts(t_event, current_counts)
            run_tag = os.path.splitext(os.path.basename(out_csv_path))[0]
            t_str = f"{t_event:.6f}".replace(".", "p")
            for ci in new_infected_comms:
                first_infected_comms.add(ci)
                out_path = os.path.join(plot_folder, f"{run_tag}_comm{ci}_t{t_str}.png")
                _plot_network_state(full_graph, sim.infected_nodes, plot_pos, out_path, show=plot_show)

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
    if plot_all_communities:
        os.makedirs(all_communities_folder, exist_ok=True)
        run_tag = os.path.splitext(os.path.basename(out_csv_path))[0]
        out_path = os.path.join(all_communities_folder, f"{run_tag}_all_communities.png")
        _plot_all_communities_curves(
            df=df,
            infection_events=infection_events,
            out_path=out_path,
            show=plot_show,
        )
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
    plot_first_infection: bool = False,
    plot_folder: str = "plots",
    plot_all_communities: bool = False,
    all_communities_folder: str = "plots/micro_all_communities",
    plot_show: bool = False,
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
            plot_first_infection=plot_first_infection,
            plot_folder=plot_folder,
            plot_layout_seed=seed,
            plot_all_communities=plot_all_communities,
            all_communities_folder=all_communities_folder,
            plot_show=plot_show,
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
    sim_cfg = cfg["micro"]
    sim_common = cfg["simulation"]

    # --- network params (your network.py signature) ---
    COMMUNITIES = int(net_cfg["communities"])          # n_communities
    NODES_PER_COMM = int(net_cfg["community_size"])    # community_size
    INTER_LINKS = int(net_cfg["inter_links"])  # inter-community links per macro edge
    MACRO_GRAPH_TYPE = str(net_cfg["macro_graph_type"])
    MICRO_GRAPH_TYPE = str(net_cfg["micro_graph_type"])
    EDGE_PROB = float(net_cfg["edge_prob"])
    SEED = int(net_cfg["seed"])

    # --- epidemic params ---
    BETA = float(virus_cfg["beta"])
    GAMMA = float(virus_cfg["gamma"])
    MODEL = int(virus_cfg["model"])                # 1=SIS, 2=SIR

    # --- simulation params ---
    T_END = float(sim_common["T_end"])
    DT_OUT = float(sim_cfg["dt_out"])             # can be 1.0, 0.5, 0.1, etc.
    RUN_ONCE = True
    SHOW_PLOTS = True
    PLOT_FIRST_INFECTION = True
    PLOT_ALL_COMMUNITIES = True
    FIRST_INFECTION_FOLDER = "plots/micro_first_infection"
    ALL_COMMUNITIES_FOLDER = "plots/micro_all_communities"

    N_SIMS = 1 if RUN_ONCE else int(sim_common["n_runs"])
    OUT_FOLDER = sim_cfg["out_folder"]
    BASE_SEED = int(sim_common["base_seed"])

    # Build network using your generator
    micro_graphs, full_graph, _ = generate_two_scale_network(
        n_communities=COMMUNITIES,
        community_size=NODES_PER_COMM,
        inter_links=INTER_LINKS,
        seed=SEED,
        macro_graph_type=MACRO_GRAPH_TYPE,
        micro_graph_type=MICRO_GRAPH_TYPE,
        edge_prob=EDGE_PROB,
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
        seeds=[BASE_SEED] if RUN_ONCE else [BASE_SEED + i for i in range(N_SIMS)],
        model=MODEL,
        plot_first_infection=PLOT_FIRST_INFECTION,
        plot_folder=FIRST_INFECTION_FOLDER,
        plot_all_communities=PLOT_ALL_COMMUNITIES,
        all_communities_folder=ALL_COMMUNITIES_FOLDER,
        plot_show=SHOW_PLOTS,
    )

    print(f"Done. Wrote {N_SIMS} CSV files (per-community rows, times 0..{T_END} step {DT_OUT}) to: {OUT_FOLDER}")

    log_run(
        simulator="Micro",
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
