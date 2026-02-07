import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from devtools.config import load_config, resolve_path


def iter_csv_files(folder):
    for name in sorted(Path(folder).iterdir()):
        if name.is_file() and name.suffix.lower() == ".csv":
            yield str(name)


def load_series_by_community(csv_path):
    series = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                community = int(float(row["community"]))
                time = float(row["time"])
                s_val = float(row["S"])
                i_val = float(row["I"])
            except (KeyError, ValueError):
                continue
            series.setdefault(community, []).append((time, s_val, i_val))
    for community in series:
        series[community].sort(key=lambda x: x[0])
    return series


def entry_timeseries(series):
    entry_time = None
    for time, _, i_val in series:
        if i_val > 0:
            entry_time = time
            break
    if entry_time is None:
        return None
    return entry_time


def time_to_full(series):
    entry_time = entry_timeseries(series)
    if entry_time is None:
        return None
    for time, s_val, _ in series:
        if time >= entry_time and s_val == 0:
            return time - entry_time
    return None


def build_entry_times(series_by_comm):
    entry_times = {}
    for community, series in series_by_comm.items():
        entry = entry_timeseries(series)
        if entry is not None:
            entry_times[community] = entry
    return entry_times


def time_to_next_community(entry_time, entry_times, community):
    next_times = [t for comm, t in entry_times.items() if comm != community and t >= entry_time]
    if not next_times:
        return None
    return min(next_times) - entry_time


def collect_distributions(folder):
    points = {}
    for csv_path in iter_csv_files(folder):
        series_by_comm = load_series_by_community(csv_path)
        entry_times = build_entry_times(series_by_comm)
        for community, series in series_by_comm.items():
            points.setdefault(community, {"full": [], "exit": []})
            full_val = time_to_full(series)
            entry_time = entry_times.get(community)
            exit_val = None
            if entry_time is not None:
                exit_val = time_to_next_community(entry_time, entry_times, community)
            if full_val is not None:
                points[community]["full"].append(full_val)
            if exit_val is not None:
                points[community]["exit"].append(exit_val)
    return points


def plot_distributions(micro_points, micromacro_points, output_path):
    communities = sorted(set(micro_points) | set(micromacro_points))
    if not communities:
        raise RuntimeError("No communities with valid entry/exit data.")

    ncols = 2
    nrows = len(communities)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.8 * nrows), squeeze=False)

    for row_idx, community in enumerate(communities):
        micro = micro_points.get(community, {"full": [], "exit": []})
        micromacro = micromacro_points.get(community, {"full": [], "exit": []})

        ax_full = axes[row_idx][0]
        ax_exit = axes[row_idx][1]

        if micro["full"]:
            ax_full.hist(
                micro["full"],
                bins=30,
                density=True,
                alpha=0.5,
                color="#2ca02c",
                label="Micro",
            )
        if micromacro["full"]:
            ax_full.hist(
                micromacro["full"],
                bins=30,
                density=True,
                alpha=0.5,
                color="#1f77b4",
                label="MicroMacro",
            )

        if micro["exit"]:
            ax_exit.hist(
                micro["exit"],
                bins=30,
                density=True,
                alpha=0.5,
                color="#2ca02c",
                label="Micro",
            )
        if micromacro["exit"]:
            ax_exit.hist(
                micromacro["exit"],
                bins=30,
                density=True,
                alpha=0.5,
                color="#1f77b4",
                label="MicroMacro",
            )

        ax_full.set_title(f"Community {community}: entry → full")
        ax_exit.set_title(f"Community {community}: entry → exit")
        ax_full.set_xlabel("Time since entry")
        ax_exit.set_xlabel("Time since entry")
        ax_full.set_ylabel("Density")
        ax_exit.set_ylabel("Density")
        ax_full.grid(True, alpha=0.2)
        ax_exit.grid(True, alpha=0.2)

    handles, labels = [], []
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)


def resolve_output_path(output_arg):
    if output_arg:
        return Path(output_arg)

    cfg, base_dir = load_config()
    net_cfg = cfg["network"]
    edge_prob_str = str(net_cfg["edge_prob"]).replace(".", "p")
    folder_name = (
        f"k_{net_cfg['communities']}_"
        f"n_{net_cfg['community_size']}_"
        f"inter_{net_cfg['inter_links']}_"
        f"macro_{net_cfg['macro_graph_type']}_"
        f"micro_{net_cfg['micro_graph_type']}_"
        f"p{edge_prob_str}"
    )
    output_dir = resolve_path(Path("plots") / folder_name, base_dir=base_dir)
    return output_dir / "expansion_analysis.png"


def main():
    parser = argparse.ArgumentParser(
        description="Density distributions of entry→full and entry→exit times per community."
    )
    parser.add_argument(
        "--micro-dir",
        default=str(Path("data") / "Simulations_Micro"),
        help="Folder with Micro simulation CSV files.",
    )
    parser.add_argument(
        "--micromacro-dir",
        default=str(Path("data") / "Simulations_MicroMacro"),
        help="Folder with MicroMacro simulation CSV files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. If omitted, uses config.json plots folder.",
    )
    args = parser.parse_args()

    micro_points = collect_distributions(args.micro_dir)
    micromacro_points = collect_distributions(args.micromacro_dir)
    output_path = resolve_output_path(args.output)
    plot_distributions(micro_points, micromacro_points, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
