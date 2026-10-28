from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "chain_cliques_arrival_comparison"
PLOT_ROOT = REPO_ROOT / "plots" / "chain_random_connected_benchmark"
PAPER_ROOT = REPO_ROOT / "paper" / "Images" / "chain_random_connected"

SCENARIOS = [
    {
        "name": "baseline",
        "group": "baseline",
        "x": 5,
        "x_label": "k",
        "degree": 6,
        "path": "chain_L10_n100_k5_beta0p003_rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
    },
    {
        "name": "k=1",
        "group": "interface",
        "x": 1,
        "x_label": "k",
        "degree": 6,
        "path": "chain_L10_n100_k1_beta0p003_rc_interface_L10_n100_k1_d6_beta0p003_main100",
        "plot_dir": "plots/k_10_n_100_inter_1_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_interface_L10_n100_k1_d6_beta0p003_main100",
    },
    {
        "name": "k=5",
        "group": "interface",
        "x": 5,
        "x_label": "k",
        "degree": 6,
        "path": "chain_L10_n100_k5_beta0p003_rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
    },
    {
        "name": "k=10",
        "group": "interface",
        "x": 10,
        "x_label": "k",
        "degree": 6,
        "path": "chain_L10_n100_k10_beta0p003_rc_interface_L10_n100_k10_d6_beta0p003_main100",
        "plot_dir": "plots/k_10_n_100_inter_10_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_interface_L10_n100_k10_d6_beta0p003_main100",
    },
    {
        "name": "k=20",
        "group": "interface",
        "x": 20,
        "x_label": "k",
        "degree": 6,
        "path": "chain_L10_n100_k20_beta0p003_rc_interface_L10_n100_k20_d6_beta0p003_main100",
        "plot_dir": "plots/k_10_n_100_inter_20_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_interface_L10_n100_k20_d6_beta0p003_main100",
    },
    {
        "name": "d=4",
        "group": "density",
        "x": 4,
        "x_label": "mean internal degree",
        "degree": 4,
        "path": "chain_L10_n100_k5_beta0p003_rc_density_L10_n100_k5_d4_beta0p003_main100",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_random_connected_p0p040404",
        "tag": "rc_density_L10_n100_k5_d4_beta0p003_main100",
    },
    {
        "name": "d=6",
        "group": "density",
        "x": 6,
        "x_label": "mean internal degree",
        "degree": 6,
        "path": "chain_L10_n100_k5_beta0p003_rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_random_connected_p0p0606061",
        "tag": "rc_baseline_L10_n100_k5_d6_beta0p003_100reps",
    },
    {
        "name": "d=10",
        "group": "density",
        "x": 10,
        "x_label": "mean internal degree",
        "degree": 10,
        "path": "chain_L10_n100_k5_beta0p003_rc_density_L10_n100_k5_d10_beta0p003_main100",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_random_connected_p0p10101",
        "tag": "rc_density_L10_n100_k5_d10_beta0p003_main100",
    },
]


def _terminal_summary(scenario):
    root = DATA_ROOT / scenario["path"]
    comp = pd.read_csv(root / "model_comparison_metrics.csv")
    term = comp[(comp["variable"] == "T_last") & (comp["layer_index"] == 9)].iloc[0]
    arrivals = pd.read_csv(root / "per_replication_arrivals.csv")
    first = arrivals[arrivals["layer_index"] == 0]
    runtime = pd.read_csv(root / "runtime_summary.csv")
    meta = pd.read_json(root / "parameters.json", typ="series")

    def completion(model):
        g = first[first["model"] == model]
        return float(g["reached_last"].mean())

    return {
        "scenario": scenario["name"],
        "group": scenario["group"],
        "x": scenario["x"],
        "degree": scenario["degree"],
        "L": int(meta["L"]),
        "n": int(meta["n"]),
        "k": int(meta["k"]),
        "edge_prob": float(meta["edge_prob"]),
        "beta": float(meta["beta"]),
        "T_end": float(meta["T_end"]),
        "micro_completion": completion("Micro"),
        "micromacro_completion": completion("MicroMacro"),
        "micro_mean_tlast": float(term["micro_mean"]),
        "micromacro_mean_tlast": float(term["micromacro_mean"]),
        "w1_tlast": float(term["wasserstein_w1"]),
        "relative_w1_tlast": float(term["relative_w1"]),
        "ks_tlast": float(term["ks_distance"]),
        "mean_shift": float(term["mean_shift"]),
        "relative_mean_shift": float(term["relative_mean_shift"]),
        "median_shift": float(term["median_shift"]),
        "q90_shift": float(term["q90_shift"]),
        "runtime_ratio": float(runtime["speedup_micro_over_micromacro"].iloc[0]),
    }


def _plot_sweep(summary, group, out_name, title):
    df = summary[summary["group"] == group].sort_values("x")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    x = df["x"]
    xlabel = "interface width k" if group == "interface" else "mean internal degree"

    axes[0, 0].plot(x, df["relative_w1_tlast"], marker="o")
    axes[0, 0].set_ylabel(r"relative $W_1(T_{last})$")

    axes[0, 1].plot(x, df["mean_shift"], marker="o", label="mean")
    axes[0, 1].plot(x, df["median_shift"], marker="s", label="median")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("MicroMacro - Micro")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x, df["micro_completion"], marker="o", label="Micro")
    axes[1, 0].plot(x, df["micromacro_completion"], marker="s", label="MicroMacro")
    axes[1, 0].set_ylabel("completion probability")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(x, df["runtime_ratio"], marker="o")
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("runtime Micro / MicroMacro")

    for ax in axes.ravel():
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    out = PLOT_ROOT / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    shutil.copy2(out, PAPER_ROOT / out.name)


def _copy_baseline_images():
    baseline = SCENARIOS[0]
    src_dir = REPO_ROOT / baseline["plot_dir"]
    tag = baseline["tag"]
    mapping = {
        f"chain_cliques_total_infected_iqr_{tag}.png": "random_connected_total_infected_iqr.png",
        f"chain_cliques_per_community_run_071_{tag}.png": "random_connected_per_community_pair.png",
        f"chain_cliques_tlast_cdf_{tag}.png": "random_connected_tlast_cdf.png",
        f"chain_cliques_paired_arrival_error_{tag}.png": "random_connected_paired_arrival_error.png",
        f"chain_cliques_error_growth_{tag}.png": "random_connected_error_growth.png",
    }
    for src_name, dst_name in mapping.items():
        src = src_dir / src_name
        if src.exists():
            shutil.copy2(src, PAPER_ROOT / dst_name)


def main():
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([_terminal_summary(s) for s in SCENARIOS])
    summary.to_csv(PLOT_ROOT / "chain_random_connected_summary.csv", index=False)
    shutil.copy2(PLOT_ROOT / "chain_random_connected_summary.csv", PAPER_ROOT / "chain_random_connected_summary.csv")
    _copy_baseline_images()
    _plot_sweep(summary, "interface", "random_connected_interface_summary.png", "Interface-width sweep")
    _plot_sweep(summary, "density", "random_connected_density_summary.png", "Internal-density sweep")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
