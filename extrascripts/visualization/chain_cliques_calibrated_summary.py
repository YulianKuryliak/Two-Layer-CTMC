import shutil
from pathlib import Path

import numpy as np

from chain_cliques_arrival_comparison import _require_matplotlib, _require_pandas


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "chain_cliques_arrival_comparison"
PLOT_ROOT = REPO_ROOT / "plots" / "chain_cliques_calibrated_benchmark"
PAPER_IMG_ROOT = REPO_ROOT / "paper" / "Images" / "chain_cliques"


SCENARIOS = [
    {
        "group": "baseline",
        "label": "L10 n100 k5",
        "L": 10,
        "n": 100,
        "k": 5,
        "folder": "chain_L10_n100_k5_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_baseline_L10_n100_k5_beta0p001_100reps",
    },
    {
        "group": "interface",
        "label": "k=1",
        "L": 10,
        "n": 100,
        "k": 1,
        "folder": "chain_L10_n100_k1_beta0p001_calibrated_interface_L10_n100_k1_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_1_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_interface_L10_n100_k1_beta0p001_100reps",
    },
    {
        "group": "interface",
        "label": "k=5",
        "L": 10,
        "n": 100,
        "k": 5,
        "folder": "chain_L10_n100_k5_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_baseline_L10_n100_k5_beta0p001_100reps",
    },
    {
        "group": "interface",
        "label": "k=10",
        "L": 10,
        "n": 100,
        "k": 10,
        "folder": "chain_L10_n100_k10_beta0p001_calibrated_interface_L10_n100_k10_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_10_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_interface_L10_n100_k10_beta0p001_100reps",
    },
    {
        "group": "interface",
        "label": "k=20",
        "L": 10,
        "n": 100,
        "k": 20,
        "folder": "chain_L10_n100_k20_beta0p001_calibrated_interface_L10_n100_k20_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_20_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_interface_L10_n100_k20_beta0p001_100reps",
    },
    {
        "group": "length",
        "label": "L=5",
        "L": 5,
        "n": 100,
        "k": 5,
        "folder": "chain_L5_n100_k5_beta0p001_calibrated_length_L5_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_5_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_length_L5_n100_k5_beta0p001_100reps",
    },
    {
        "group": "length",
        "label": "L=10",
        "L": 10,
        "n": 100,
        "k": 5,
        "folder": "chain_L10_n100_k5_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_baseline_L10_n100_k5_beta0p001_100reps",
    },
    {
        "group": "length",
        "label": "L=15",
        "L": 15,
        "n": 100,
        "k": 5,
        "folder": "chain_L15_n100_k5_beta0p001_calibrated_length_L15_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_15_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_length_L15_n100_k5_beta0p001_100reps",
    },
    {
        "group": "size",
        "label": "n=50",
        "L": 10,
        "n": 50,
        "k": 5,
        "folder": "chain_L10_n50_k5_beta0p001_calibrated_size_L10_n50_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_50_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_size_L10_n50_k5_beta0p001_100reps",
    },
    {
        "group": "size",
        "label": "n=100",
        "L": 10,
        "n": 100,
        "k": 5,
        "folder": "chain_L10_n100_k5_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_100_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_baseline_L10_n100_k5_beta0p001_100reps",
    },
    {
        "group": "size",
        "label": "n=200",
        "L": 10,
        "n": 200,
        "k": 5,
        "folder": "chain_L10_n200_k5_beta0p001_calibrated_size_L10_n200_k5_beta0p001_100reps",
        "plot_dir": "plots/k_10_n_200_inter_5_macro_chain_micro_complete_p0p031",
        "tag": "calibrated_size_L10_n200_k5_beta0p001_100reps",
    },
]


def _scenario_row(info):
    pd = _require_pandas()
    path = DATA_ROOT / info["folder"]
    wide = pd.read_csv(path / "arrival_times_wide.csv")
    comp = pd.read_csv(path / "model_comparison_metrics.csv")
    runtime = pd.read_csv(path / "runtime_summary.csv")
    tlast = comp[comp["variable"].eq("T_last")].iloc[0]

    row = {key: info[key] for key in ("group", "label", "L", "n", "k", "folder", "plot_dir", "tag")}
    for model in ("Micro", "MicroMacro"):
        vals = wide.loc[wide["model"].eq(model), "T_last"].dropna()
        row[f"completion_{model}"] = float(len(vals) / 100.0)
        row[f"median_Tlast_{model}"] = float(vals.median()) if len(vals) else np.nan
        row[f"mean_Tlast_{model}"] = float(vals.mean()) if len(vals) else np.nan
    row["w1_Tlast"] = float(tlast["wasserstein_w1"])
    row["relative_w1_Tlast"] = float(tlast["relative_w1"])
    row["ks_Tlast"] = float(tlast["ks_distance"])
    row["mean_shift_Tlast"] = float(tlast["mean_shift"])
    row["relative_mean_shift_Tlast"] = float(tlast["relative_mean_shift"])
    row["median_shift_Tlast"] = float(tlast["median_shift"])
    runtimes = {item["model"]: float(item["total_runtime_seconds"]) for _, item in runtime.iterrows()}
    row["runtime_micro"] = runtimes.get("Micro", np.nan)
    row["runtime_micromacro"] = runtimes.get("MicroMacro", np.nan)
    row["runtime_ratio_micro_over_micromacro"] = row["runtime_micro"] / row["runtime_micromacro"]
    return row


def _plot_sweep(df, group, x_col, out_name):
    plt = _require_matplotlib()
    sdf = df[df["group"].eq(group)].drop_duplicates(subset=[x_col]).sort_values(x_col)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    axes = axes.ravel()
    axes[0].plot(sdf[x_col], sdf["w1_Tlast"], marker="o", color="#2f6f9f")
    axes[0].set_ylabel(r"$W_1(T_{last})$")
    axes[1].plot(sdf[x_col], sdf["relative_w1_Tlast"], marker="o", color="#2f6f9f")
    axes[1].set_ylabel(r"relative $W_1$")
    axes[2].plot(sdf[x_col], sdf["completion_Micro"], marker="o", label="Micro", color="#2f6f9f")
    axes[2].plot(sdf[x_col], sdf["completion_MicroMacro"], marker="s", label="MicroMacro", color="#c95f36")
    axes[2].set_ylabel("completion")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(fontsize=8)
    axes[3].plot(sdf[x_col], sdf["runtime_ratio_micro_over_micromacro"], marker="o", color="#3b7d4a")
    axes[3].axhline(1.0, color="black", alpha=0.45, linewidth=1.0)
    axes[3].set_ylabel("runtime Micro / MicroMacro")
    for ax in axes:
        ax.set_xlabel(x_col)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = PLOT_ROOT / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _load_total_stack(trajectory_root, t_grid):
    pd = _require_pandas()
    stacks = {}
    for model in ("Micro", "MicroMacro"):
        rows = []
        for path in sorted(Path(trajectory_root).glob(f"run_*/{model}.csv")):
            df = pd.read_csv(path)
            curve = df.groupby("time", as_index=False)["I"].sum().sort_values("time")
            t = curve["time"].to_numpy(dtype=float)
            y = curve["I"].to_numpy(dtype=float)
            rows.append(np.interp(t_grid, t, y, left=y[0], right=y[-1]))
        stacks[model] = np.vstack(rows) if rows else np.empty((0, t_grid.size))
    return stacks


def _plot_interface_dynamics():
    pd = _require_pandas()
    plt = _require_matplotlib()
    items = [
        (
            1,
            "data/Simulations_ChainCliques_v2/k_10_n_100_inter_1_macro_chain_micro_complete_beta0p001_calibrated_interface_L10_n100_k1_T5000_beta0p001_100reps",
            "plots/k_10_n_100_inter_1_macro_chain_micro_complete_p0p031/chain_cliques_paired_arrival_error_calibrated_interface_L10_n100_k1_T5000_beta0p001_100reps.csv",
            5000.0,
        ),
        (
            5,
            "data/Simulations_ChainCliques_v2/k_10_n_100_inter_5_macro_chain_micro_complete_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps",
            "plots/k_10_n_100_inter_5_macro_chain_micro_complete_p0p031/chain_cliques_paired_arrival_error_calibrated_baseline_L10_n100_k5_beta0p001_100reps.csv",
            2500.0,
        ),
        (
            10,
            "data/Simulations_ChainCliques_v2/k_10_n_100_inter_10_macro_chain_micro_complete_beta0p001_calibrated_interface_L10_n100_k10_beta0p001_100reps",
            "plots/k_10_n_100_inter_10_macro_chain_micro_complete_p0p031/chain_cliques_paired_arrival_error_calibrated_interface_L10_n100_k10_beta0p001_100reps.csv",
            2500.0,
        ),
        (
            20,
            "data/Simulations_ChainCliques_v2/k_10_n_100_inter_20_macro_chain_micro_complete_beta0p001_calibrated_interface_L10_n100_k20_beta0p001_100reps",
            "plots/k_10_n_100_inter_20_macro_chain_micro_complete_p0p031/chain_cliques_paired_arrival_error_calibrated_interface_L10_n100_k20_beta0p001_100reps.csv",
            2500.0,
        ),
    ]
    colors = {"Micro": "#2f6f9f", "MicroMacro": "#c95f36"}
    fig, axes = plt.subplots(len(items), 2, figsize=(11.0, 9.2))
    for row, (k, trajectory_root, paired_csv, x_max) in enumerate(items):
        ax_dyn, ax_err = axes[row]
        t_grid = np.arange(0.0, x_max + 1.0, 1.0)
        stacks = _load_total_stack(REPO_ROOT / trajectory_root, t_grid)
        for model in ("Micro", "MicroMacro"):
            stack = stacks[model]
            if stack.size == 0:
                continue
            q25 = np.quantile(stack, 0.25, axis=0)
            med = np.quantile(stack, 0.50, axis=0)
            q75 = np.quantile(stack, 0.75, axis=0)
            ax_dyn.fill_between(t_grid, q25, q75, color=colors[model], alpha=0.18, linewidth=0)
            ax_dyn.plot(t_grid, med, color=colors[model], linewidth=1.8, label=model)
        ax_dyn.set_title(f"k={k}: total infected")
        ax_dyn.set_ylabel(r"$I_{tot}(t)$")
        ax_dyn.set_xlim(0.0, x_max)
        ax_dyn.grid(True, alpha=0.25)
        if row == 0:
            ax_dyn.legend(fontsize=8)

        p = REPO_ROOT / paired_csv
        if p.exists():
            err = pd.read_csv(p)
            if not err.empty:
                x = err["layer_index"].to_numpy(dtype=float)
                ax_err.fill_between(x, err["q25"], err["q75"], color="#c95f36", alpha=0.22, linewidth=0)
                ax_err.plot(x, err["median"], marker="o", color="#c95f36", linewidth=1.8)
        ax_err.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax_err.set_title(f"k={k}: paired arrival error")
        ax_err.set_ylabel(r"$T_i^{MM}-T_i^{Micro}$")
        ax_err.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("time" if ax is axes[-1][0] else "layer index")
    fig.tight_layout()
    out = PLOT_ROOT / "calibrated_interface_dynamics_by_k.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _copy_for_paper(src, name):
    PAPER_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    dst = PAPER_IMG_ROOT / name
    shutil.copy2(src, dst)
    return dst


def main():
    pd = _require_pandas()
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [_scenario_row(info) for info in SCENARIOS]
    df = pd.DataFrame(rows)
    summary_path = PLOT_ROOT / "calibrated_summary_metrics.csv"
    df.to_csv(summary_path, index=False)

    plots = [
        _plot_sweep(df, "interface", "k", "calibrated_interface_summary.png"),
        _plot_sweep(df, "length", "L", "calibrated_chain_length_summary.png"),
        _plot_sweep(df, "size", "n", "calibrated_community_size_summary.png"),
        _plot_interface_dynamics(),
    ]

    baseline = next(item for item in SCENARIOS if item["group"] == "baseline")
    bdir = REPO_ROOT / baseline["plot_dir"]
    tag = baseline["tag"]
    paper_sources = [
        (bdir / f"chain_cliques_total_infected_iqr_{tag}.png", "calibrated_total_infected_iqr.png"),
        (bdir / f"chain_cliques_per_community_run_005_{tag}_completed_pair.png", "calibrated_per_community_pair.png"),
        (bdir / f"chain_cliques_tlast_cdf_{tag}.png", "calibrated_tlast_cdf.png"),
        (bdir / f"chain_cliques_paired_arrival_error_{tag}.png", "calibrated_paired_arrival_error.png"),
        (bdir / f"chain_cliques_error_growth_{tag}.png", "calibrated_error_growth.png"),
        (plots[0], "calibrated_interface_summary.png"),
        (plots[1], "calibrated_chain_length_summary.png"),
        (plots[2], "calibrated_community_size_summary.png"),
        (plots[3], "calibrated_interface_dynamics_by_k.png"),
    ]
    copied = []
    for src, name in paper_sources:
        if src.exists():
            copied.append(_copy_for_paper(src, name))
    print(summary_path)
    for path in plots:
        print(path)
    for path in copied:
        print(path)


if __name__ == "__main__":
    main()
