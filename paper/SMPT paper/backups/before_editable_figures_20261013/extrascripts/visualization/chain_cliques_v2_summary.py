import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "chain_cliques_arrival_comparison"
PLOT_ROOT = REPO_ROOT / "plots" / "chain_cliques_v2_benchmark"
PAPER_IMAGE_ROOT = REPO_ROOT / "paper" / "Images" / "chain_cliques"


SCENARIOS = {
    "interface": [
        {"x": 1, "folder": "chain_L10_n100_k1_beta0p005_v2_interface_L10_n100_k1_100reps"},
        {"x": 5, "folder": "chain_L10_n100_k5_beta0p005_v2_baseline_L10_n100_k5_100reps"},
        {"x": 20, "folder": "chain_L10_n100_k20_beta0p005_v2_interface_L10_n100_k20_100reps"},
    ],
    "length": [
        {"x": 5, "folder": "chain_L5_n100_k5_beta0p005_v2_length_L5_n100_k5_100reps"},
        {"x": 10, "folder": "chain_L10_n100_k5_beta0p005_v2_baseline_L10_n100_k5_100reps"},
        {"x": 15, "folder": "chain_L15_n100_k5_beta0p005_v2_length_L15_n100_k5_100reps"},
    ],
    "size": [
        {"x": 50, "folder": "chain_L10_n50_k5_beta0p005_v2_size_L10_n50_k5_100reps"},
        {"x": 100, "folder": "chain_L10_n100_k5_beta0p005_v2_baseline_L10_n100_k5_100reps"},
        {"x": 200, "folder": "chain_L10_n200_k5_beta0p005_v2_size_L10_n200_k5_100reps"},
    ],
}

BASELINE_FOLDER = "chain_L10_n100_k5_beta0p005_v2_baseline_L10_n100_k5_100reps"
BASELINE_PLOT_DIR = REPO_ROOT / "plots" / "k_10_n_100_inter_5_macro_chain_micro_complete_p0p031"
BASELINE_TAG = "v2_baseline_L10_n100_k5_100reps"


def _read_scenario(folder: str) -> dict:
    path = DATA_ROOT / folder
    comp = pd.read_csv(path / "model_comparison_metrics.csv")
    summary = pd.read_csv(path / "summary_metrics.csv")
    runtime = pd.read_csv(path / "runtime_summary.csv")
    params = json.loads((path / "parameters.json").read_text(encoding="utf-8"))

    t_last = comp[comp["variable"] == "T_last"].iloc[0]
    micro_last = summary[(summary["model"] == "Micro") & (summary["variable"] == "T_last")].iloc[0]
    mm_last = summary[(summary["model"] == "MicroMacro") & (summary["variable"] == "T_last")].iloc[0]
    rt_micro = runtime[runtime["model"] == "Micro"].iloc[0]
    rt_mm = runtime[runtime["model"] == "MicroMacro"].iloc[0]

    return {
        "folder": folder,
        "L": int(params["L"]),
        "n": int(params["n"]),
        "k": int(params["k"]),
        "T_end": float(params["T_end"]),
        "n_runs": int(params["n_runs"]),
        "n_micro_completed": int(t_last["n_micro"]),
        "n_micromacro_completed": int(t_last["n_micromacro"]),
        "p_micro_completed": float(micro_last["p_observed"]),
        "p_micromacro_completed": float(mm_last["p_observed"]),
        "micro_mean_tlast": float(t_last["micro_mean"]),
        "micromacro_mean_tlast": float(t_last["micromacro_mean"]),
        "mean_shift_tlast": float(t_last["mean_shift"]),
        "median_shift_tlast": float(t_last["median_shift"]),
        "q90_shift_tlast": float(t_last["q90_shift"]),
        "w1_tlast": float(t_last["wasserstein_w1"]),
        "ks_tlast": float(t_last["ks_distance"]),
        "micro_total_runtime_s": float(rt_micro["total_runtime_seconds"]),
        "micromacro_total_runtime_s": float(rt_mm["total_runtime_seconds"]),
        "speedup_micro_over_micromacro": float(rt_micro["speedup_micro_over_micromacro"]),
    }


def _plot_sweep(df: pd.DataFrame, x_col: str, x_label: str, title: str, out_name: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    axes[0].plot(df[x_col], df["w1_tlast"], marker="o", color="#2f6f9f")
    axes[0].set_title(r"$W_1(T_{\mathrm{last}})$")
    axes[0].set_ylabel("time")

    axes[1].plot(df[x_col], df["mean_shift_tlast"], marker="o", label="mean", color="#c95f36")
    axes[1].plot(df[x_col], df["median_shift_tlast"], marker="s", label="median", color="#3b7d4a")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_title("MicroMacro - Micro shift")
    axes[1].legend()

    axes[2].plot(df[x_col], df["p_micro_completed"], marker="o", label="Micro", color="#2f6f9f")
    axes[2].plot(df[x_col], df["p_micromacro_completed"], marker="s", label="MicroMacro", color="#c95f36")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title(r"Completion by $T_{\mathrm{end}}$")
    axes[2].legend()

    axes[3].plot(df[x_col], df["speedup_micro_over_micromacro"], marker="o", color="#7f5aa2")
    axes[3].axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    axes[3].set_title("Runtime ratio Micro / MicroMacro")

    for ax in axes:
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()

    out = PLOT_ROOT / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _copy_baseline_plots() -> list[Path]:
    mapping = {
        f"chain_cliques_tlast_cdf_{BASELINE_TAG}.png": "v2_baseline_tlast_cdf.png",
        f"chain_cliques_selected_delay_density_{BASELINE_TAG}.png": "v2_baseline_delay_density.png",
        f"chain_cliques_error_growth_{BASELINE_TAG}.png": "v2_baseline_error_growth.png",
    }
    out_paths = []
    for src_name, dst_name in mapping.items():
        src = BASELINE_PLOT_DIR / src_name
        dst = PLOT_ROOT / dst_name
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for sweep_name, scenarios in SCENARIOS.items():
        for scenario in scenarios:
            row = _read_scenario(scenario["folder"])
            row["sweep"] = sweep_name
            row["x"] = scenario["x"]
            rows.append(row)
    summary = pd.DataFrame(rows).drop_duplicates(subset=["sweep", "folder"])
    summary.to_csv(PLOT_ROOT / "v2_summary_metrics.csv", index=False)

    outputs = _copy_baseline_plots()
    outputs.append(_plot_sweep(summary[summary["sweep"] == "interface"], "x", "interface width k", "Interface-width sweep", "v2_interface_width_summary.png"))
    outputs.append(_plot_sweep(summary[summary["sweep"] == "length"], "x", "chain length L", "Chain-length sweep", "v2_chain_length_summary.png"))
    outputs.append(_plot_sweep(summary[summary["sweep"] == "size"], "x", "community size n", "Community-size sweep", "v2_community_size_summary.png"))

    for path in outputs:
        shutil.copy2(path, PAPER_IMAGE_ROOT / path.name)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Wrote:")
    for path in outputs:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
