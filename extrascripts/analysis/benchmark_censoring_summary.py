from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "chain_cliques_arrival_comparison"
OUT = ROOT / "plots" / "benchmark_censoring"
BOOTSTRAPS = 1000

SCENARIOS = [
    ("clique", "baseline", "chain_L10_n100_k5_beta0p001_calibrated_baseline_L10_n100_k5_beta0p001_100reps"),
    ("clique", "k=1", "chain_L10_n100_k1_beta0p001_calibrated_interface_L10_n100_k1_beta0p001_100reps"),
    ("clique", "k=10", "chain_L10_n100_k10_beta0p001_calibrated_interface_L10_n100_k10_beta0p001_100reps"),
    ("clique", "k=20", "chain_L10_n100_k20_beta0p001_calibrated_interface_L10_n100_k20_beta0p001_100reps"),
    ("clique", "L=5", "chain_L5_n100_k5_beta0p001_calibrated_length_L5_n100_k5_beta0p001_100reps"),
    ("clique", "L=15", "chain_L15_n100_k5_beta0p001_calibrated_length_L15_n100_k5_beta0p001_100reps"),
    ("clique", "n=50", "chain_L10_n50_k5_beta0p001_calibrated_size_L10_n50_k5_beta0p001_100reps"),
    ("clique", "n=200", "chain_L10_n200_k5_beta0p001_calibrated_size_L10_n200_k5_beta0p001_100reps"),
    ("sparse", "baseline", "chain_L10_n100_k5_beta0p003_rc_baseline_L10_n100_k5_d6_beta0p003_100reps"),
    ("sparse", "k=1", "chain_L10_n100_k1_beta0p003_rc_interface_L10_n100_k1_d6_beta0p003_main100"),
    ("sparse", "k=10", "chain_L10_n100_k10_beta0p003_rc_interface_L10_n100_k10_d6_beta0p003_main100"),
    ("sparse", "k=20", "chain_L10_n100_k20_beta0p003_rc_interface_L10_n100_k20_d6_beta0p003_main100"),
    ("sparse", "d=4", "chain_L10_n100_k5_beta0p003_rc_density_L10_n100_k5_d4_beta0p003_main100"),
    ("sparse", "d=10", "chain_L10_n100_k5_beta0p003_rc_density_L10_n100_k5_d10_beta0p003_main100"),
]


def _load(folder):
    root = DATA / folder
    wide = pd.read_csv(root / "arrival_times_wide.csv")
    params = pd.read_json(root / "parameters.json", typ="series")
    table = wide.pivot(index="replication_id", columns="model", values="T_last")
    return table[["Micro", "MicroMacro"]], float(params["T_end"])


def _metrics(table, horizon, rng):
    micro = table["Micro"].to_numpy(float)
    macro = table["MicroMacro"].to_numpy(float)
    restricted_micro = np.where(np.isfinite(micro), micro, horizon)
    restricted_macro = np.where(np.isfinite(macro), macro, horizon)

    def estimates(idx):
        m, mm = micro[idx], macro[idx]
        rmst_shift = np.mean(restricted_macro[idx] - restricted_micro[idx])
        m_obs, mm_obs = m[np.isfinite(m)], mm[np.isfinite(mm)]
        if not len(m_obs) or not len(mm_obs):
            return rmst_shift, np.nan, np.nan
        shift = np.mean(mm_obs) - np.mean(m_obs)
        rel_w1 = wasserstein_distance(m_obs, mm_obs) / np.mean(m_obs)
        return rmst_shift, shift, rel_w1

    point = estimates(np.arange(len(table)))
    boot = np.asarray([estimates(rng.integers(0, len(table), len(table))) for _ in range(BOOTSTRAPS)])
    ci = np.nanquantile(boot, [0.025, 0.975], axis=0)
    return {
        "n_runs": len(table),
        "T_end": horizon,
        "completion_micro": np.isfinite(micro).mean(),
        "completion_micromacro": np.isfinite(macro).mean(),
        "rmst_micro": restricted_micro.mean(),
        "rmst_micromacro": restricted_macro.mean(),
        "rmst_shift": point[0],
        "rmst_shift_ci_low": ci[0, 0],
        "rmst_shift_ci_high": ci[1, 0],
        "completed_mean_shift": point[1],
        "completed_mean_shift_ci_low": ci[0, 1],
        "completed_mean_shift_ci_high": ci[1, 1],
        "completed_relative_w1": point[2],
        "completed_relative_w1_ci_low": ci[0, 2],
        "completed_relative_w1_ci_high": ci[1, 2],
    }


def _survival_plot(family, folder, destination):
    table, horizon = _load(folder)
    grid = np.linspace(0.0, horizon, 401)
    rng = np.random.default_rng(20260710)
    colors = {"Micro": "#2f6f9f", "MicroMacro": "#c95f36"}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for model in ("Micro", "MicroMacro"):
        values = table[model].to_numpy(float)
        curves = []
        for _ in range(BOOTSTRAPS):
            sample = values[rng.integers(0, len(values), len(values))]
            curves.append([1.0 - np.isfinite(sample[sample <= t]).sum() / len(sample) for t in grid])
        curve = np.array([1.0 - np.isfinite(values[values <= t]).sum() / len(values) for t in grid])
        lo, hi = np.quantile(np.asarray(curves), [0.025, 0.975], axis=0)
        ax.step(grid, curve, where="post", color=colors[model], label=model)
        ax.fill_between(grid, lo, hi, step="post", color=colors[model], alpha=0.18)
    ax.set(xlabel="time", ylabel=r"$P(T_{last}>t)$", xlim=(0, horizon), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT / f"{family}_baseline_tlast_survival.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, destination / out.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260710)
    rows = []
    for family, scenario, folder in SCENARIOS:
        table, horizon = _load(folder)
        rows.append({"family": family, "scenario": scenario, "folder": folder, **_metrics(table, horizon, rng)})
    summary_path = OUT / "censoring_bootstrap_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    shutil.copy2(summary_path, DATA / "censoring_bootstrap_summary.csv")
    _survival_plot("clique", SCENARIOS[0][2], ROOT / "paper" / "Images" / "chain_cliques")
    _survival_plot("sparse", SCENARIOS[8][2], ROOT / "paper" / "Images" / "chain_random_connected")
    print(summary_path)


if __name__ == "__main__":
    main()
