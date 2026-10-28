import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MICRO = "FULL_MICRO_REFERENCE"
EXACT = "ORACLE_EXACT_HAZARD_EXACT_PLACEMENT"
PLACEMENT = "ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT"
BOUNDARY = "AGGREGATE_HAZARD_BOUNDARY_PLACEMENT"
CURRENT = "AGGREGATE_HAZARD_UNIFORM_PLACEMENT"
ORDER = [MICRO, EXACT, PLACEMENT, BOUNDARY, CURRENT]
SHORT = {MICRO: "Full Micro", EXACT: "Exact/exact", PLACEMENT: "Exact/uniform",
         BOUNDARY: "Aggregate/boundary", CURRENT: "Aggregate/uniform"}
COLORS = dict(zip(ORDER, ["#222222", "#2878b5", "#e07a1f", "#429c74", "#c23b55"]))


def paired_ci(values, seed=270000, reps=1000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for index in range(reps):
        estimates[index] = rng.choice(values, len(values), replace=True).mean()
    return values.mean(), *np.quantile(estimates, [0.025, 0.975])


def contrasts(runs):
    table = runs.pivot(index="replication_id", columns="model_variant", values="censored_T_last")
    definitions = {
        "oracle_control": (EXACT, MICRO),
        "placement_effect_exact_hazard": (PLACEMENT, EXACT),
        "hazard_abstraction_boundary_proxy": (BOUNDARY, EXACT),
        "full_current_error": (CURRENT, MICRO),
        "additional_uniform_effect_under_aggregate": (CURRENT, BOUNDARY),
    }
    rows = []
    for offset, (name, (left, right)) in enumerate(definitions.items()):
        estimate, low, high = paired_ci(table[left] - table[right], seed=270000 + offset)
        rows.append({"contrast": name, "left": left, "right": right,
                     "delta_RMST": estimate, "ci_low": low, "ci_high": high})
    placement_exact = table[PLACEMENT] - table[EXACT]
    placement_aggregate = table[CURRENT] - table[BOUNDARY]
    estimate, low, high = paired_ci(placement_aggregate - placement_exact, seed=270100)
    rows.append({"contrast": "placement_context_difference", "left": "(E-D)", "right": "(C-B)",
                 "delta_RMST": estimate, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def survival_plot(runs, output, horizon):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    grid = np.linspace(0, horizon, 401)
    for variant in ORDER:
        values = runs.loc[runs.model_variant == variant, "censored_T_last"].to_numpy()
        survival = np.array([(values > time).mean() for time in grid])
        ax.step(grid, survival, where="post", label=SHORT[variant], color=COLORS[variant])
    ax.set(xlabel="Terminal-arrival time", ylabel=r"$P(T_{last}>t)$", xlim=(0, horizon), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(output, dpi=220); plt.close(fig)


def rmst_plot(summary, intervals, output):
    data = summary.set_index("model_variant").loc[ORDER]
    ci = intervals.set_index("model_variant").loc[ORDER]
    y = np.arange(len(ORDER)); estimates = data.delta_RMST_vs_micro.to_numpy()
    errors = np.vstack([estimates-ci.ci_low.to_numpy(), ci.ci_high.to_numpy()-estimates])
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.axvline(0, color="#777777", lw=1)
    ax.errorbar(estimates, y, xerr=errors, fmt="o", color="#222222", capsize=3)
    ax.set(yticks=y, yticklabels=[SHORT[v] for v in ORDER], xlabel=r"$\Delta$RMST vs Full Micro (time units)")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(output, dpi=220); plt.close(fig)


def layer_plots(arrivals, delays, output_error, output_delay):
    reference = arrivals[arrivals.model_variant == MICRO][["replication_id", "layer_index", "T_i"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for variant in ORDER[1:]:
        other = arrivals[arrivals.model_variant == variant][["replication_id", "layer_index", "T_i"]]
        merged = reference.merge(other, on=["replication_id", "layer_index"], suffixes=("_micro", "_variant"))
        merged["error"] = merged.T_i_variant - merged.T_i_micro
        stats = merged.groupby("layer_index").error.quantile([.25, .5, .75]).unstack()
        ax.plot(stats.index, stats[.5], marker="o", label=SHORT[variant], color=COLORS[variant])
        ax.fill_between(stats.index, stats[.25], stats[.75], color=COLORS[variant], alpha=.15)
    ax.axhline(0, color="#777777", lw=1); ax.set(xlabel="Community index", ylabel="Paired cumulative arrival error")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(output_error, dpi=220); plt.close(fig)

    reference = delays[delays.model_variant == MICRO]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for variant in ORDER[1:]:
        other = delays[delays.model_variant == variant]
        merged = reference.merge(other, on=["replication_id", "layer_index"], suffixes=("_micro", "_variant"))
        merged["shift"] = merged.Delta_T_i_variant - merged.Delta_T_i_micro
        stats = merged.groupby("layer_index")["shift"].quantile([.25, .5, .75]).unstack()
        ax.plot(stats.index, stats[.5], marker="o", label=SHORT[variant], color=COLORS[variant])
        ax.fill_between(stats.index, stats[.25], stats[.75], color=COLORS[variant], alpha=.15)
    ax.axhline(0, color="#777777", lw=1); ax.set(xlabel="Transition $i\\rightarrow i+1$", ylabel="Paired local-delay shift")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(output_delay, dpi=220); plt.close(fig)


def trajectory_plot(path, output):
    grouped = []
    for chunk in pd.read_csv(path, usecols=["replication_id", "model_variant", "time", "I"], chunksize=500000):
        grouped.append(chunk.groupby(["replication_id", "model_variant", "time"], as_index=False).I.sum())
    totals = pd.concat(grouped).groupby(["replication_id", "model_variant", "time"], as_index=False).I.sum()
    stats = totals.groupby(["model_variant", "time"]).I.quantile([.25, .5, .75]).unstack()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for variant in ORDER:
        item = stats.loc[variant]
        ax.plot(item.index, item[.5], label=SHORT[variant], color=COLORS[variant])
        ax.fill_between(item.index, item[.25], item[.75], color=COLORS[variant], alpha=.12)
    ax.set(xlabel="Time", ylabel="Total infected nodes", xlim=(0, totals.time.max()))
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(output, dpi=220); plt.close(fig)


def diagnostics_plots(runs, output_counts, output_scatter):
    aggregate = runs[runs.model_variant.isin([BOUNDARY, CURRENT])]
    sums = aggregate.groupby("model_variant")[["premature_hazard_opportunities", "undefined_exact_placement_events"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    labels = [SHORT[v] for v in [BOUNDARY, CURRENT]]
    axes[0].bar(labels, sums.loc[[BOUNDARY, CURRENT], "premature_hazard_opportunities"], color=[COLORS[BOUNDARY], COLORS[CURRENT]])
    axes[1].bar(labels, sums.loc[[BOUNDARY, CURRENT], "undefined_exact_placement_events"], color=[COLORS[BOUNDARY], COLORS[CURRENT]])
    axes[0].set_ylabel("Positive aggregate / zero exact intervals"); axes[1].set_ylabel("Imports with undefined exact placement")
    for ax in axes: ax.tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(output_counts, dpi=220); plt.close(fig)

    reference = runs[runs.model_variant == MICRO][["replication_id", "censored_T_last"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for variant in [BOUNDARY, CURRENT]:
        item = runs[runs.model_variant == variant].merge(reference, on="replication_id", suffixes=("", "_micro"))
        error = item.censored_T_last - item.censored_T_last_micro
        ax.scatter(item.signed_hazard_error, error, s=18, alpha=.65, label=SHORT[variant], color=COLORS[variant])
    ax.axhline(0, color="#777777", lw=1); ax.set(xlabel="Integrated signed hazard error", ylabel="Terminal arrival error")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output_scatter, dpi=220); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("data_dir", type=Path); parser.add_argument("--report-dir", type=Path, default=Path("reports/ablation_hazard_placement")); args = parser.parse_args()
    data = args.data_dir; figures = args.report_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(data / "runs.csv"); arrivals = pd.read_csv(data / "arrival_times.csv"); delays = pd.read_csv(data / "local_delays.csv")
    summary = pd.read_csv(data / "summary_metrics.csv"); intervals = pd.read_csv(data / "bootstrap_intervals.csv")
    contrast_table = contrasts(runs); contrast_table.to_csv(data / "causal_contrasts.csv", index=False)
    survival_plot(runs, figures / "survival.png", runs.censored_T_last.max())
    rmst_plot(summary, intervals, figures / "rmst_shift.png")
    layer_plots(arrivals, delays, figures / "cumulative_error.png", figures / "local_delay_shift.png")
    trajectory_plot(data / "trajectories.csv", figures / "infected_dynamics.png")
    diagnostics_plots(runs, figures / "premature_hazard.png", figures / "hazard_vs_arrival.png")
    print(contrast_table.to_string(index=False))


if __name__ == "__main__":
    main()
