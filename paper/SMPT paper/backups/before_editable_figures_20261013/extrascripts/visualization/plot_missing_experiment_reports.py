#!/usr/bin/env python3
"""Make report figures from the completed missing-experiment outputs only.

The script intentionally performs no simulation.  It reads the saved CSV/JSON
outputs and writes one 300-dpi two-panel PNG to each report's figures folder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
REPORTS = ROOT / "reports" / "claude_experiments"
COLORS = {"CE-01": "#0072B2", "CE-06": "#D55E00", "Full": "#000000",
          "Event-driven": "#009E73", "tau=0.25": "#56B4E9",
          "tau=0.5": "#E69F00", "tau=1": "#CC79A7",
          "near": "#009E73", "random": "#E69F00", "far": "#D55E00"}


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 11, "legend.fontsize": 8.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})


def _zero(ax) -> None:
    ax.axhline(0, color="0.35", lw=0.9, zorder=0)
    ax.grid(axis="y", color="0.9", lw=0.7, zorder=0)


def _save(fig, report: str) -> Path:
    path = REPORTS / report / "figures" / "report_figure.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def ce12() -> Path:
    summary = pd.read_csv(OUT / "claude_experiments/ce12_power_calibration/power_summary.csv")
    power = pd.read_csv(OUT / "claude_experiments/ce12_power_calibration/empirical_power.csv")
    summary["source"] = np.where(summary.experiment.eq("CE-01"), "CE-01", "CE-06")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    margins = [0.025, 0.05, 0.10]
    positions, values, colors = [], [], []
    for i, margin in enumerate(margins):
        for j, source in enumerate(("CE-01", "CE-06")):
            vals = summary.loc[(summary.margin_fraction_T_end == margin) &
                               (summary.source == source), "paired_normal_N80"].to_numpy()
            positions.append(i * 3 + j); values.append(vals); colors.append(COLORS[source])
    boxes = ax0.boxplot(values, positions=positions, widths=.72, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "lw": 1.4})
    for box, color in zip(boxes["boxes"], colors):
        box.set(facecolor=color, alpha=.72, edgecolor=color)
    for pos, vals, color in zip(positions, values, colors):
        ax0.scatter(np.full(len(vals), pos), vals, s=12, color=color, alpha=.35, zorder=3)
    ax0.set_xticks([.5, 3.5, 6.5], ["2.5%", "5%", "10%"])
    ax0.set_xlabel("Signed-effect margin (% of each scenario $T_{end}$)")
    ax0.set_ylabel("Paired normal N80 (pairs)")
    ax0.set_title("Required N80 across saved comparisons")
    ax0.legend([plt.Line2D([], [], color=COLORS[x], lw=7) for x in ("CE-01", "CE-06")],
               ["CE-01", "CE-06 regime map"], frameon=False, loc="upper right")
    ax0.text(.02, .02, "Boxes: IQR; whiskers: range; dots: comparisons", transform=ax0.transAxes, fontsize=8)
    # CE-01 has 100 paired runs; the CE-06 regime-map cells have 60.
    now = power.loc[((power.experiment == "CE-01") & power.sample_size_pairs.eq(100)) |
                    ((power.experiment == "CE-06") & power.sample_size_pairs.eq(60))].copy()
    now["source"] = np.where(now.experiment.eq("CE-01"), "CE-01", "CE-06")
    for source in ("CE-01", "CE-06"):
        group = now.loc[now.source.eq(source)]
        for margin in margins:
            vals = group.loc[group.margin_fraction_T_end.eq(margin), "empirical_power"]
            x = margins.index(margin) + (-.12 if source == "CE-01" else .12)
            ax1.scatter(np.full(len(vals), x), vals, color=COLORS[source], alpha=.30, s=13)
            ax1.plot(x, vals.median(), "o", color=COLORS[source], ms=6, zorder=3)
    ax1.axhline(.80, color="0.25", ls="--", lw=1, label="80% target")
    ax1.set_xticks(range(3), ["2.5%", "5%", "10%"])
    ax1.set_ylim(0, 1.04); ax1.set_ylabel("Empirical paired-test power")
    ax1.set_xlabel("Signed-effect margin (% of each scenario $T_{end}$)")
    ax1.set_title("Empirical power at completed sample sizes")
    ax1.legend([plt.Line2D([], [], color=COLORS["CE-01"], marker="o", lw=0),
                plt.Line2D([], [], color=COLORS["CE-06"], marker="o", lw=0),
                plt.Line2D([], [], color="0.25", ls="--")],
               ["CE-01: n = 100", "CE-06: n = 60", "80% target"], frameon=False, loc="lower right")
    fig.suptitle("CE-12 paired power calibration (restricted arrival times)", y=1.02, fontsize=12)
    return _save(fig, "ce12_power_calibration")


def _layer_delta(arrivals: pd.DataFrame, reference: str, models: list[str]) -> pd.DataFrame:
    wide = arrivals.pivot_table(index=["replication_id", "layer"], columns="model",
                                values="restricted_T_i", aggfunc="first")
    rows = []
    for model in models:
        delta = wide[model] - wide[reference]
        rows.extend({"model": model, "layer": layer, "delta": value}
                    for (_, layer), value in delta.dropna().items())
    return pd.DataFrame(rows).groupby(["model", "layer"], as_index=False).delta.mean()


def oracle() -> Path:
    base = OUT / "missing_experiments/oracle_bridge_ceiling/full"
    summary = pd.read_csv(base / "summary.csv")
    arrivals = pd.read_csv(base / "arrivals.csv")
    labels = {"MM_uniform": "Uniform", "MM_oracle_source_bridge": "Oracle bridge",
              "MM_degree_proxy": "Degree proxy"}
    order = list(labels)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    rows = summary.set_index("model").loc[order]
    x = np.arange(len(order)); est = rows.delta_RMST.to_numpy()
    ax0.errorbar(x, est, yerr=np.vstack([est - rows.delta_RMST_ci95_low, rows.delta_RMST_ci95_high - est]),
                 fmt="o", color="#0072B2", capsize=4, lw=1.6)
    _zero(ax0); ax0.set_xticks(x, [labels[k] for k in order]); ax0.set_ylabel("Δ restricted mean arrival time\n(vs FullMicro; negative = earlier)")
    ax0.set_title("Terminal layer (layer 9): paired 95% CI")
    layers = _layer_delta(arrivals, "FullMicro", order)
    for model in order:
        frame = layers[layers.model.eq(model)]
        ax1.plot(frame.layer, frame.delta, "o-", lw=2, color={order[0]: "#E69F00", order[1]: "#009E73", order[2]: "#CC79A7"}[model], label=labels[model])
    _zero(ax1); ax1.set_xticks(range(5)); ax1.set_xlabel("Arrival layer"); ax1.set_ylabel("Mean Δ restricted arrival time\n(vs FullMicro; negative = earlier)")
    ax1.set_title("Layerwise accumulation"); ax1.legend(frameon=False)
    fig.suptitle("Source-side bridge oracle ablation (restricted at horizon = 4,000; n = 100 paired runs)", y=1.02, fontsize=12)
    return _save(fig, "oracle_bridge_ceiling")


def synchronization() -> Path:
    base = OUT / "missing_experiments/synchronization_audit/full"
    summary = pd.read_csv(base / "summary.csv"); arrivals = pd.read_csv(base / "arrivals.csv")
    models = ["ProductionMM_tau0.25", "ProductionMM_tau0.5", "ProductionMM_tau1"]
    nice = {models[0]: "tau=0.25", models[1]: "tau=0.5", models[2]: "tau=1"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    xs = np.arange(3)
    for reference, offset, color, name in [("FullMicro", -.13, COLORS["Full"], "FullMicro"), ("EventDrivenAggregate", .13, COLORS["Event-driven"], "Event-driven")]:
        rows = summary[(summary.reference == reference) & summary.model.isin(models)].set_index("model").loc[models]
        est = rows.delta_RMST.to_numpy()
        ax0.errorbar(xs + offset, est, yerr=np.vstack([est - rows.delta_RMST_ci95_low, rows.delta_RMST_ci95_high - est]),
                     fmt="o", color=color, capsize=3, lw=1.3, label=name)
    _zero(ax0); ax0.set_xticks(xs, [nice[m] for m in models]); ax0.set_xlabel("Production synchronization interval")
    ax0.set_ylabel("Terminal Δ restricted mean arrival time\n(negative = earlier)"); ax0.set_title("Terminal layer: paired 95% CI"); ax0.legend(frameon=False)
    layers = _layer_delta(arrivals, "FullMicro", models)
    for model in models:
        frame = layers[layers.model.eq(model)]
        ax1.plot(frame.layer, frame.delta, "o-", lw=2, color=COLORS[nice[model]], label=nice[model])
    _zero(ax1); ax1.set_xticks(range(5)); ax1.set_xlabel("Arrival layer")
    ax1.set_ylabel("Mean Δ restricted arrival time\n(vs FullMicro; negative = earlier)"); ax1.set_title("Layerwise accumulation across tau"); ax1.legend(frameon=False)
    fig.suptitle("Synchronization audit (restricted at horizon = 4,000; n = 100 paired runs)", y=1.02, fontsize=12)
    return _save(fig, "synchronization_audit")


def gateway() -> Path:
    base = OUT / "missing_experiments/gateway_distance/full"
    summary = pd.read_csv(base / "summary.csv"); contrasts = pd.read_csv(base / "contrasts.csv")
    terminal = summary[summary.layer.eq(summary.layer.max())].set_index("mode").loc[["near", "random", "far"]]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    modes = ["near", "random", "far"]; x = np.arange(3)
    bars = ax0.bar(x, terminal.delta_RMST, color=[COLORS[m] for m in modes], width=.62)
    _zero(ax0); ax0.set_xticks(x, ["Near", "Random", "Far"]); ax0.set_ylabel("Terminal Δ restricted mean arrival time\n(MicroMacro − Micro; negative = earlier)")
    ax0.set_title("Terminal discrepancy by gateway architecture\n(restricted at horizon = 2,000; 100 paired network blocks)", fontsize=10.5)
    ax0.set_ylim(-560, 170)  # preserve room for the architecture annotations
    for i, (_, row) in enumerate(terminal.iterrows()):
        ax0.text(i, row.delta_RMST + (18 if row.delta_RMST >= 0 else -28),
                 f"Dmin={row.D_min_mean:.2f}\nDmean={row.D_mean_mean:.2f}\nΔcomp={row.delta_completion:+.2f}", ha="center", va="bottom" if row.delta_RMST >= 0 else "top", fontsize=8)
    labels = ["Random − near", "Far − near", "Far − random"]
    y = np.arange(3); est = contrasts.estimate.to_numpy()
    ax1.errorbar(est, y, xerr=np.vstack([est - contrasts.ci95_low, contrasts.ci95_high - est]), fmt="o", color="#0072B2", capsize=4, lw=1.6)
    ax1.axvline(0, color="0.35", lw=.9); ax1.grid(axis="x", color="0.9", lw=.7)
    ax1.set_yticks(y, labels); ax1.invert_yaxis(); ax1.set_xlabel("Paired terminal-error contrast (time units)")
    ax1.set_title("Paired network-block 95% CI")
    return _save(fig, "gateway_distance")


def main() -> None:
    _style()
    for make in (ce12, oracle, synchronization, gateway):
        print(make())


if __name__ == "__main__":
    main()
