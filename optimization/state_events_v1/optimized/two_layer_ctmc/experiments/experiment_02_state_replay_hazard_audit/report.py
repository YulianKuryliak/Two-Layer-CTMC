from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from .config import STAGE_RUNS


def _value(stats: pd.DataFrame, metric: str, column: str = "mean") -> float:
    row = stats[stats["metric"] == metric]
    if len(row) != 1:
        raise ValueError(f"missing unique statistic {metric}")
    return float(row.iloc[0][column])


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value == 0:
        return "0"
    return f"{value:.5g}"


def _statistics_rows(stats: pd.DataFrame) -> str:
    labels = {
        "A_exact": r"$A_{\mathrm{exact}}$", "A_aggregate": r"$A_{\mathrm{aggregate}}$",
        "signed_hazard_mass": "Signed hazard mass", "absolute_hazard_mass": "Absolute hazard mass",
        "over_hazard_mass": "Over-hazard mass", "under_hazard_mass": "Under-hazard mass",
        "premature_hazard_mass": "Premature hazard mass", "missed_hazard_mass": "Missed hazard mass",
        "premature_fraction": "Premature fraction",
    }
    lines = []
    for row in stats.itertuples(index=False):
        lines.append(f"{labels.get(row.metric, row.metric)} & {_fmt(row.mean)} & {_fmt(row.median)} & [{_fmt(row.ci95_low)}, {_fmt(row.ci95_high)}] \\\\")
    return "\n".join(lines)


def _stratified_rows(table: pd.DataFrame, group_column: str) -> str:
    selected = table[table["metric"].isin(["signed_hazard_mass", "absolute_hazard_mass", "premature_fraction"])]
    pivot = selected.pivot(index=group_column, columns="metric", values="mean").reset_index()
    lines = []
    for _, row in pivot.iterrows():
        raw_label = row[group_column]
        label = str(int(raw_label)) if isinstance(raw_label, float) and raw_label.is_integer() else str(raw_label)
        label = label.replace("_", r"\_")
        lines.append(
            f"{label} & {_fmt(row['signed_hazard_mass'])} & {_fmt(row['absolute_hazard_mass'])} & {_fmt(row['premature_fraction'])} \\\\"
        )
    return "\n".join(lines)


def render_report(stage_dir: Path, report_dir: Path, tests_text: str = "See recorded test-results file.") -> Path:
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(stage_dir / "summaries" / "overall_statistics.csv")
    metadata = pd.read_csv(stage_dir / "summaries" / "replay_metadata.csv")
    layer_stats = pd.read_csv(stage_dir / "summaries" / "layer_summary.csv")
    phase_stats = pd.read_csv(stage_dir / "summaries" / "phase_summary.csv")
    phase_premature = phase_stats[phase_stats["metric"] == "premature_fraction"].set_index("epidemic_phase")
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_target = report_dir / "figures"
    figures_target.mkdir(exist_ok=True)
    for figure in sorted((stage_dir / "figures").glob("*.png")):
        shutil.copy2(figure, figures_target / figure.name)
    development = stage_dir.name.startswith("validation_20")
    status = "Development report: 20-run validation, not final conclusions" if development else f"Final report: {manifest['n_runs']} replications"
    results_heading = "Validation results" if development else "Final results"
    result_adjective = "validation" if development else "final"
    analysis_adjective = "development" if development else "final"
    test_summary = tests_text.replace("_", r"\_").replace("%", r"\%")
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx,hyperref,float,longtable}}
\usepackage[T1]{{fontenc}}
\title{{Experiment 02 --- State-Replay Hazard Audit}}
\author{{Two-Layer CTMC reproducibility audit}}
\date{{{status}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This experiment measures pure formula-level rate-abstraction error by evaluating exact active-edge and aggregate prevalence hazards on identical Full Micro states. This document reports {manifest['n_runs']} replay replications. {'These are validation results and must not be read as final scientific conclusions.' if development else ''}
\end{{abstract}}

\section{{Research question and relation to Experiment 01}}
Experiment 01 separated hazard and placement mechanisms using separate stochastic trajectories. Experiment 02 removes trajectory, placement, and synchronization differences: it asks how much the aggregate formula differs from the exact inter-community transmission rate on the same microscopic Full Micro state.

\section{{Experimental design}}
The baseline is a chain of $L=10$ connected sparse random communities, $n=100$, mean internal degree approximately 6, and five unit-weight inter-community edges per adjacent pair. Rates are $\beta_{{\rm in}}=\beta_{{\rm out}}=\beta_{{\rm macro}}=0.003$, $\gamma=0$, and $T_{{\rm end}}=4000$. No aggregate trajectory is simulated. The current stage contains {manifest['n_runs']} independent network/event replications; inference first aggregates within replication.

\section{{Mathematical definitions}}
For directed interface $i\to j$ and microscopic state $x$,
\begin{{align}}
a^*_{{ij}}(x)&=\beta_{{\rm out}}\sum_{{(u,v)\in E_{{ij}}}}w_{{uv}}\mathbf{{1}}[X_u=I,X_v=S],\\
\lambda_{{ij}}(x)&=\beta_{{\rm macro}}W_{{ij}}\bar i_i(x)\bar s_j(x).
\end{{align}}
Signed, absolute, and symmetric relative errors are $\lambda-a^*$, $|\lambda-a^*|$, and $2|\lambda-a^*|/(\lambda+a^*)$ when the denominator is positive. All hazard-mass quantities are exact sums of piecewise-constant interval values times interval duration.

\section{{Reconstruction and replay}}
Full Micro was \emph{{not rerun}}. The canonical Experiment 01 final-100 event archive supplies replications 1--{manifest['n_runs']}. With $\gamma=0$, the seed and every node-level infection event determine the microscopic state exactly. Networks are regenerated with the archived network seeds, and every event edge and community identifier is checked. A post-event state is used on $[t_r,t_{{r+1}})$ and the final interval is clipped at $T_{{\rm end}}$. {int(metadata['events_beyond_horizon_ignored'].sum())} logged events beyond the horizon were ignored before state mutation. Experiment 01 did not archive edge lists, so regeneration depends on the current network generator and NetworkX version; Experiment 02 records stable graph hashes.

\section{{State categories and stratification}}
The mutually exclusive categories are both-zero, premature-positive, missed-positive, overestimation, underestimation, and approximate-match. Approximate match uses absolute tolerance $10^{{-12}}$ and relative tolerance $10^{{-9}}$ after zero cases are classified. Premature-positive cases are retained explicitly. For monotone SI dynamics, source phases are not-started ($I/N=0$), early ($0<I/N\le0.10$), middle ($0.10<I/N\le0.50$), and late ($I/N>0.50$). These fixed bands distinguish establishment, growth, and saturation while remaining comparable across replications. The archived descriptive stratified table covers source infected and destination susceptible fractions, both boundary-node counts, active-edge count and weight, layer, both times since first local infection, and phase. Those interval-level summaries are descriptive only; layer and phase uncertainty is computed after within-replication aggregation.

\section{{Sanity and semantic checks}}
Automated tests cover brute-force exact hazards, the production aggregate formula, zero and positive active-edge cases, exact piecewise integration, premature mass, deterministic replay, reconstruction correctness, Experiment 01 identifiers, and production-file hashes. During replay, every state is also checked against the production MicroEngine total weighted infection rate and every aggregate entry against MacroEngine. Maximum replay differences were {_fmt(metadata['max_micro_rate_difference'].max())} (micro) and {_fmt(metadata['max_macro_formula_difference'].max())} (macro). {test_summary}

\section{{{results_heading}}}
The replay contains {int(manifest['positive_duration_state_intervals'])} exact state intervals and {int(manifest['directed_pair_interval_rows'])} directed-interface rows. Confidence intervals below bootstrap replications, not event intervals.
\begin{{table}}[H]\centering
\caption{{Replication-level {result_adjective} summaries; bootstrap 95\% confidence intervals for the mean.}}
\begin{{tabular}}{{lrrr}}\toprule
Metric & Mean & Median & 95\% CI \\\midrule
{_statistics_rows(stats)}
\bottomrule\end{{tabular}}\end{{table}}

\begin{{table}}[H]\centering
\caption{{Mean replication-level results by chain interface. Each interface includes both directions.}}
\begin{{tabular}}{{rrrr}}\toprule
Interface & Signed mass & Absolute mass & Premature fraction \\\midrule
{_stratified_rows(layer_stats, 'interface_index')}
\bottomrule\end{{tabular}}\end{{table}}

\begin{{table}}[H]\centering
\caption{{Mean replication-level results by source epidemic phase.}}
\begin{{tabular}}{{lrrr}}\toprule
Phase & Signed mass & Absolute mass & Premature fraction \\\midrule
{_stratified_rows(phase_stats, 'epidemic_phase')}
\bottomrule\end{{tabular}}\end{{table}}

\begin{{figure}}[H]\centering\includegraphics[width=.62\linewidth]{{figures/01_exact_vs_aggregate.png}}\caption{{Exact and aggregate hazards evaluated on common Full Micro states. Points are a visualization subsample; inference is replication-level.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.49\linewidth]{{figures/02_signed_error_distribution.png}}\includegraphics[width=.49\linewidth]{{figures/03_over_under_mass.png}}\caption{{Replication distributions of integrated signed error and its over/under components.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.62\linewidth]{{figures/04_premature_fraction.png}}\caption{{Fraction of aggregate cumulative hazard accumulated when exact inter-community transmission is impossible.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.49\linewidth]{{figures/05_error_vs_source_fraction.png}}\includegraphics[width=.49\linewidth]{{figures/06_error_vs_active_edges.png}}\caption{{Descriptive event-level stratification. Intervals are dependent and are not treated as independent observations.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.49\linewidth]{{figures/07_layer_integrated_error.png}}\includegraphics[width=.49\linewidth]{{figures/08_representative_time_series.png}}\caption{{Layer-wise integrated error and representative common-state time series. Uncertainty claims use replication summaries.}}\end{{figure}}

\section{{Scientific interpretation}}
The signed mass must be interpreted together with absolute, over-, under-, and cumulative premature components: cancellation can make signed error small even when local discrepancies are large. The reported {result_adjective} estimates characterize pure formula-level rate-abstraction error for this baseline. Both over- and under-hazard mass are present, although the replication-level mean over-hazard component is larger for this baseline. This does not establish that aggregate hazards universally overestimate exact hazards.

Across replications, mean cumulative exact and aggregate hazards were {_fmt(_value(stats, 'A_exact'))} and {_fmt(_value(stats, 'A_aggregate'))}; mean signed and absolute discrepancy masses were {_fmt(_value(stats, 'signed_hazard_mass'))} and {_fmt(_value(stats, 'absolute_hazard_mass'))}. Mean over-hazard mass ({_fmt(_value(stats, 'over_hazard_mass'))}) exceeded mean under-hazard mass ({_fmt(_value(stats, 'under_hazard_mass'))}), but the nonzero under component confirms state-dependent sign changes. A mean {_fmt(100 * _value(stats, 'premature_fraction'))}\% of aggregate cumulative hazard accumulated in states with zero exact active-edge hazard (95\% bootstrap interval {_fmt(100 * _value(stats, 'premature_fraction', 'ci95_low'))}--{_fmt(100 * _value(stats, 'premature_fraction', 'ci95_high'))}\%). The phase-stratified mean premature fractions were {_fmt(100 * phase_premature.loc['early', 'mean'])}\% early, {_fmt(100 * phase_premature.loc['middle', 'mean'])}\% middle, and {_fmt(100 * phase_premature.loc['late', 'mean'])}\% late. These phase values describe where formula mismatch occurs on the replayed trajectories; they are not independent event-level estimates.

\section{{What this experiment explains---and does not explain}}
It explains discrepancies between two hazard formulas on common Full Micro states. It cannot explain import-placement error, production synchronization error, full endogenous trajectory error, event-time bias by signed error alone, or robustness to other topologies or $\gamma>0$.

\section{{Limitations}}
This {analysis_adjective} analysis has {manifest['n_runs']} replications. Network edges were regenerated rather than archived by Experiment 01. Event-level correlations are descriptive because intervals within a replication are dependent. The phase thresholds are mechanistic fixed bands, not data-selected changepoints. Generalization remains limited to the chain topology and $\gamma=0$ baseline.

\section{{Recommended next experiments}}
Experiment 03 is the Production vs Event-Driven Synchronization Audit, intended to explain the difference between approximately 7.5\% error for event-driven aggregate/uniform Experiment 01 and approximately 42.6\% in the production sparse benchmark. Experiment 04 is the Gateway-Distance Experiment with near/medium/far gateway placement at fixed $n,m,d,k,\beta$, and community count. Experiment 04 was not run here.

\section{{Exact reproduction commands}}
\begin{{verbatim}}
.venv/bin/python -m unittest tests.test_experiment_02_state_replay_hazard_audit
.venv/bin/python extrascripts/runners/run_experiment_02_state_replay_hazard_audit.py --stage smoke --resume
.venv/bin/python extrascripts/runners/run_experiment_02_state_replay_hazard_audit.py --stage validation --resume
.venv/bin/python extrascripts/analysis/analyze_experiment_02_state_replay_hazard_audit.py --stage validation
# Only after explicit approval:
.venv/bin/python extrascripts/runners/run_experiment_02_state_replay_hazard_audit.py --stage final --audit-approved --resume
\end{{verbatim}}
\end{{document}}
"""
    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=report_dir, text=True, capture_output=True,
    )
    (report_dir / "latex_build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not (report_dir / "report.pdf").exists():
        raise RuntimeError(f"LaTeX build failed; see {report_dir / 'latex_build.log'}")
    return tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 02 and build its LaTeX report")
    parser.add_argument("--stage", choices=tuple(STAGE_RUNS), default="validation")
    parser.add_argument("--output-root", type=Path, default=Path("output/experiment_02_state_replay_hazard_audit"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/experiment_02_state_replay_hazard_audit"))
    args = parser.parse_args()
    stage_dir = args.output_root / f"{args.stage}_{STAGE_RUNS[args.stage]}"
    tests_path = args.output_root / "test_results.txt"
    tests_text = tests_path.read_text(encoding="utf-8").strip() if tests_path.exists() else "Test results file not present."
    render_report(stage_dir, args.report_dir, tests_text)


if __name__ == "__main__":
    main()
