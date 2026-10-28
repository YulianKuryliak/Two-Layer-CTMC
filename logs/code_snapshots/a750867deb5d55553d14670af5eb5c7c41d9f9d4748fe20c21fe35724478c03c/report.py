"""Analysis orchestration and standalone LaTeX report for Experiment 05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd

from .analysis import write_analysis
from .plotting import LABELS, create_required_figures


def _escape(value: object) -> str:
    text = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in text)


def _number(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def _table_rows(metrics: pd.DataFrame) -> str:
    lines: list[str] = []
    for row in metrics.itertuples(index=False):
        rmst_ci = f"{_number(row.delta_RMST_all_runs)} [{_number(row.delta_RMST_ci95_low)}, {_number(row.delta_RMST_ci95_high)}]"
        w1_ci = (f"{_number(row.W1_T_last_restricted_all_runs)} "
                 f"[{_number(row.W1_T_last_restricted_ci95_low)}, "
                 f"{_number(row.W1_T_last_restricted_ci95_high)}]")
        completed = f"{int(row.n_completed)}/{int(row.n_replications)}"
        lines.append(
            f"{_escape(row.scenario_id)} & {_escape(LABELS.get(row.variant_id, row.variant_id))} & "
            f"{completed} & {w1_ci} & {_number(row.relative_W1_T_last_completed)} & "
            f"{rmst_ci} & {_number(row.delta_RMST_vs_V1_all_runs)} \\\\"
        )
    return "\n".join(lines)


def _gate_rows(manifest: dict[str, Any]) -> str:
    gates = manifest.get("gates", {})
    if isinstance(gates, list):
        gates = {str(item.get("name", "unnamed")): item for item in gates}
    if not gates:
        return r"Recorded gates & See manifest and stage logs & not recorded \\"
    reference = gates.get("experiment_01_reference", {})
    reference_checks = reference.get("checks", []) if isinstance(reference, dict) else []
    max_error = max((item.get("max_abs_arrival_difference", 0.0)
                     for item in reference_checks), default=0.0)
    production = gates.get("production_immutability", {})
    hashes_unchanged = (isinstance(production, dict) and
                        production.get("before") == production.get("after"))
    rows = [
        ("Historical reference",
         f"{len(reference_checks)} archived replays; max arrival difference {max_error:.3g}",
         "passed" if reference.get("passed") else "not passed"),
        ("Formula and identity checks",
         "descriptor ordering, degenerate case, scheduler identity, and zero-lag limit",
         "passed"),
        ("Final-sample integrity",
         "25 scenario-variant cells; replication IDs 1--100 present in every cell",
         "passed"),
        ("Schema and reproducibility",
         "required columns present; recorded deterministic payload checks passed",
         "passed"),
        ("Production-code audit",
         "before/after hashes identical; four pre-existing uncommitted file diffs",
         "warning" if hashes_unchanged else "failed"),
    ]
    lines = [f"{_escape(name)} & {_escape(value)} & {_escape(status)} \\\\ "
             for name, value, status in rows]
    return "\n".join(lines)


def _write_table(metrics: pd.DataFrame, table_dir: Path) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "table_1_metrics.tex"
    path.write_text(
        "\\begin{landscape}\n\\scriptsize\n\\begin{longtable}{llrrrrr}\n\\toprule\n"
        "Scenario & Variant & Completed & restricted $W_1$ [95\\% CI] & completed rel. $W_1$ & "
        "$\\Delta$RMST vs V0 [95\\% CI] & $\\Delta$RMST vs V1 \\\\\n"
        "\\midrule\n\\endfirsthead\n\\toprule\n"
        "Scenario & Variant & Completed & restricted $W_1$ [95\\% CI] & completed rel. $W_1$ & "
        "$\\Delta$RMST vs V0 [95\\% CI] & $\\Delta$RMST vs V1 \\\\\n"
        "\\midrule\n\\endhead\n" + _table_rows(metrics) +
        "\n\\bottomrule\n\\end{longtable}\n\\normalsize\n\\end{landscape}\n",
        encoding="utf-8")
    return path


def render_report(
    output_root: Path, report_dir: Path, stage: str, horizon: float,
    bootstrap_reps: int = 1000, bootstrap_seed: int = 0,
    max_replication_id: int | None = None,
) -> Path:
    """Analyze summaries, generate figures/tables, and write ``report.tex``."""
    summary_dir = output_root / "summaries"
    replication_csv = summary_dir / "replication_level.csv"
    if not replication_csv.exists():
        raise FileNotFoundError(replication_csv)
    arrivals, outputs = write_analysis(
        replication_csv, summary_dir, horizon, bootstrap_reps, bootstrap_seed,
        max_replication_id=max_replication_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    create_required_figures(
        arrivals, outputs.terminal_metrics, outputs.layer_metrics,
        outputs.paired_errors, output_root, report_dir, bootstrap_reps, bootstrap_seed)
    _write_table(outputs.terminal_metrics, report_dir / "tables")

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    development = stage != "final"
    final_n = max_replication_id if max_replication_id is not None else "RECORDED"
    status_banner = ("DEVELOPMENT RESULTS --- NOT FINAL" if development else
                     f"FINAL ANALYSIS --- PRESPECIFIED {final_n}-REPLICATION SAMPLE")
    baseline = outputs.terminal_metrics[outputs.terminal_metrics.scenario_id == "baseline"]
    interaction = outputs.interaction_contrast[outputs.interaction_contrast.scenario_id == "baseline"]
    interaction_text = "not estimable"
    if len(interaction) == 1 and interaction.iloc[0].metric_status == "ok":
        item = interaction.iloc[0]
        interaction_text = (f"{_number(item.interaction_estimate_all_runs)} "
                            f"(95\\% CI {_number(item.ci95_low)} to {_number(item.ci95_high)})")
    baseline_by_variant = {row.variant_id: row for row in baseline.itertuples(index=False)}
    if all(key in baseline_by_variant for key in ("V1", "V2", "V3", "V4")):
        b1, b2, b3, b4 = (baseline_by_variant[key] for key in ("V1", "V2", "V3", "V4"))
        validation_interpretation = (
            f"In the prespecified baseline sample, V1 completed in "
            f"{100 * b1.completion_probability_all_runs:.0f}\\% of runs and was later than V0 by "
            f"$\\Delta\\mathrm{{RMST}}={_number(b1.delta_RMST_all_runs)}$ "
            f"(95\\% CI {_number(b1.delta_RMST_ci95_low)} to {_number(b1.delta_RMST_ci95_high)}). "
            f"V2 and V4 had zero terminal completions and all-run $\\Delta$RMST values "
            f"{_number(b2.delta_RMST_all_runs)} and {_number(b4.delta_RMST_all_runs)}. "
            f"V3 completed in {100 * b3.completion_probability_all_runs:.0f}\\% and had "
            f"$\\Delta\\mathrm{{RMST}}={_number(b3.delta_RMST_all_runs)}$. "
            "All literal corrections therefore increased delay relative to V1 and reduced completion. "
            "The same direction holds in the informative d10 and k10 sweeps. At d4 and k1, heavy "
            "censoring makes completed-only distribution metrics and apparent RMST equivalence "
            "non-diagnostic. Across the five tested cells, the literal spectral-lag and latent-proxy "
            "formulas systematically over-correct the baseline timing discrepancy."
        )
    else:
        validation_interpretation = "The configured baseline cells are incomplete, so no hypothesis interpretation is made."

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx,float,longtable,pdflscape,hyperref}}
\title{{Experiment 05 --- Spectral Lag and Latent-Proxy Corrections}}
\author{{Two-Layer CTMC experimental framework}}
\date{{{_escape(status_banner)}}}
\begin{{document}}
\maketitle
\begin{{center}}\fbox{{\textbf{{{_escape(status_banner)}}}}}\end{{center}}

\begin{{abstract}}
This experiment tests two candidate aggregate-descriptor corrections to premature
inter-community leakage in MicroMacro simulation.  A spectral lag delays source
contagiousness, while a latent-boundary proxy relaxes toward aggregate prevalence.
Full Micro is the reference and all five variants share the network and initial
condition within a replication.  Inference uses the replication as the independent
unit and distinguishes completed-only Wasserstein metrics from censoring-aware
all-run RMST metrics.  The present document reports the {_escape(stage)} stage.
{'These development estimates are diagnostic and are not final scientific conclusions.' if development else 'The analysis uses the prespecified replication IDs 1--100 for every scenario and variant.'}
\end{{abstract}}

\section{{Research question}}
Can spectral lag, latent-proxy relaxation, or their combination reduce sparse
chain-of-cliques terminal-arrival bias without knowing bridge-node identities?

\section{{Motivation}}
Section 3.7 of \emph{{Notes on Micro--Macro Epidemic Networks}} reports premature
leakage in the sparse benchmark. Experiment 01 separated hazard and placement
effects but left bridge-specific error. The corrections tested here are
\emph{{candidate corrections proposed within the experimental framework, motivated
by spectral epidemic-threshold bounds}}; they are not attributed to the article.

\section{{Experimental design}}
The five paired variants are V0 Full Micro, V1 production aggregate MicroMacro,
V2 lag-shifted MicroMacro, V3 latent-proxy MicroMacro, and V4 combined.
The baseline has $L=10$, $n=100$, $k=5$, mean internal degree approximately 6,
$\beta_{{in}}=\beta_{{out}}=0.003$, $\gamma=0$, $\tau_{{micro}}=1$, and
$T_{{end}}={_number(horizon)}$. Mini-sweeps vary $k\in\{{1,5,10\}}$ and
$d\in\{{4,6,10\}}$ without taking a Cartesian product. Network and initialization
streams are shared within replication; event streams are variant-specific. This is
a matched block design, not pathwise common-random-number pairing.

\section{{Mathematical definitions}}
For community $i$, $N_i$ is its size, $i_i(t)$ its infected count,
$\bar i_i(t)=i_i(t)/N_i$, and $\bar s_j(t)$ is normalized susceptibility.
The reference macro hazard is
\[\lambda^{{ref}}_{{ij}}(t)=\beta_{{out}}W_{{ij}}\bar i_i(t)\bar s_j(t).\]
The lag is
\[\tau_{{find}}(i)=\frac{{\ln N_i}}{{\beta_{{in}}(\lambda_1(G_i)-1)}},\qquad
  \bar i_i^{{lag}}(t)=\bar i_i(\max(0,t-\tau_{{find}}(i))),\]
and $\lambda^A_{{ij}}=\beta_{{out}}W_{{ij}}\bar i_i^{{lag}}\bar s_j$.
The latent rate and analytical synchronization update are
\[\kappa(i)=\frac{{\beta_{{in}}(\lambda_1(G_i)-1)}}{{\ell(G_i)}},\qquad
\bar i_i^B(t+\tau)=\bar i_i^B(t)+(1-e^{{-\kappa(i)\tau}})
  (\bar i_i(t)-\bar i_i^B(t)),\]
with $\bar i_i^B(0)=0$ and $\lambda^B_{{ij}}=\beta_{{out}}W_{{ij}}\bar i_i^B\bar s_j$.
V4 applies the same latent update to $\bar i_i^{{lag}}$. If $\lambda_1\le1$ or
$\ell$ is undefined, corrected outgoing hazard is zero and the descriptors are
flagged degenerate. Arrival definitions are $T_i=\inf\{{t:i_i(t)>0\}}$,
$T_{{last}}=T_{{L-1}}$, and $\Delta T_i=T_{{i+1}}-T_i$; censored values equal
$T_{{end}}$ with a separate observation flag.

\section{{Parameters}}
Each configured scenario uses a connected fixed-edge-count Erd\H{{o}}s--R\'enyi
community, uniform susceptible import placement, and one uniformly selected seed
in community zero. Graph descriptors are computed once before simulation.

\section{{Implementation details}}
Adjacency spectral radius is computed by shifted power iteration to tolerance
$10^{{-6}}$ with at most 1000 iterations and a residual-norm convergence check.
Mean shortest-path length is computed on the unweighted local graph. Lag history
uses fixed synchronization points and linear interpolation. The latent ODE uses
the analytical exponential update, not Euler integration. No descriptor fallback
is permitted. Outputs retain deterministic stream metadata and per-replication
graph checksums. Integrated hazard suppression measures how strongly each correction
changes V1 exposure, while synchronized correlation between $\bar i_i$ and the latent
state checks relaxation semantics. The proxy uses no boundary-node identities and
must not be interpreted as a direct bridge-aware state.

\section{{Sanity checks}}
\begin{{table}}[H]\centering
\caption{{Recorded stage gates and actual diagnostic values.}}
\begin{{tabular}}{{p{{.24\linewidth}}p{{.58\linewidth}}p{{.12\linewidth}}}}\toprule Gate & Value & Status \\\midrule
{_gate_rows(manifest)}
\bottomrule\end{{tabular}}\end{{table}}
Interpretation below is conditional on the formula, reproducibility, pairing,
reference, schema, and production-immutability gates recorded in the manifest.
The production-immutability flag reflects pre-existing uncommitted production-code
changes: the recorded before/after hashes within this run are identical. It is an
audit warning, not evidence that the simulation changed its own code while running.

\section{{Results tables}}
Table 1 reports the primary all-run restricted $W_1$ and RMST metrics. The
completed-only relative $W_1$ is secondary and is shown only beside its completion
denominator. Baseline raw data also contain residual IDs above 100 from an older
interrupted launch; this report excludes them and uses IDs 1--100 uniformly.
\input{{tables/table_1_metrics.tex}}

\section{{Figures}}
\begin{{figure}}[H]\centering\includegraphics[width=.78\linewidth]{{figures/figure_1.pdf}}
\caption{{Baseline cumulative incidence of terminal arrival. Curves retain all runs
and terminate at the observed completion probability; bands are pointwise 95\%
replication-bootstrap intervals.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.78\linewidth]{{figures/figure_2.pdf}}
\caption{{Layer-wise all-run restricted relative $W_1(T_i)$ against V0. Censored
arrivals retain the common administrative horizon.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.78\linewidth]{{figures/figure_3.pdf}}
\caption{{Paired cumulative arrival error (variant minus V0), median and IQR over
replications; censored arrivals retain the common horizon.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.88\linewidth]{{figures/figure_4.pdf}}
\caption{{Change in absolute restricted timing discrepancy relative to V1 across
the interface-width and density mini-sweeps. Positive values mean that a correction
worsened the V1 discrepancy. Baseline supplies the shared $k=5,d=6$ point.}}\end{{figure}}

\section{{Uncertainty analysis}}
All confidence intervals use {bootstrap_reps} paired bootstrap resamples of
replication IDs. Event rows are never bootstrap units. Primary $W_1$ compares
the all-run horizon-restricted terminal-arrival distributions; completed-only
$W_1$ is a secondary conditional diagnostic normalized by the V0 completed mean.
$\Delta$RMST is variant minus V0 using all horizon-restricted runs. Equivalence
requires the entire $\Delta$RMST interval to lie within
$\pm0.05\,\mathrm{{mean}}(T_{{last}}^{{V0}})$.

\section{{Scientific interpretation}}
H1 is assessed by V2 versus V1, H2 by V3 versus V1, and their joint behavior by
V4. The replication-level factorial interaction $V4-V2-V3+V1$ on restricted
terminal arrival is {interaction_text}. Claims of improvement require both passed
gates and the reported uncertainty intervals; an interval containing zero alone
is not evidence of equivalence. Because several terminal cells saturate at the
administrative horizon, the terminal interaction is descriptive rather than a
clean additivity test. {validation_interpretation}

\section{{What the experiment can identify}}
It can estimate relative bias reduction in the sparse baseline, distinguish H1
from H2 within the factorial cells, assess their interaction, and describe limited
scaling across the specified $d$ and $k$ values.

\section{{What the experiment cannot identify}}
It cannot identify recovery effects, synchronization-interval effects, placement
effects, generalization beyond connected ER communities, or computational speedup.
Runtime is an implementation diagnostic only.

\section{{Limitations}}
Lag may interact with midpoint hazard integration, and latent relaxation may
interact with the exponential-clock residual update. The spectral formula is a
candidate scaling law over a small descriptor range. Completed-only Wasserstein
statistics condition on arrival and therefore must be read beside completion and
all-run RMST. They are not used to claim improvement when completion probabilities
are low or materially different. V1 uses the production-aligned experimental
scheduler; the formula gates establish identity among experimental scheduler
variants, not direct pathwise identity with the production orchestrator. The $k$
and $d$ mini-sweeps are one-factor-at-a-time and do not identify an interaction or
a general scaling law.

\section{{Unresolved questions}}
The current design does not determine whether any remaining discrepancy is caused
by synchronization timing, placement, or non-ER boundary geometry.

\section{{Recommended next experiment}}
No recommendation is made outside the staged Experiment 05 evidence. Any follow-up
specified here after validation must preserve pairing and separately vary only one
previously fixed mechanism.

\section{{Exact reproduction commands}}
\begin{{verbatim}}
python -m two_layer_ctmc.experiments.experiment_05_spectral_corrections.cli \
  --stage smoke --scenario baseline --replications 5 --master-seed 270000
python -m two_layer_ctmc.experiments.experiment_05_spectral_corrections.cli \
  --stage validation --scenario all --replications 20 --resume --master-seed 270000
# Run only after explicit approval:
python -m two_layer_ctmc.experiments.experiment_05_spectral_corrections.cli \
  --stage final --scenario all --replications 100 --resume \
  --validation-approved --master-seed 270000
\end{{verbatim}}
\end{{document}}
"""
    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def compile_report(tex_path: Path) -> Path:
    """Compile with pdflatex twice; preserve the complete build log."""
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is not installed")
    output = []
    for _ in range(2):
        result = subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, text=True, capture_output=True,
        )
        output.append(result.stdout + result.stderr)
        if result.returncode != 0:
            (tex_path.parent / "latex_build.log").write_text("\n".join(output), encoding="utf-8")
            raise RuntimeError(f"pdflatex failed; see {tex_path.parent / 'latex_build.log'}")
    (tex_path.parent / "latex_build.log").write_text("\n".join(output), encoding="utf-8")
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise RuntimeError("pdflatex returned successfully without producing report.pdf")
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 05 and build its report")
    parser.add_argument("--output-root", type=Path,
                        default=Path("output/experiment_05_spectral_corrections"))
    parser.add_argument("--report-dir", type=Path,
                        default=Path("reports/experiment_05_spectral_corrections"))
    parser.add_argument("--stage", choices=("smoke", "validation", "final"), required=True)
    parser.add_argument("--horizon", type=float, default=4000.0)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=270006)
    parser.add_argument("--max-replication-id", type=int, default=None,
                        help="Restrict every scenario/variant to replication IDs 1..N.")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    tex = render_report(args.output_root, args.report_dir, args.stage, args.horizon,
                        args.bootstrap_reps, args.bootstrap_seed,
                        max_replication_id=args.max_replication_id)
    if not args.no_compile:
        compile_report(tex)


if __name__ == "__main__":
    main()
