#!/usr/bin/env python3
"""Export the saved leaf-benchmark results as editable PGFPlots tables.

This is a conversion step only: it reads the completed benchmark and saved
summary table, without running either simulator or recomputing any bootstrap
interval.  The companion ``.tex`` snippets contain just TikZ/PGFPlots code;
they are intended to be input from the SMPT manuscript.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "SMPT paper" / "figures_pgfplots"
CLIQUE_INPUT = ROOT / "output" / "main_clean_benchmarks_independent_rng" / "benchmark_observables.csv"
CLIQUE_METRICS = ROOT / "output" / "analysis" / "clique_leaf" / "summary_metrics.json"
ASPL_SUMMARY = (
    ROOT
    / "output"
    / "analysis"
    / "n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial"
    / "figures"
    / "aspl_export_summary.csv"
)

BENCHMARK = "clique_leaf_n100_k2"
FULL = "Full Micro"
TWOSCALE = "Production MicroMacro + independent RNG"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(path: Path, frame: pd.DataFrame) -> None:
    """Write a whitespace-delimited PGFPlots table with enough precision."""
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"Non-finite numeric value in {path.name}")
    frame.to_csv(path, sep=" ", index=False, float_format="%.17g")
    # The data files are part of the figure source.  Check that their decimal
    # representation can be read back without changing any plotted number.
    restored = pd.read_csv(path, sep=r"\s+")
    if list(restored.columns) != list(frame.columns) or len(restored) != len(frame):
        raise ValueError(f"Round-trip structure check failed for {path.name}")
    for column in numeric.columns:
        original = frame[column].to_numpy(float)
        recovered = restored[column].to_numpy(float)
        tolerance = 8 * np.finfo(float).eps * max(1.0, float(np.abs(original).max()))
        if not np.allclose(original, recovered, rtol=0.0, atol=tolerance):
            raise ValueError(f"Round-trip numeric check failed for {path.name}:{column}")


def ecdf(values: np.ndarray, xmin: float) -> pd.DataFrame:
    values = np.sort(np.asarray(values, dtype=float))
    # The explicit zero-probability point lets PGFPlots' const plot reproduce
    # Matplotlib's ``step(..., where='post')`` lower tail.  The anchor must be
    # at or left of the first observation, so all 10,000 ECDF observations and
    # their ranks remain in the editable source; the axis clips off-panel data.
    ranks = np.arange(1, len(values) + 1) / len(values)
    anchor = min(xmin, float(values[0]))
    return pd.DataFrame({
        "x": np.r_[anchor, values],
        "y": np.r_[0.0, ranks],
    })


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return float(centre - half_width), float(centre + half_width)


def export_clique_leaf(generated: list[Path]) -> dict[str, object]:
    data = pd.read_csv(CLIQUE_INPUT)
    data = data.loc[data["benchmark"].eq(BENCHMARK)].copy()
    if len(data) != 20_000:
        raise ValueError(f"Expected 20,000 {BENCHMARK} rows, found {len(data)}")

    by_model: dict[str, pd.DataFrame] = {}
    for model in (FULL, TWOSCALE):
        subset = data.loc[data["simulator"].eq(model)].sort_values("replication_id")
        if len(subset) != 10_000 or subset["replication_id"].nunique() != 10_000:
            raise ValueError(f"Expected 10,000 unique replications for {model}")
        by_model[model] = subset

    if not np.array_equal(
        by_model[FULL]["replication_id"].to_numpy(), by_model[TWOSCALE]["replication_id"].to_numpy()
    ):
        raise ValueError("Full and TwoScale clique--leaf rows are not paired")

    for stem, column, xmin in (("export", "total_export_time", 0.0), ("signed_delay", "bridge_to_export_delay", -20.0)):
        for label, model in (("full", FULL), ("twoscale", TWOSCALE)):
            path = OUT / f"leaf_clique_{stem}_{label}.dat"
            write_table(path, ecdf(by_model[model][column].to_numpy(float), xmin))
            generated.append(path)

    rows = []
    for x, (label, model) in enumerate((("Full", FULL), ("TwoScale", TWOSCALE)), start=1):
        values = by_model[model]["bridge_to_export_delay"].to_numpy(float)
        count, total = int((values < 0).sum()), len(values)
        low, high = wilson_interval(count, total)
        rows.append(
            {
                "x": x,
                "rate_pct": 100 * count / total,
                "low_pct": 100 * low,
                "high_pct": 100 * high,
                "count": count,
                "total": total,
                "label": label,
            }
        )
    bypass = OUT / "leaf_clique_bypass.dat"
    write_table(bypass, pd.DataFrame(rows))
    generated.append(bypass)

    metrics = json.loads(CLIQUE_METRICS.read_text(encoding="utf-8"))
    return {
        "replications_per_model": 10_000,
        "w1_export": metrics["W1_export"],
        "ks_export": metrics["KS_export"],
        "bypass_counts": {row["label"]: int(row["count"]) for row in rows},
    }


def export_source_sweep(generated: list[Path]) -> dict[str, object]:
    summary = pd.read_csv(ASPL_SUMMARY)
    expected = {(float(aspl), int(k)) for aspl in np.arange(1.0, 6.0 + 0.001, 0.25) for k in (1, 5)}
    observed = {(float(row.target_aspl), int(row.k)) for row in summary.itertuples()}
    if observed != expected or len(summary) != len(expected) or summary.duplicated(["target_aspl", "k"]).any():
        raise ValueError("The saved ASPL summary does not contain exactly ASPL 1--6 by 0.25 and k=1,5")
    if not (summary["paired_runs"].eq(400) & summary["full_completion"].eq(100) & summary["mm_completion"].eq(100)).all():
        raise ValueError("The saved ASPL summary has incomplete or censored cells")

    columns = [
        "target_aspl",
        "full_export_mean", "full_export_mean_ci_low", "full_export_mean_ci_high",
        "mm_export_mean", "mm_export_mean_ci_low", "mm_export_mean_ci_high",
        "relative_w1_pct", "relative_w1_pct_ci_low", "relative_w1_pct_ci_high",
        "mm_bypass_pct", "mm_bypass_pct_ci_low", "mm_bypass_pct_ci_high",
    ]
    renamed = {
        "target_aspl": "aspl",
        "full_export_mean": "full_mean", "full_export_mean_ci_low": "full_low", "full_export_mean_ci_high": "full_high",
        "mm_export_mean": "twoscale_mean", "mm_export_mean_ci_low": "twoscale_low", "mm_export_mean_ci_high": "twoscale_high",
        "relative_w1_pct": "w1", "relative_w1_pct_ci_low": "w1_low", "relative_w1_pct_ci_high": "w1_high",
        "mm_bypass_pct": "bypass", "mm_bypass_pct_ci_low": "bypass_low", "mm_bypass_pct_ci_high": "bypass_high",
    }
    maxima = {
        "time": float(summary[["full_export_mean_ci_high", "mm_export_mean_ci_high"]].to_numpy(float).max()),
        "w1": float(summary["relative_w1_pct"].max()),
        "bypass": float(summary[["mm_bypass_pct", "mm_bypass_pct_ci_high"]].to_numpy(float).max()),
    }
    ymax = {name: 1.05 * value if value > 0 else 1.0 for name, value in maxima.items()}
    result: dict[str, object] = {
        "cells_per_k": 21,
        "paired_runs_per_cell": 400,
        "shared_ymax": ymax,
    }
    for k in (1, 5):
        table = summary.loc[summary["k"].eq(k), columns].sort_values("target_aspl").rename(columns=renamed)
        if len(table) != 21 or not table["aspl"].is_unique:
            raise ValueError(f"Expected 21 distinct ASPL values for k={k}")
        path = OUT / f"leaf_source_sweep_k{k}.dat"
        write_table(path, table)
        generated.append(path)
        # Match the original plotting script: intervals with identical bounds
        # are not drawn, because they do not quantify uncertainty.
        visible = table.loc[table["bypass_high"] > table["bypass_low"]]
        ci_path = OUT / f"leaf_source_sweep_k{k}_bypass_ci.dat"
        write_table(ci_path, visible[["aspl", "bypass", "bypass_low", "bypass_high"]])
        generated.append(ci_path)
    return result


CLIQUE_TEX = r"""% Editable PGFPlots version of clique_leaf_endpoint_and_mechanism.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=3 by 1, horizontal sep=1.55cm},
  paper axis, width=.24\linewidth, height=3.7cm, scale only axis,
  legend style={font=\scriptsize, draw=none},
]
\nextgroupplot[
  title={\textbf{a} Export times},
  xlabel={total export time}, ylabel={empirical CDF},
  xmin=0, xmax=600, ymin=0, ymax=1.01,
  legend style={at={(.98,.36)}, anchor=south east},
]
  \addplot+[color=FullBlue, solid, no markers, const plot, line width=.8pt]
    table[x=x,y=y]{figures_pgfplots/leaf_clique_export_full.dat};
  \addlegendentry{Full}
  \addplot+[color=ScaleOrange, dashed, no markers, const plot, line width=.8pt]
    table[x=x,y=y]{figures_pgfplots/leaf_clique_export_twoscale.dat};
  \addlegendentry{TwoScale}
  \node[anchor=south east, align=right, fill=white, draw=black!20, inner sep=2pt, font=\scriptsize]
    at (rel axis cs:.97,.07) {$W_1=3.249$\\$D_{\rm KS}=0.0142$};
\nextgroupplot[
  title={\textbf{b} Export minus bridge},
  xlabel={$T_{\mathrm{export}}-T_B$}, ylabel={empirical CDF},
  xmin=-20, xmax=100, ymin=0, ymax=.55,
]
  \path[fill=black!8] (axis cs:-20,0) rectangle (axis cs:0,.55);
  \addplot+[black!75, no markers, line width=.35pt] coordinates {(0,0) (0,.55)};
  \addplot+[color=FullBlue, solid, no markers, const plot, line width=.8pt]
    table[x=x,y=y]{figures_pgfplots/leaf_clique_signed_delay_full.dat};
  \addplot+[color=ScaleOrange, dashed, no markers, const plot, line width=.8pt]
    table[x=x,y=y]{figures_pgfplots/leaf_clique_signed_delay_twoscale.dat};
  \node[anchor=north west, font=\scriptsize, text=black!65] at (axis cs:-18,.51) {export before bridge};
\nextgroupplot[
  title={\textbf{c} Bridge bypass},
  ylabel={bridge-bypass runs (\%)},
  xmin=.5, xmax=2.5, ymin=-.03, ymax=1.35, clip=false,
  xtick={1,2}, xticklabels={Full,TwoScale},
]
  \addplot+[
    only marks, mark=*, mark size=1.7pt, color=black!80,
    error bars/.cd, y dir=both, y explicit,
  ] table[x=x,y=rate_pct,
      y error minus expr=\thisrow{rate_pct}-\thisrow{low_pct},
      y error plus expr=\thisrow{high_pct}-\thisrow{rate_pct}]
      {figures_pgfplots/leaf_clique_bypass.dat};
  \node[anchor=south, align=center, font=\scriptsize] at (axis cs:1,.14) {0/10000\\(0.00\%)};
  \node[anchor=south, align=center, font=\scriptsize] at (axis cs:2,1.18) {98/10000\\(0.98\%)};
\end{groupplot}
\end{tikzpicture}
"""


SOURCE_SWEEP_TEX = r"""% Editable PGFPlots version of fig_aspl_export_fidelity.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=3 by 2, horizontal sep=1.55cm, vertical sep=1.35cm},
  paper axis, width=.24\linewidth, height=3.7cm, scale only axis,
  xmin=.85, xmax=6.15, xtick={1,2,...,6},
]
\nextgroupplot[
  title={$k=1$: export time}, ylabel={mean export time},
  xticklabels=\empty, ymin=0, ymax=__YMAX_TIME__,
  legend style={font=\scriptsize, draw=none, at={(.98,.98)}, anchor=north east},
]
  \addplot+[color=FullBlue, mark=*, mark size=1.1pt, mark options={fill=FullBlue,draw=FullBlue}, line width=.6pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=full_mean,y error minus expr=\thisrow{full_mean}-\thisrow{full_low},y error plus expr=\thisrow{full_high}-\thisrow{full_mean}]
    {figures_pgfplots/leaf_source_sweep_k1.dat};
  \addlegendentry{Full}
  \addplot+[color=ScaleOrange, mark=square*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=twoscale_mean,y error minus expr=\thisrow{twoscale_mean}-\thisrow{twoscale_low},y error plus expr=\thisrow{twoscale_high}-\thisrow{twoscale_mean}]
    {figures_pgfplots/leaf_source_sweep_k1.dat};
  \addlegendentry{TwoScale}
\nextgroupplot[title={$k=1$: $W_1$ error}, ylabel={$W_1/\overline T_{\rm Full}$ (\%)}, xticklabels=\empty, ymin=0, ymax=__YMAX_W1__]
  \addplot+[color=ScaleOrange, mark=*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt]
    table[x=aspl,y=w1]{figures_pgfplots/leaf_source_sweep_k1.dat};
\nextgroupplot[title={$k=1$: bridge bypass}, ylabel={export before any bridge (\%)}, xticklabels=\empty, ymin=0, ymax=__YMAX_BYPASS__]
  \addplot+[color=ScaleOrange, mark=*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt]
    table[x=aspl,y=bypass]{figures_pgfplots/leaf_source_sweep_k1.dat};
  \addplot+[color=ScaleOrange, mark=none, line width=.45pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=bypass,y error minus expr=\thisrow{bypass}-\thisrow{bypass_low},y error plus expr=\thisrow{bypass_high}-\thisrow{bypass}]
    {figures_pgfplots/leaf_source_sweep_k1_bypass_ci.dat};
\nextgroupplot[title={$k=5$: export time}, xlabel={target ASPL}, ylabel={mean export time}, ymin=0, ymax=__YMAX_TIME__]
  \addplot+[color=FullBlue, mark=*, mark size=1.1pt, mark options={fill=FullBlue,draw=FullBlue}, line width=.6pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=full_mean,y error minus expr=\thisrow{full_mean}-\thisrow{full_low},y error plus expr=\thisrow{full_high}-\thisrow{full_mean}]
    {figures_pgfplots/leaf_source_sweep_k5.dat};
  \addplot+[color=ScaleOrange, mark=square*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=twoscale_mean,y error minus expr=\thisrow{twoscale_mean}-\thisrow{twoscale_low},y error plus expr=\thisrow{twoscale_high}-\thisrow{twoscale_mean}]
    {figures_pgfplots/leaf_source_sweep_k5.dat};
\nextgroupplot[title={$k=5$: $W_1$ error}, xlabel={target ASPL}, ylabel={$W_1/\overline T_{\rm Full}$ (\%)}, ymin=0, ymax=__YMAX_W1__]
  \addplot+[color=ScaleOrange, mark=*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt]
    table[x=aspl,y=w1]{figures_pgfplots/leaf_source_sweep_k5.dat};
\nextgroupplot[title={$k=5$: bridge bypass}, xlabel={target ASPL}, ylabel={export before any bridge (\%)}, ymin=0, ymax=__YMAX_BYPASS__]
  \addplot+[color=ScaleOrange, mark=*, mark size=1.1pt, mark options={fill=ScaleOrange,draw=ScaleOrange}, line width=.6pt]
    table[x=aspl,y=bypass]{figures_pgfplots/leaf_source_sweep_k5.dat};
  \addplot+[color=ScaleOrange, mark=none, line width=.45pt,
    error bars/.cd, y dir=both, y explicit]
    table[x=aspl,y=bypass,y error minus expr=\thisrow{bypass}-\thisrow{bypass_low},y error plus expr=\thisrow{bypass_high}-\thisrow{bypass}]
    {figures_pgfplots/leaf_source_sweep_k5_bypass_ci.dat};
\end{groupplot}
\end{tikzpicture}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    clique = export_clique_leaf(generated)
    sweep = export_source_sweep(generated)
    source_sweep_tex = SOURCE_SWEEP_TEX
    for name, value in sweep["shared_ymax"].items():
        source_sweep_tex = source_sweep_tex.replace(f"__YMAX_{name.upper()}__", f"{value:.12g}")
    snippets = {OUT / "clique_leaf.tex": CLIQUE_TEX, OUT / "source_sweep.tex": source_sweep_tex}
    for path, content in snippets.items():
        path.write_text(content, encoding="utf-8")
        generated.append(path)
    manifest = {
        "purpose": "Editable PGFPlots conversion from saved leaf-benchmark outputs; no simulation or statistical recomputation.",
        "sources": {
            str(CLIQUE_INPUT.relative_to(ROOT)): sha256(CLIQUE_INPUT),
            str(CLIQUE_METRICS.relative_to(ROOT)): sha256(CLIQUE_METRICS),
            str(ASPL_SUMMARY.relative_to(ROOT)): sha256(ASPL_SUMMARY),
        },
        "clique_leaf": clique,
        "source_sweep": sweep,
        "generated": {path.name: sha256(path) for path in generated},
    }
    (OUT / "leaf_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(generated)} editable-figure files to {OUT}")


if __name__ == "__main__":
    main()
