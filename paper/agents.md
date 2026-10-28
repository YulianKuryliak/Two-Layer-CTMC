# Agent Instructions For Paper Work

Use this file as local guidance when editing the paper or benchmark results.

## Main Files

- Main article text: `paper/text.tex`.
- Bibliography: `paper/references.bib`.
- Images used in the article: `paper/Images/`.
- Project plots and generated images: `plots/`.
- Chain-of-cliques images: `paper/Images/chain_cliques/`.
- Current notes/status files: `paper/state.md`, `paper/changes.md` when present.

## General Rules

- Work economically. Do not scan the whole repository unless needed.
- Save usage limits deliberately, but do not be so strict that article quality, correctness, or reproducibility suffers.
- Prefer `rg` / `rg --files` for searching.
- Do not refactor unrelated code.
- Do not rewrite old sections unless explicitly asked.
- Preserve the old Chain-of-cliques benchmark section for comparison.
- For new work, update only the relevant subsection, figure, script, or data artifact.
- Do not add dependencies unless clearly necessary.
- If a figure already exists and only needs regeneration, regenerate from saved data instead of rerunning simulations.
- If a simulation must be rerun, save full outputs so individual runs can be inspected later.

## Resource And Model Budget

- Use the smallest model and reasoning level that is adequate for the subtask.
- For planning scientific changes, benchmark design, or paper-level interpretation, prefer a stronger planning model such as GPT-5.5 medium/high when available.
- For simple searches, file listing, formatting checks, small CSV summaries, or mechanical edits, use cheaper models or lower reasoning when available, such as GPT-5.4 mini / GPT-5.4-class models.
- Use stronger models only when the task needs scientific judgment, mathematical interpretation, nontrivial code changes, or careful paper writing.
- If using subagents, assign model strength and reasoning depth by task difficulty:
  - lightweight model for locating files, summarizing logs, checking paths, or producing small mechanical diffs;
  - medium/high model for experiment design, interpreting simulator discrepancies, or rewriting scientific prose.
- Do not spawn subagents for trivial single-file edits.
- Keep prompts to subagents focused and pass only the files/results they need.
- Prefer reusing known context from this file and recent results over re-reading large files.

## Chain-of-Cliques Benchmark State

The paper currently contains:

- The original Chain-of-cliques benchmark.
- An extended calibrated Chain-of-cliques benchmark.

The calibrated benchmark uses:

- `L=10`, `n=100`, baseline `k=5`.
- `beta=0.001`, `gamma=0`, `tau_micro=1`, `macro_T=1`.
- Paired Micro and MicroMacro runs.
- Full trajectory CSVs saved per run.
- Median/IQR trajectory plots.
- Arrival-time CDFs with bootstrap confidence bands.
- Paired cumulative arrival errors `T_i^MicroMacro - T_i^Micro`.

Important interpretation:

- `Delta T_i = T_{i+1} - T_i` is local delay between neighboring communities.
- `T_i^MicroMacro - T_i^Micro` is paired cumulative arrival error from simulation start.
- For comparing simulators, cumulative arrival error and `W1(T_i)` are usually more informative than local `Delta T_i`.
- Completion probability must be reported whenever terminal arrival metrics are computed.
- Arrival times after `T_end` are censored and should not be treated as completed arrivals.
- Within one experiment, compare results that share the same experimental meaning: same horizon, same parameter family, same censoring rule, and same metric definition.
- Do not compare incomparable settings as if they were the same experiment. For example, do not present `k=1,T_end=5000` as directly equivalent to `k=5,10,20,T_end=2500` without stating that the observation window differs.
- Different horizons can be used as diagnostics, but they must be labeled as diagnostics or explained explicitly.

## Relevant Scripts

- Main benchmark script:
  `extrascripts/chain_cliques_arrival_comparison.py`

- Calibrated summary/plot script:
  `extrascripts/chain_cliques_calibrated_summary.py`

- Older v2 summary script:
  `extrascripts/chain_cliques_v2_summary.py`

Prefer using existing scripts instead of writing new one-off logic. If a one-off is needed, keep it small and preserve generated data.

## Key Output Locations

Simulation summaries:

- `data/chain_cliques_arrival_comparison/`

Full per-run trajectories:

- `data/Simulations_ChainCliques_v2/`

Plots:

- `plots/chain_cliques_calibrated_benchmark/`
- `plots/chain_cliques_v2_benchmark/`
- `plots/k_*_n_*_inter_*_macro_chain_micro_complete_p0p031/`

Images used by the article:

- `paper/Images/chain_cliques/`

Important current calibrated images include:

- `calibrated_total_infected_iqr.png`
- `calibrated_per_community_pair.png`
- `calibrated_tlast_cdf.png`
- `calibrated_paired_arrival_error.png`
- `calibrated_error_growth.png`
- `calibrated_interface_summary.png`
- `calibrated_interface_dynamics_by_k.png`
- `calibrated_chain_length_summary.png`
- `calibrated_community_size_summary.png`

## Simulation Provenance

Recent calibrated scenarios include:

- Baseline: `L=10,n=100,k=5,T_end=2500,n_runs=100`.
- Interface sweep: `k=1,5,10,20`; `k=1` also has a long-horizon `T_end=5000` diagnostic used in the interface dynamics figure.
- Length sweep: `L=5,10,15`.
- Size sweep: `n=50,100,200`.

Do not mix horizons without explaining it. In particular:

- `k=1,T_end=5000` is useful for showing the narrow-interface tail.
- It is not the same observation window as `k=5,10,20,T_end=2500`.

## Figure Rules

- For infection dynamics, prefer median line plus interquartile band over single representative runs.
- For CDF uncertainty, use bootstrap confidence bands, not trajectory IQR.
- For per-community curves, one paired completed run is acceptable, but label it as illustrative.
- Use figure captions to state what the line and shaded band mean.
- Avoid figures that only decorate the section. Each figure should answer a specific comparison or diagnostic question.
- For interface comparison, show both:
  - total infected median/IQR curves;
  - paired cumulative arrival errors.

## Verification

For script syntax:

```bash
python -m py_compile extrascripts/chain_cliques_arrival_comparison.py extrascripts/chain_cliques_calibrated_summary.py
```

For paper build:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error text.tex
```

LaTeX may report overfull hbox warnings. These are not necessarily blockers if the PDF builds.

## Reporting Back

When reporting changes to the user, keep it short:

1. What changed.
2. Which files changed.
3. How to verify.
4. Remaining risks.
