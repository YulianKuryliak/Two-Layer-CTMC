# Project: Two-Layer-CTMC

Research paper on a two-scale stochastic epidemic simulator (MicroMacro vs full Gillespie CTMC).
Paper: `paper/text.tex` (pdflatex, ~50 pages, English).
Experiment reports: `reports/claude_experiments/` (TeX+PDF, English).
Ukrainian summaries: `docs/` (Markdown).

## Key Files

- **Paper**: `paper/text.tex`, `paper/references.bib`, `paper/Images/`
- **Simulators**: `two_layer_ctmc/simulators/micro.py`, `micromacro.py`
- **Network**: `two_layer_ctmc/network.py` — `generate_two_scale_network()`
- **Experiments**: `extrascripts/run_ce*.py` → `output/claude_experiments/`
- **Reports (TeX)**: `reports/claude_experiments/` — shared preamble at `reports/claude_experiments/preamble.tex`
- **Reports (Ukrainian)**: `docs/ce*/report.md`
- **Experiment plan**: `docs/TODO_experiments.md`

## Core Concepts

- **MicroMacro**: two-scale simulator with macro import hazard `λ_ij = β_out * ī_i(t)` and uniform placement
- **Full Micro**: full Gillespie CTMC on complete network (ground truth)
- **ΔRMST**: mean(T_MicroMacro) − mean(T_Micro); negative = premature arrival
- **Rate abstraction error**: aggregate macro hazard vs exact gateway hazard
- **Import placement error**: uniform target selection vs boundary-node selection
- **Key topology**: chain-of-cliques, L=10, n=100, d=6, k=5, β=0.003

## Completed Experiments

- **CE-01** (ablation): rate abstraction dominates (ΔRMST=−209); placement significant only under exact hazard
- **CE-04** (SIR): recovery worsens error monotonically; 44pp false-positive epidemic rate at R₀=1
- **CE-03** (beta ratio): complete — sign flip in ΔRMST: +4796 at r=0.1 (late), −1355 at r=1.0, −1638 at r=2.0 (premature); crossover r*≈0.5–0.8; micro_completion=0.01 at r=5 (99% false-positive)

## Technical Rules

- Python: `.venv/bin/python` — always use WSL: `wsl bash -c "cd /mnt/c/..."`
- LaTeX: `pdflatex` only — NO Cyrillic (texlive-lang-cyrillic not installed)
- Reports compile from their own folder: `cd reports/claude_experiments/ce0X_*/`
- Do not rewrite paper sections not explicitly requested
- Do not rerun simulations unless asked — prefer saved data in `output/`

## Language Rules for Reports

- **Body text** → Ukrainian: аналіз, висновки, опис механізмів
- **Terms/metrics/abbreviations** → English: ΔRMST, bootstrap CI, rel-W₁, rate abstraction, MicroMacro, Gillespie, SI/SIR, beta, gamma, T_end
- **File/variable names** → always English: `beta_out`, `r=0.1`, `summary.csv`
- **LaTeX reports** → English only (Cyrillic not supported in pdflatex here)

## Model Budget Hints

- **File/log search, LaTeX label check, CSV preview** → switch to cheapest model (mini/flash)
- **Python scripting, report drafting, debug** → default model (Sonnet 4.6) is fine
- **Paper section rewriting, experiment design, math review, critical review** → use o3 or strongest available
- Attach only the relevant file section, not the whole paper
- For data analysis: pass metric values inline, not raw CSV files
