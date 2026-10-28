# Experiment Status Summary — Two-Layer-CTMC

**Purpose of this file:** cross-model handoff. Any model picking up this project
should read this first to understand what the experiments are, what is done,
what is running, and where the research is heading. Written in English for
portability; body reports in `docs/` follow the repo convention (Ukrainian body,
English terms).

**Last updated:** 2026-08-04

**Cross-experiment synthesis:**
[`reports/research_synthesis_uk/report.pdf`](../reports/research_synthesis_uk/report.pdf)
(Ukrainian; evidence, discrepancy regimes, and calibration implications).

---

## 1. What the project is

A two-scale stochastic epidemic simulator on a chain-of-cliques / two-scale
network. We compare an approximate **MicroMacro** simulator against the exact
**Full Micro** Gillespie CTMC (ground truth). The paper's focus is
**approximation error** (when/why MicroMacro is biased), NOT runtime speedup.

- **Full Micro** = complete-network Gillespie = ground truth.
- **MicroMacro** = two-scale surrogate. Macro import hazard
  `λ_ij = β_macro · T · W_ij · (I_i/N_i) · (S_j/N_j)`; uniform placement of imports.
- **Hard constraint (parameters-only):** any error correction may use only
  *network parameters/statistics* the macro layer can know — spectral gap (Fiedler
  `μ2`), degree stats, `W`, community sizes, boundary node **count/degree
  distribution** — NEVER exact edges, node identities, or boundary positions.
  Oracle (exact-edge) variants are used only as an *upper bound* on how much error
  is removable, never as a deployable model.

## 2. Error mechanisms under study

1. **Rate abstraction error** — aggregate macro hazard vs exact gateway hazard
   (dominant; CE-01 ΔRMST = −209).
2. **Import placement error** — uniform target selection vs boundary-aware.
3. **Synchronization / horizon / censoring** artifacts (handled via metrics).

## 3. Metrics (all paired-seed, censoring-aware)

- **ΔRMST** = mean(T_MicroMacro) − mean(T_Micro). Negative = premature arrival.
- **dRMST_norm = ΔRMST / T_end** — the primary cross-scenario metric (fixes the
  horizon confound where T_end varies ~10× across regimes).
- **rel_W1** — relative Wasserstein-1 of arrival-time distributions.
- **Δp (delta_completion)** — completion-probability gap (censoring flag).
- **gap_closure** — fraction of the off-baseline error a correction removes.
  `+1` = fully fixed, `0` = no change, `<0` = correction made it WORSE.
- Bootstrap CIs; Kaplan–Meier where completion differs >5–10%.

## 4. Corrections (parameters-only) — the key deliverable (Track B)

| ID | Name | Formula (parameters-only) | Mode flag | Status |
|----|------|---------------------------|-----------|--------|
| C1 | Spectral alpha | `(I_i/N_i)^{α_i}`, `α_i=clamp(1/μ2,1,6)`, μ2=Fiedler | `alpha_mode="spectral"` | ✓ live (CE-10) |
| C2 | Boundary-availability φ | `φ_i=min(1, boundary_frac_i·(1+μ2_i·I_i/N_i))` | `correction_mode="boundary_phi"` | ✓ implemented |
| C12 | Renormalized rate φ | `φ_i = μ2_i/(μ2_i + r)` | `correction_mode="renorm_phi"` | ✓ implemented |
| C3 | Reach-time onset delay | `τ_reach` gate | `correction_mode="onset_delay"` | ✗ deferred |

All threaded backward-compatibly `MacroEngine ← Orchestrator ← Simulator`;
default `off` preserves legacy behavior. Candidate analysis of 12 strategies
C1–C12 in `docs/error_reduction_research/01_analysis_strategies.md`.

**IMPORTANT BUG HISTORY:** C1 spectral α initially used the 2nd-*largest*
normalized-Laplacian eigenvalue (a bipartiteness artifact, ~1.83) → α<1 →
*amplified* import → made error worse. Fixed to Fiedler value μ2 = 2nd-*smallest*
eigenvalue, `α=clamp(1/μ2,1,6)`, damping-only. Post-fix CE-10 works
(e.g. r1_d12_k5 gap_closure = +0.79).

## 5. Experiment ledger

Grids: `smoke` (tiny sanity), `pilot` (moderate, ~9–12 cells), `full` (36 cells).
Runners in `extrascripts/runners/run_ce*.py` / `run_er01_*.py`; all support
`--workers N` and `--resume` (per-cell CSV, safe to re-run).

| Exp | Purpose | Status | Headline result |
|-----|---------|--------|-----------------|
| CE-01 | Ablation: rate vs placement | ✓ done | Rate abstraction dominates (ΔRMST=−209) |
| CE-03 | Beta-ratio sweep | ✓ done | Sign flip: +4796 (r=0.1) → −1638 (r=2); horizon-confounded (see CE-11) |
| CE-04 | SIR vs SI | ✓ done | Recovery worsens error; 44pp false-positive epidemic at R₀=1 |
| CE-06 | Regime map (params→error) | ✓ done (smoke/pilot/**full** 36×60) | Error grows with r, falls with d/μ2; sign flip ~r0.5,d≥6 |
| CE-07 | Calibration floor (struct vs calib error) | ✓ smoke/pilot; **full queued after ER-01** | c* varies 0.5–1.2; structural_fraction 0.22–0.78; pilot is in-sample |
| CE-08 | Negative control (dense clique chain) | ✓ **full 200 done** | ΔRMST=8.2, CI [-39.0,54.4]; no systematic offset detected |
| CE-09 | Boundary/gateway diagnostic | ✓ **full 18×40 done** | Static boundary placement improves only 1/6 regimes; design is not mode-paired |
| CE-10 | Correction ladder (C1/C2/C12) | ✓ smoke/pilot/**full** 12 | Post-fix spectral closes gap on moderate d |
| CE-11 | Horizon-controlled CE-03 re-analysis | ✓ done | Robust sign flip on common horizon |
| CE-12 | Statistical power / N calibration | ✗ not started | — (100 reps under-powered at k=5) |
| ER-01 | Multi-topology validation of C2/C12 | ✓ **smoke** 12; **pilot running (54 cells)** | **C2/C12 good on random_connected & complete, REGRESS on clique_leaves** |

## 6. THE current open question (where it's heading)

ER-01 smoke (n=8 reps, r∈{1,2}, d=6) across macro∈{chain,complete} ×
micro∈{complete, random_connected, clique_leaves}:

- `random_connected` (target sparse case): **C2 gap_closure +0.62…+0.75** ✓
- `complete` (well-mixed identity case): off_dRMST_norm≈0, C2 ≈ +0.63 (no harmful
  regression; near-identity as required) ✓
- `clique_leaves`: **C2 gap_closure −1.5, −2.4, −18.2** ✗ — correction badly
  overshoots. Cause: `boundary_within_deg_mean=99` (fully-connected within
  community) makes the boundary-fraction heuristic misfire.

**Conclusion:** C2/C12 are NOT yet universally safe. The minimax target
(gap_closure ≥ 0.5 on worst-case topology, ≈0 on identity) is met on 2 of 3
micro-topologies but fails on `clique_leaves`. Next steps must (a) get CIs via
ER-01 **pilot** (more reps, more r/d), and (b) isolate the boundary-degree
predictor via CE-09 to understand/repair the `clique_leaves` failure — possibly
gating φ by boundary_within_degree so it self-disables when the community is
internally dense.

## 7. What is running now (overnight, `extrascripts/runners/run_overnight_pending.sh`)

Sequential stages (each internally multi-threaded via `--workers`; sequential to
avoid CPU oversubscription on the 16-core machine), cheap+important first,
expensive last, all `--resume`:

1. **CE-09** smoke → pilot → full — completed successfully on 2026-08-04.
2. **CE-08** full — completed successfully on 2026-08-04.
3. **ER-01** pilot — currently running; do not interpret partial outputs.
4. **CE-07** full — queued after ER-01; resumable.

Note: LLM subagents do not accelerate the simulations themselves (CPU-bound
Gillespie); parallelism that matters is the `--workers` thread pool inside each
runner. Subagents are useful only for *analysis/writing* after data lands.

## 8. How to operate

- Launch overnight: `wsl bash -c "cd <repo> && tmux new-session -d -s overnight 'bash extrascripts/runners/run_overnight_pending.sh'"`
  (tmux survives terminal/VS Code closure; an attached terminal does NOT — it
  stalls into stopped 'T' state on disconnect).
- Check progress: `wsl tmux capture-pane -t overnight -p | tail -30` or read
  `output/logs/overnight_<STAMP>/master.log`.
- Reattach: `wsl tmux attach -t overnight` (detach with Ctrl+B then D).
- Python: `.venv/bin/python` via WSL. Never inline `python -c "..."` from
  PowerShell (parser mangles `from`/`in`); write a temp `.py`/`.sh` and run it.
- Reports: LaTeX in `reports/` = English only (pdflatex, no Cyrillic) OR lualatex
  for Ukrainian; markdown summaries in `docs/` = Ukrainian body + English terms.

## 9. Outputs on disk

- `output/claude_experiments/ce0X_*_{smoke,pilot,full}/` — `summary.csv`,
  `cells/arrivals_*.csv`, `manifest.json`.
- `output/error_reduction_research/er01_multitopo_smoke_smoke/` — ER-01 smoke.
- `docs/error_reduction_research/` — analysis (01), ranking (01b), execution log
  (02), validation matrix (04), evidence (05), RESULTS.md, BRIEF.md.
- `output/logs/overnight_*/` — run logs.

## 10. Interpretation reports

- `reports/claude_experiments/ce06_regime_map/` — full regime-map implications.
- `reports/claude_experiments/ce07_calibration_floor/` — pilot development report; not final.
- `reports/claude_experiments/ce08_negative_control/` — dense negative control.
- `reports/claude_experiments/ce09_boundary_isolation/` — boundary diagnostic plus design audit.
- `reports/claude_experiments/ce10_correction_ladder/` — spectral correction regime analysis.
- `docs/PROJECT_ACTIVITY_LOG.md` — concise cross-agent activity record.
