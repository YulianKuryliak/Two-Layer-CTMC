# Task Brief: Reduce the MicroMacro Approximation Error (multi-topology)

Persistent research pipeline, **separate from the CE-06..CE-12 experiment program**.
Two roles run in sequence and **record their results as files** in this folder:

1. **Analyst** (`01_analysis_strategies.md`): survey broadly — NOT limited to our
   current ideas — every plausible way to reduce the error. Rank candidates.
2. **Executor** (`02_execution_log.md` + code): implement the top-ranked
   parameters-only candidates and validate them across MULTIPLE topologies.

The overriding requirement: a correction must **generalize across different
network topologies**, not be tuned to chain-of-cliques.

---

## 1. System under study

- **Full Micro** = Gillespie CTMC on the complete two-scale network = ground truth.
- **MicroMacro** = two-scale approximation. Intra-community spread is simulated
  exactly (micro Gillespie per community); cross-community infection is replaced
  by an aggregate **macro import hazard**:

  `lambda_{i->j} = beta_macro * T * W_ij * infectivity_i * susceptibility_j`

  with `infectivity_i = I_i / N_i`, `susceptibility_j = S_j / N_j`.
  On a macro import event, the target infected node is placed **uniformly at
  random** among susceptibles of community j.

- **What the macro layer may see (HARD CONSTRAINT):** only *network parameters*
  — the inter-community weight matrix `W`, community sizes `N_i`, and
  **aggregate structural statistics of each community** (degree distribution,
  normalized-Laplacian spectrum / Fiedler value `mu2`, spectral radius,
  boundary-node COUNT, etc.). It may **NEVER** use exact node identities,
  positions, which specific nodes are infected, or which specific nodes carry
  cross-community edges. Oracle variants that peek at exact structure are allowed
  ONLY as upper-bound diagnostics and can never be the deployed solution.

## 2. Error mechanisms (what we are fighting)

1. **Rate abstraction error (dominant).** The aggregate `I_i/N_i` assumes every
   infected node is instantly available to export. In reality infection must
   first reach the community's **boundary/gateway nodes** before it can cross;
   sparse/poorly-mixed communities delay this. Result: MicroMacro exports **too
   early** -> premature arrival -> negative dRMST.
2. **Import placement error.** Uniform placement of the imported infection vs the
   true boundary-localized entry point. Matters mainly under exact hazard.
3. **Synchronization / numerical error.** Macro tau-step vs continuous micro time.

## 3. Metrics (censoring-aware, paired)

- **dRMST** = mean(T_MicroMacro) - mean(T_Micro), restricted-mean-survival-time
  style, censoring-aware. Primary. Negative = premature.
- **dRMST_norm = dRMST / T_end** — horizon-commensurate, use to compare ACROSS
  scenarios/topologies (different T_end).
- **rel_W1** — relative Wasserstein-1 between arrival-time distributions.
- **delta_completion (Δp)** — completion-probability gap (false epidemic rate).
- **KM survival / arrival CDF** for shape.
- Always use **paired seeds** (same network_seed and run_seed for Micro vs each
  MicroMacro variant). Never compare across different horizons/censoring rules.

## 4. Multi-topology generalization requirement (CRITICAL)

Any proposed correction MUST be evaluated on a topology matrix, not one point:

- **macro_graph_type** in {`chain`, `complete`, `star`}
- **micro_graph_type** in {`complete`, `random_connected`, `clique_leaves`}
  (and `chain` micro as a hard sparse stress case)
- vary internal degree `d` (sparse..dense) and import ratio `r = beta_out/beta_in`.

A candidate that reduces error on one topology but **increases** it on another is
**rejected** (or must be gated by a parameters-only topology descriptor).
Report a **per-topology table** of dRMST_norm and rel_W1: baseline (off) vs
corrected, plus `gap_closure = (|off| - |corrected|) / |off|`. The winning
correction is the one with the best **worst-case** gap_closure across topologies
(minimax), not the best average.

## 5. Current state (one candidate already in code)

- Spectral damping exponent: `infectivity_i = (I_i/N_i)^alpha_i`,
  `alpha_i = clamp(1/mu2, 1, 6)`, `mu2` = Fiedler value (2nd-smallest eigenvalue
  of the community normalized Laplacian). Opt-in via `alpha_mode="spectral"`.
  Files: `two_layer_ctmc/engines/macro_engine.py`,
  `two_layer_ctmc/orchestrators/micromacro.py`, `two_layer_ctmc/network.py`.
- Smoke (chain-of-cliques, r=2, d=4): gap_closure ~ +0.56, rel_W1 0.68->0.30,
  but OVERSHOOTS (dRMST_norm -0.41 -> +0.18) at extreme sparsity (alpha hit cap).
- This is only ONE mechanism. The Analyst must go broader.

## 6. Reproducibility notes

- Python: `.venv/bin/python` via WSL:
  `wsl bash -c "cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC && .venv/bin/python ..."`
- Network builder: `two_layer_ctmc.network.generate_two_scale_network(
  n_communities, community_size, inter_links, seed, macro_graph_type,
  micro_graph_type, edge_prob, leaf_count, leaf_degree, star_leaf_attachment)`.
- Simulators: `two_layer_ctmc/simulators/micro.py`, `.../micromacro.py`
  (`MicroMacroSimulator(..., alpha_mode=...)`).
- Reference runner/harness (paired seeds, adaptive T_end, metrics):
  `extrascripts/runners/run_ce06_regime_map.py` and `extrascripts/runners/run_ce10_correction_ladder.py`.
- SI dynamics = `model="SIR"` with `gamma=0` (the literal string "SI" is rejected).
- Keep new sims to **smoke grids** (few seeds, few cells) unless the user approves
  a large run. Do not disturb currently-running CE-06 / CE-10 jobs (share CPU).

## 7. Deliverable format

- `01_analysis_strategies.md` (Analyst): ranked candidate corrections. For EACH:
  name, mechanism, which error source it targets, parameters-only feasibility
  (yes/no + which descriptors), expected multi-topology generality, implementation
  sketch, validation plan, risks/failure modes. End with a ranked shortlist and a
  recommended execution order.
- `02_execution_log.md` (Executor): for each implemented candidate — code location,
  the multi-topology smoke results table (per-topology dRMST_norm/rel_W1/gap_closure),
  what worked, what generalized, what did not, and next steps. Keep a running
  `RESULTS.md` summary table if useful.
- Language: body Ukrainian, terms/metrics/filenames English (project convention).
