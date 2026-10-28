# Multi-Topology Validation Matrix Specification

**Purpose:** Turnkey specification for validating any candidate correction identically across topologies. The Executor copies commands from this file.

---

## 1. TOPOLOGY MATRIX

Each cell specifies (macro_graph_type, micro_graph_type, d, r) with concrete `generate_two_scale_network()` arguments.

### 1.1 Matrix Definition (9 cells)

| Cell ID | macro_graph_type | micro_graph_type | Description | n_communities (L) | community_size (n) | inter_links (k) | edge_prob | leaf_count | leaf_degree |
|---------|---|---|---|---|---|---|---|---|---|
| **A1** | chain | complete | dense micro baseline | 10 | 100 | 5 | — | — | — |
| **A2** | chain | random_connected | medium sparsity, d≈4 | 10 | 100 | 5 | 0.04 | — | — |
| **A3** | chain | random_connected | sparse, d≈2 | 10 | 100 | 5 | 0.02 | — | — |
| **A4** | chain | chain | hardest sparse case (d≈1) | 10 | 100 | 5 | — | — | — |
| **B1** | complete | complete | dense macro, dense micro | 10 | 100 | 50 | — | — | — |
| **B2** | complete | random_connected | mixed topology | 10 | 100 | 50 | 0.04 | — | — |
| **C1** | star | complete | hub-and-spoke macro, dense hub | 10 | 100 | — | — | — | — |
| **C2** | star | random_connected | hub-and-spoke macro, sparse hub (d≈4) | 10 | 100 | — | 0.04 | — | — |
| **D1** | chain | clique_leaves | structured sparsity: 50-node clique + 6 leaves | 10 | 50 | 5 | — | 6 | 5 |

### 1.2 Parameter Mapping Details

#### Macro graph type
- **chain**: inter-community edges form a line graph (community i ↔ community i±1).
- **complete**: all pairs of communities connected.
- **star**: community 0 is the hub (connected to all others); communities 1..L-1 are leaves (connected only to community 0).

#### Micro graph type
- **complete**: full clique within each community (all nodes connected).
- **random_connected**: Erdős–Rényi random graph with fixed average degree d, guaranteed connected.
  - Mapped to argument: `edge_prob = d / (n - 1)`
  - Example: d≈4 on n=100 → edge_prob=0.04; d≈2 → edge_prob=0.02
- **chain**: line graph within each community (nodes form a path).
- **clique_leaves**: central clique (core_size nodes) plus leaf_count leaves attached to leaf_degree clique members.
  - Arguments: `community_size` ignored; instead pass `leaf_count` and `leaf_degree`.
  - Example in D1: community_size=50 means clique_size=50, plus leaf_count=6 leaves, each attaching to leaf_degree=5 clique members.
  - Total community size = 50 + 6 = 56.

#### Import ratio r
- r = beta_out / beta_in
- For all cells: **use r = 2.0** (chosen as the premature-arrival regime per CE-10 smoke results).
- beta_in = 0.003 (SI dynamics: gamma=0, so "SI" realized as SIR with gamma=0).

#### Special case: star macro
- `macro_graph_type="star"` enforces:
  - community 0: always a clique of size community_size (e.g., 100 nodes, full connectivity).
  - communities 1..L-1: single-node leaves (no internal edges, only cross-community link to hub).
  - Arguments `leaf_count`, `leaf_degree` are **ignored** in star mode.
  - **Correction behavior**: Corrections that depend on μ₂ (Fiedler value) or spectral damping will self-disable or return α_i = 1 for single-node communities (μ₂ undefined / = 1 by convention). This is expected and not a failure.

---

## 2. SMOKE BUDGET (Minimal CPU Overload)

### 2.1 Configuration

```
n_runs = 5           # replications per cell
n_cells = 9          # topology matrix cells
n_workers = 2        # parallel processes
```

**Total work estimate:**
- Serial sims per cell: 5 × 1 (Micro) + 5 × 2 (MicroMacro off + corrected) = 15 per cell.
- Total across 9 cells: ~135 simulator runs.
- With --workers 2: ~68 sequential batches; ~30–45 minutes wall-clock (depending on T_end and CPU contention).

### 2.2 Worker Count Justification

- **Current CPU state**: CE-06 uses --workers 8 (but likely throttled by core count ≤16); CE-10 uses --workers 3.
- **Recommendation**: --workers 2 avoids:
  - Competing for L3 cache with other processes.
  - Blocking the main terminal/IDE.
  - Triggering OS scheduler contention.
- **Minimum**: 1 worker (sequential, slower but lower contention).
- **Maximum safe**: 3 workers (leaves 13 cores for system and other tasks on a 16-core machine).

---

## 3. PER-CANDIDATE VALIDATION PROTOCOL

### 3.1 Paired-Seed Convention

For each cell (scenario) and replication:
```
network_seed = base_seed + rep_id - 1
run_seed     = base_seed + 50000 + rep_id - 1
```

Example: base_seed=900000, rep_id=1 → network_seed=900000, run_seed=950000.
Use the same (network_seed, run_seed) pair for all simulators (Full Micro, MicroMacro baseline, MicroMacro corrected).

### 3.2 Per-Cell Workflow

1. **Full Micro (ground truth, once per cell/rep):**
   - Run `MicroSimulator(..., full_graph=full_graph, ...)` on the complete two-scale network.
   - Seed: run_seed.
   - Extract arrival times T_i for each community i (last node to become infected).
   - Output: arrivals_Micro[i] for i=0..L-1.

2. **MicroMacro Baseline (alpha_mode="off", correction_mode="off"):**
   - Run `MicroMacroSimulator(..., alpha_mode="off", correction_mode="off")`.
   - Seed: same run_seed.
   - Extract arrivals_MicroMacro_off[i].

3. **MicroMacro Corrected (candidate parameters, e.g., alpha_mode="spectral", correction_mode="off"):**
   - Run `MicroMacroSimulator(..., alpha_mode=<candidate>, correction_mode=<candidate>)`.
   - Seed: same run_seed.
   - Extract arrivals_MicroMacro_corrected[i].

### 3.3 Metrics Computation (Per Cell)

After all n_runs replications, compute:

- **dRMST_off** = mean(T_MicroMacro_off) − mean(T_Micro), restricted-mean-survival-time (censoring: T_censored = min(T_observed, T_end)).
- **dRMST_off_norm** = dRMST_off / T_end.
- **dRMST_corrected** = mean(T_MicroMacro_corrected) − mean(T_Micro).
- **dRMST_corrected_norm** = dRMST_corrected / T_end.
- **rel_W1_off** = W₁(arrivals_Micro, arrivals_MicroMacro_off) / mean(arrivals_Micro).
- **rel_W1_corrected** = W₁(arrivals_Micro, arrivals_MicroMacro_corrected) / mean(arrivals_Micro).
- **gap_closure** = (|dRMST_off| − |dRMST_corrected|) / |dRMST_off| [if |dRMST_off| > 0; else NaN].
- **completion_Micro**, **completion_MicroMacro_off**, **completion_MicroMacro_corrected**: fraction of runs where T_i < T_end (not censored).
- **95% CI on dRMST**: bootstrap (n_bootstrap=1000) on the difference vector diffs = T_MicroMacro − T_Micro.

### 3.4 Acceptance Rule (Minimax over Topologies)

A candidate correction is **approved** if:

1. **Across all 9 topology cells**, gap_closure ≥ 0 (improvement or neutral) **OR self-disabled** (e.g., star leaf cells).
2. **Worst-case (minimum) gap_closure > 0.1** (i.e., closes ≥10% of error on every topology).
3. **No cell worsens** (gap_closure < 0) unless the cell is a special case (star leaves with single-node communities).

A candidate is **rejected** if any topology shows gap_closure < −0.05 (more than 5% error increase) and is not self-disabled.

---

## 4. RESULTS TABLE TEMPLATE

Fill one row per cell, one row pair per (off, corrected) variant.

### 4.1 Main Summary Table

| Cell ID | L | n | macro | micro | d_avg | r | T_end | n_runs | dRMST_off | dRMST_off_norm | rel_W1_off | dRMST_corrected | dRMST_corrected_norm | rel_W1_corrected | gap_closure | completion_Micro | verdict |
|---------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 10 | 100 | chain | complete | 99 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | ✓/✗/self-disable |
| A2 | 10 | 100 | chain | random_connected | 4 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| A3 | 10 | 100 | chain | random_connected | 2 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| A4 | 10 | 100 | chain | chain | 1 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| B1 | 10 | 100 | complete | complete | 99 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| B2 | 10 | 100 | complete | random_connected | 4 | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| C1 | 10 | 100/1 | star | complete | 99 (hub)/0 (leaf) | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| C2 | 10 | 100/1 | star | random_connected | 4 (hub)/0 (leaf) | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |
| D1 | 10 | 56 | chain | clique_leaves | 7.8 (avg) | 2.0 | — | 5 | — | — | — | — | — | — | — | — | — |

**Verdict key:**
- ✓ = Approved (gap_closure > 0.1 everywhere, no regressions).
- ✗ = Rejected (gap_closure < 0 on at least one non-self-disabled cell).
- self-disable = Correction was not activated on this cell (e.g., single-node micro communities in star case).

### 4.2 Detailed Per-Cell Report

For each cell, also record:
- **CI_low, CI_high**: 95% bootstrap confidence interval on dRMST_off and dRMST_corrected.
- **Micro/MicroMacro completion rates**: fraction of runs that reached last layer before T_end.
- **KM survival curve plots** (optional): overlay of Full Micro, MicroMacro off, MicroMacro corrected arrival CDF.

---

## 5. EXACT RUNNER SPECIFICATION

### 5.1 Existing Harness (Reuse CE-10)

The reference harness is `extrascripts/runners/run_ce10_correction_ladder.py`:
- **Input**: CellConfig list (topology matrix cells + r, d, beta ratios), --grid smoke/pilot/full, --workers N.
- **Output**: CSV of (scenario, replication_id, model, variant, layer, T_i, descriptors).
- **Adaptation**: CE-10 currently hardcodes macro=chain, micro=random_connected. For full multi-topology validation, either:
  - Option A: Extend CE-10 to accept a topology matrix JSON and iterate.
  - Option B: Create a lightweight wrapper `run_correction_validation.py` that calls CE-10-style logic per cell.

### 5.2 New Runner Harness (Option B: Minimal)

**File:** `extrascripts/runners/run_correction_validation.py` (to be written)

**Signature:**
```bash
.venv/bin/python extrascripts/runners/run_correction_validation.py \
  --topology-matrix docs/error_reduction_research/topology_matrix.json \
  --correction-mode <name> \
  --grid smoke|pilot|full \
  --workers <N> \
  [--resume]
```

**Arguments:**
- `--topology-matrix`: JSON file defining the 9 cells (macro_graph_type, micro_graph_type, d, r, etc.).
- `--correction-mode`: candidate identifier, e.g., "spectral", "boundary_damping", "off" (baseline).
- `--grid`: smoke (n_runs=5), pilot (n_runs=30), full (n_runs=60).
- `--workers`: parallel process count (recommended: 2 for busy machine).
- `--resume`: skip already-completed cells.

**Output:**
- `output/claude_experiments/correction_validation_{correction_mode}_{grid}/`
  - `cells/arrivals_{cell_id}.csv` (per-cell arrival data).
  - `summary_{correction_mode}_{grid}.csv` (metrics per cell: dRMST_off, rel_W1_off, gap_closure, etc.).
  - `manifest.json` (metadata: started, cells, correction, grid, worker_count).

**Reuse from CE-10:**
- `CellConfig` dataclass.
- `adaptive_T_end()` function.
- `compute_descriptors()`, `compute_metrics()` helpers.
- `_arrival_times_micro()`, `_arrival_times_mm()` extractors.
- `MicroSimulator`, `MicroMacroSimulator` classes.

**Implementation sketch** (do NOT write the full harness; just the command line):
```
1. Load topology_matrix.json (list of 9 CellConfig records).
2. For each cell:
   a. Generate network (paired seeds as per CE-10 convention).
   b. Run Full Micro once (seed=run_seed).
   c. Run MicroMacro twice (off + candidate) on identical run_seed.
   d. Extract arrivals, compute metrics.
   e. Write arrivals_{cell_id}.csv.
3. Aggregate metrics into summary_{correction_mode}_{grid}.csv.
4. Report verdict (approved/rejected) based on minimax gap_closure rule.
```

---

## 6. TOPOLOGY MATRIX JSON (Input Configuration)

**File:** `docs/error_reduction_research/topology_matrix.json`

```json
{
  "cells": [
    {
      "id": "A1",
      "L": 10, "n": 100, "k": 5,
      "macro_graph_type": "chain",
      "micro_graph_type": "complete",
      "edge_prob": null,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "A2",
      "L": 10, "n": 100, "k": 5,
      "macro_graph_type": "chain",
      "micro_graph_type": "random_connected",
      "edge_prob": 0.04,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "A3",
      "L": 10, "n": 100, "k": 5,
      "macro_graph_type": "chain",
      "micro_graph_type": "random_connected",
      "edge_prob": 0.02,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "A4",
      "L": 10, "n": 100, "k": 5,
      "macro_graph_type": "chain",
      "micro_graph_type": "chain",
      "edge_prob": null,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "B1",
      "L": 10, "n": 100, "k": 50,
      "macro_graph_type": "complete",
      "micro_graph_type": "complete",
      "edge_prob": null,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "B2",
      "L": 10, "n": 100, "k": 50,
      "macro_graph_type": "complete",
      "micro_graph_type": "random_connected",
      "edge_prob": 0.04,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0
    },
    {
      "id": "C1",
      "L": 10, "n": 100, "k": null,
      "macro_graph_type": "star",
      "micro_graph_type": "complete",
      "edge_prob": null,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0,
      "_note": "star mode: community_size=100 (hub), other communities=single-node leaves"
    },
    {
      "id": "C2",
      "L": 10, "n": 100, "k": null,
      "macro_graph_type": "star",
      "micro_graph_type": "random_connected",
      "edge_prob": 0.04,
      "leaf_count": null, "leaf_degree": null,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0,
      "_note": "star mode: hub has edge_prob=0.04 (d≈4); leaves are single-node"
    },
    {
      "id": "D1",
      "L": 10, "n": 50,
      "macro_graph_type": "chain",
      "micro_graph_type": "clique_leaves",
      "edge_prob": null,
      "leaf_count": 6, "leaf_degree": 5,
      "k": 5,
      "beta_in": 0.003, "r": 2.0, "gamma": 0.0,
      "_note": "clique_leaves: 50-node clique + 6 leaves each attaching to 5 clique members; total size=56"
    }
  ],
  "smoke": {
    "n_runs": 5,
    "workers": 2,
    "estimated_wall_clock_minutes": 45
  }
}
```

---

## 7. EXAMPLE COMMAND LINES (Ready to Copy)

All commands assume WSL bash and `.venv` Python venv in the repo root.

### 7.1 Smoke Baseline (MicroMacro off)

```bash
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
wsl bash -c ".venv/bin/python extrascripts/runners/run_ce10_correction_ladder.py --grid smoke --workers 2"
```

**Output:** `output/claude_experiments/ce10_correction_ladder_smoke/summary_*.csv`

### 7.2 Multi-Topology Smoke (with hypothetical candidate "spectral_damping_v2")

```bash
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
wsl bash -c ".venv/bin/python extrascripts/runners/run_correction_validation.py \
  --topology-matrix docs/error_reduction_research/topology_matrix.json \
  --correction-mode spectral_damping_v2 \
  --grid smoke \
  --workers 2"
```

**Output:** `output/claude_experiments/correction_validation_spectral_damping_v2_smoke/`

### 7.3 Baseline Metrics (no correction, all topologies)

```bash
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
wsl bash -c ".venv/bin/python extrascripts/runners/run_correction_validation.py \
  --topology-matrix docs/error_reduction_research/topology_matrix.json \
  --correction-mode off \
  --grid smoke \
  --workers 2"
```

### 7.4 Resume After Interrupt

```bash
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
wsl bash -c ".venv/bin/python extrascripts/runners/run_correction_validation.py \
  --topology-matrix docs/error_reduction_research/topology_matrix.json \
  --correction-mode <name> \
  --grid smoke \
  --workers 2 \
  --resume"
```

### 7.5 Subsequent Pilot / Full (after smoke passes)

```bash
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
wsl bash -c ".venv/bin/python extrascripts/runners/run_correction_validation.py \
  --topology-matrix docs/error_reduction_research/topology_matrix.json \
  --correction-mode <name> \
  --grid pilot \
  --workers 2"
```
