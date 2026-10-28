# Error Reduction Research — Running Results Summary

**Last updated:** 2026-07-31

Compact summary table of per-topology × correction results from error-reduction experiments.

---

## Implementation Status

| Correction | Mode | Status | Code Location |
|---|---|---|---|
| C1 (spectral alpha) | `alpha_mode="spectral"` | ✓ Baseline (existing) | `MacroEngine`, CE-10 |
| **C2 (boundary-availability phi)** | `correction_mode="boundary_phi"` | ✓ Implemented | `MacroEngine._compute_hazards_matrix` |
| **C12 (renormalized rate phi)** | `correction_mode="renorm_phi"` | ✓ Implemented | `MacroEngine._compute_hazards_matrix` |
| C3 (reach-time onset delay) | `correction_mode="onset_delay"` | ✗ Deferred (phase 2) | — |

---

## Unit Test Results (L=3, n=10, chain/complete, r=2.0)

Quick validation test showing C2/C12 dampen macro hazard correctly:

| Variant | Hazard[0,1] | Damping vs off |
|---|---|---|
| Baseline (off) | 0.004200 | — |
| C2 (boundary_phi) | 0.001307 | **68.9%** ✓ |
| C12 (renorm_phi) | 0.001500 | **64.3%** ✓ |

**Interpretation:** Both corrections reduce premature export hazard significantly. C2 slightly stronger than C12 in this example. Full multi-topology validation pending (see 02_execution_log.md).

---

## Multi-Topology Smoke Results (PENDING)

Planned topology matrix:

| macro_type | micro_type | r | d | off_dRMST_norm | C2_gap_closure | C12_gap_closure |
|---|---|---|---|---|---|---|
| chain | complete | 1.0 | 6.0 | — | — | — |
| chain | complete | 2.0 | 6.0 | — | — | — |
| chain | random_connected | 1.0 | 6.0 | — | — | — |
| chain | random_connected | 2.0 | 6.0 | — | — | — |
| chain | clique_leaves | 1.0 | 6.0 | — | — | — |
| chain | clique_leaves | 2.0 | 6.0 | — | — | — |
| complete | complete | 1.0 | 6.0 | — | — | — |
| complete | complete | 2.0 | 6.0 | — | — | — |
| complete | random_connected | 1.0 | 6.0 | — | — | — |
| complete | random_connected | 2.0 | 6.0 | — | — | — |
| complete | clique_leaves | 1.0 | 6.0 | — | — | — |
| complete | clique_leaves | 2.0 | 6.0 | — | — | — |

**Status:** Harness created (`run_er01_multitopo_smoke.py`), code validated. Full smoke run pending due to CPU contention with CE-06/CE-10 experiments. Planned run after CE-06 completes or on reduced network size (L=5, n=50).

---

## Minimax Gap Closure (worst-case across topologies)

To be computed from multi-topology smoke:

```
minimax_C2 = min(C2_gap_closure for all topologies)
minimax_C12 = min(C12_gap_closure for all topologies)
```

**Target:** minimax ≥ 0.5 (closes at least 50% of error gap on worst-case topology without regression).

**Self-disable check:** complete/complete topology should show gap_closure ≈ 0 (identity, no improvement needed, no regression).

---

## Next Steps

1. Complete multi-topology smoke run (`run_er01_multitopo_smoke.py --grid smoke --workers 2`)
2. Fill in the results table above
3. Compute minimax gap_closure for C2 and C12
4. If minimax ≥ 0.5 and no regression on complete micro → deploy winner as default correction
5. If both show similar minimax, prefer C12 (simpler analytical form)
6. If neither reaches minimax ≥ 0.5, consider C3 (onset delay) or refinement of phi formulas
