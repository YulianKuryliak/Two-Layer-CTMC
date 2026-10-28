# Experiment 05 implementation plan

1. Snapshot production hashes and document code/paper discrepancies before
   writing experimental code.
2. Implement graph descriptors, corrected contagiousness policies, a
   production-aligned isolated scheduler, deterministic streams, schemas,
   resume-safe CLI, analysis, plots, and report generation under the
   Experiment 05 namespace.
3. Run hard implementation gates before the five-replication baseline smoke:
   degenerate λ1=1 behavior; pathwise V1 identity under injected
   `tau_find=0` and `kappa=infinity`; deterministic descriptor convergence;
   and the expected descriptor ordering
   `tau_find(clique) < tau_find(sparse)`,
   `kappa(clique) > kappa(sparse)` at equal (n,β).  Also require
   reproducibility, pairing, schema, finite-value, and production-immutability
   gates.  Stop on failure of any of these implementation/data-integrity gates.
4. Preserve the completed 20-replication finite-clique pilot as a nonblocking
   sensitivity diagnostic.  Its original two-percent relative-W1 expectation
   was falsified because the literal (K_{100}) formulas imply
   `tau_find=15.66` and `1/kappa=3.40`, not zero correction.  Do not change the
   formulas or tune any scale using sparse outcomes; report the clique result
   and revised interpretation transparently.
5. If the hard gates and smoke pass, run twenty matched development
   replications for baseline and `k1`.  Estimate V2−V1, V3−V1, and V4−V1,
   plus the factorial interaction `(V4-V2)-(V3-V1)`, at replication level.
   Treat lack of improvement as a scientific result rather than a gate
   failure.  Validation results remain development-only and cannot establish
   H1, H2, or additivity without the planned final sample.
6. Implement the historical Experiment 01 drift gate as a separate control:
   rerun Full Micro with archived Experiment 01 settings and per-replication
   seeds (`base_seed=270000`, `initial_node=0`) and compare against
   `output/ablation_hazard_placement/final_100`.  Require exact arrival and
   trajectory agreement wherever archived precision/serialization supports
   it, plus the prespecified completed-terminal KS `p > 0.05`; otherwise stop
   and document the exact incompatibility.  Do not substitute a duplicate run
   of Experiment 05 V0, and do not compare the main uniform-random Experiment
   05 initialization as if it were the fixed-node Experiment 01 design.
7. Generate the development analysis and PDF report. Do not run the final
   five-scenario, 100-replication stage without `--validation-approved` and a
   separate user instruction.

Scientific payloads are deterministic and compared byte-for-byte after
canonical serialization. Runtime, timestamps, and environment metadata are
explicitly excluded from that reproducibility comparison.

The primary scientific contract remains the five specified variants with the
literal lag prehistory and analytical latent update.  Boundary-node timing may
be measured in V0 as a mechanistic diagnostic (elapsed from each community's
arrival to first infection of an outgoing boundary node), but boundary
identity is never an input to V2–V4.  No post-hoc scaling selected from baseline
or `k1` outcomes may be promoted to a confirmatory result.
