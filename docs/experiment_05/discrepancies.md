# Experiment 05 production/paper audit

Audit date: 2026-07-17. This document records observed semantics before any
Experiment 05 implementation. Production files are not modified by the
experiment.

## Production semantics

- `MacroEngine` evaluates
  `beta_macro * macro_T * W_ij * (I_i/N_i) * (S_j/N_j)`.
  `W_ij` is used without normalization. For the chain generator it is the
  number of distinct inter-community edges, hence `W_ij = k`.
- The production MicroMacro scheduler estimates the total hazard at the
  midpoint of each synchronization segment using prospective micro-model
  clones. It accumulates a midpoint rectangle and linearly interpolates a
  threshold crossing.
- At an interpolated macro-event time, the real micro models are advanced to
  that time and the source--destination pair is sampled from hazards computed
  from the actual event-time counts. Thus the hazard used for timing and the
  hazard used for the event mark need not be the same matrix.
- One exponential total-macro budget persists across synchronization
  boundaries and zero-hazard intervals. A fresh budget is drawn after each
  macro event.
- A macro import invalidates the cached destination micro event and samples a
  new event under the changed state. No explicit local residual-budget number
  is stored.
- Import placement is uniform over susceptible destination nodes.
- `alphas` are accepted and stored by the macro engine, and mentioned in a
  docstring, but the executed hazard is linear in `I_i/N_i`.
- Production MicroMacro uses a shared Python RNG. Experiment 05 therefore
  needs an isolated orchestrator to provide the requested separated streams.

## Differences from the paper

1. Sections 2.5--2.6 describe cumulative hazard along the evolving
   piecewise-constant micro summaries. Production uses midpoint quadrature on
   synchronization segments, so its hazard integral is approximate.
2. The paper presents one predictable intensity for both event timing and
   pair selection. Production uses midpoint counts for timing and actual
   event-time counts for pair selection.
3. The paper gives an explicit local residual-budget update. Production
   instead invalidates the cached local event and resamples. This is
   distributionally compatible with the paper's memorylessness statement,
   but is not a pathwise residual-budget implementation.
4. The paper says macro summaries change with micro events. Production only
   observes them through midpoint probes, synchronization endpoints, and
   macro-event refreshes.

## Experiment-design qualifications

- The specified lag formula uses
  `i_lag(t) = i(max(0, t - tau_find))`. Consequently it holds the nonzero
  initial infected fraction over the pre-lag interval rather than imposing a
  zero prehistory. Experiment 05 will implement this formula literally and
  interpret it as delayed growth, not a complete delay of hazard onset.
- Canonical Experiment 01 uses `initial_node=0`; the sparse Table-2 benchmark
  also uses node 0 and different seeds. Main Experiment 05 uses a
  uniform-random initial node. The reference gate will therefore generate new
  matched-uniform Experiment-01-style Full Micro runs, rather than treating
  archived outputs as an exchangeable reference population.
- The existing Table-2 benchmark is configured by saved JSON parameters, not
  YAML. Experiment 05 creates its own YAML scenario files.
- Production midpoint error remains an explicit confounder. Corrected variants
  preserve the scheduler rather than silently replacing it with an exact
  event-driven scheduler.

## Scientific protocol revision after the clique pilot

The finite-clique pilot falsified the proposed acceptance expectation that all
corrected variants should lie within two percent relative Wasserstein distance
of V1.  This is not evidence of a failed correction implementation.  For the
specified (K_{100}) control, the formulas themselves give

```text
lambda_1 = 99, ell = 1,
tau_find = ln(100) / (0.003 * 98) = 15.66,
kappa = 0.003 * 98 = 0.294,
1 / kappa = 3.40.
```

These are finite, non-negligible time scales, and their effects can accumulate
over nine interfaces.  Therefore the observed clique discrepancies (V2 0.214,
V3 0.068, and V4 0.265 relative W1 versus V1) are compatible with the literal
formulas even though they contradict the original two-percent expectation.
The finite-clique result is retained as a nonblocking sensitivity diagnostic
and as a warning that "minimal on a clique" is only an asymptotic descriptor
statement, not a valid (K_{100}) outcome-level acceptance criterion.

Hard implementation gates remain the exact limits: injected
`tau_find = 0` and `kappa = infinity` must reproduce V1 pathwise under common
random streams, while the degenerate λ1=1 behavior must remain exactly zero.
For equal (n) and β, the descriptor-level dense control must also satisfy
`tau_find(clique) < tau_find(sparse)` and
`kappa(clique) > kappa(sparse)`.  These checks validate implementation and the
intended spectral ordering without imposing an outcome threshold that is not
implied by the formulas.

The five primary sparse variants and their formulas are unchanged.  No
coefficient, delay, relaxation rate, threshold, or prehistory convention will
be calibrated against sparse Experiment 05 outcomes.  In particular, the
literal `max(0, t - tau_find)` rule retains the initial infected fraction in
the seed community before `tau_find`; this is delayed growth rather than a
zero-hazard prehistory and must be reported as such.

The original Experiment 05 reference check reran its own V0 configuration and
seeds and was consequently a reproducibility check, not an independent test
for historical production drift.  A valid Experiment 01 drift gate must use
the archived Experiment 01 contract (`base_seed=270000`, `initial_node=0`, and
the archived per-replication network and micro-event seeds) and compare newly
generated Full Micro arrivals/trajectories with
`output/ablation_hazard_placement/final_100`.  Exact comparisons should be used
where the archived serialization permits them; otherwise the gate must report
the precise compatibility limitation and use the prespecified completed
terminal KS check in addition to seed/config provenance.  The main Experiment
05 uniform-random initialization remains unchanged and is not exchangeable
with this separate historical-drift control.

These differences do not require a production change and do not make the
experiment infeasible.

## Pre-experiment production hashes

The following SHA-256 values were recorded before Experiment 05 code was
created. The production-scoped `git diff` was empty.

```text
engines/macro_engine.py       7c64d622893be7e9e42b068e5767955ce145da9d98eff91e2fb262720d2ab073
engines/micro_engine.py       c0dcc3751829aa62c04ba29a585244c247c642951049bac13f52fe2ea5f993b0
engines/micro_model.py        78bd7063c6cccfac29eca42af126cfd8ed148d4accc553d15ea350a122aa5d72
orchestrators/micro.py        1f944abea92f3ff7020a0cb87fc5d72b788f8db3005932ab132c12d4b7cc8d73
orchestrators/micromacro.py   159132dad7f59d1babf9ac8f72b997f3bfc2e39b633a40a26a0f7edd955c8695
simulators/micro.py           d3269c68e28d0282e2908b9d17c565beabbd79f401e2baf9eacda42f55507e0f
simulators/micromacro.py      7a8993c5ce9e2ed59ab1de69917510d7f3bac1f4382dab12f2f8cc61041774ce
network.py                    26538d17ac1d1fa8a05369fc95f77c3d4f24e794c0fec58008be691b355a3b8c
types.py                      659f7c3f602acc5a1c2f7bc0088ae4e715e6436a9ad6140661a95f31b0dd8bbb
simulate.py                   35848e1f881bae3c0380634a8252198a745345b56888a705218f9eb18e575093
```

The `simulators/micro.py` checksum is verified programmatically by the
production-immutability gate; the tabulation above is descriptive and the
machine-readable manifest is authoritative.
