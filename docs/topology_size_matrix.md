# Calibrated SI topology/size matrix

## Scientific design

This runner collects Full Gillespie versus original Production MicroMacro with
independent RNG per MicroModel. It does not edit production Python sources.
It measures total approximation error of this Production algorithm; it does
not identify a specific rate, placement or synchronization error by itself.

Defaults in `extrascripts/runners/configs/topology_size_matrix.json`:

| Parameter | Values |
|---|---|
| Macro graph | Chain, L=10 |
| Nodes per community n | 50, 100, 250, 1000 |
| Internal graphs | random_connected d=6,10,20; complete |
| Links per adjacent community pair k | 1,5,10,20 |
| Epidemic | SI, gamma=0, initial node=0, unit edge weights |
| beta_in / beta_out | 1; common beta within each configuration |
| macro_T | 1 (dimensionless hazard multiplier, NOT a timestep) |
| Main replications | 100 paired networks per configuration |
| Full-only calibration | 5 separate network/event seeds per configuration |
| Target calibration median | max(T_last,T_99) = 2500 |
| Main observation horizon | 10000 for both simulators |
| Base dimensionless step rho=beta*tau_micro | 0.003 |
| Refinement audit rho | 0.001 and 0.0003 |
| Refinement cells | All 64 graph configurations |
| Refinement replications | 100 each; null in config follows the main runs count |

The default matrix has 64 cells, 6400 paired replications, 12800 baseline
simulator runs, 12800 additional independent-MM refinement runs, and 320 Full-only
calibration runs. Refinement uses the SAME networks and event seeds as those
main replications. Every Full/base-MM/refined-MM parameter set has 100 runs.
This is 25600 data-collection simulations plus calibration. Refinement results
must still be inspected before claiming convergence.

## Why beta can vary

With gamma=0 and beta_in=beta_out=beta, Full has generator Q_beta=beta*Q_1.
Consequently all event times scale as 1/beta, while the stochastic sequence of
events has the same law. Independent micro RNG ownership does not alter this
time-scaling fact. There is no need to search a beta grid.

The runner simulates Full at beta_ref=1 on calibration seeds, measures

```
C = median(max(T_last, T_99))
beta = C / 2500
tau_micro = rho / beta
```

`T_last` is first infection of the last community. `T_99` is first infection
of ceil(0.99*L*n) nodes. Using both supports subsequent epidemic curves, rather
than calibrating solely on a single arrival. T_all is also saved in main data.
The normalized pilot horizon starts from a rough branching/bridge estimate;
if Full is censored, that pilot is rerun at four times the horizon with the
same graph/seeds, up to the configured attempt count. No censored value is
replaced with its horizon or used to fit beta. Calibration failures are recorded;
collection proceeds for successfully calibrated cells and leaves explicit gaps
for --resume to retry later.

Beta is calibrated separately for each (n,d,k), never separately for the two
simulators and never per main replication. Calibration uses ONLY Full and
disjoint seeds. It does not minimize Full/MM disagreement.

The 500–10000 request is interpreted as a useful observation scale: the target
median is 2500 and the horizon is 10000. No finite horizon or beta can guarantee
that every stochastic arrival falls between 500 and 10000. Runs below 500 are
retained; runs not reaching endpoints by 10000 are retained as right-censored.
`completion` refers to community arrival, not all-node saturation or job success.

## Plotting time correctly

Both physical `time=t` and `time_u=beta*t` are saved. Compare densities/sizes/k
on `time_u` to recover their physical relative dynamics at common beta=1.
`time_relative=beta*t/C` additionally expresses time relative to the Full-only
calibration reference. This second normalization removes between-cell reference
speed differences and is useful for shape comparisons; do not confuse it with
the common-beta time axis. Full and MM use exactly the same transformation.
Never independently shift or normalize Full and MM by their own fitted medians:
doing that would remove the timing discrepancy of interest.

Different calibrated cells have different normalized horizons `U_end=beta*10000`.
Use a common normalized observation interval or explicitly handle censoring
when comparing survival/distribution distances across cells. Raw physical-time
distances from different-beta cells are not directly comparable.

## Synchronization and independent RNG

Varying beta at fixed tau would change numerical error. Therefore base rho is
fixed across cells and tau=rho/beta. `macro_T` stays 1. The runner regression
checks time scaling on Full and independent MM, including infection identities.
Holding rho fixed removes a confound from beta calibration, but does not ensure
equal accuracy across graph densities: rho*d can grow with n for cliques.
Inspect the refinement comparisons before attributing size/density effects solely
to rate abstraction or uniform placement. No convergence is assumed.

The existing experimental implementation supplies one Random stream per
MicroModel; clones copy source-community RNG state and refresh restores it.
Cache discard, normal repeated imports, production scheduler, midpoint
integration, aggregate hazard and uniform placement are retained. Macro draws
still use the explicitly seeded worker-local Python global stream. Every job
sets its deterministic event seed; workers do not share RNG state.

## Network ensemble and computational settings

`random_connected` is the existing network.py rejection sampler for uniform
G(n,m) conditioned on connected, with m=round(d*n/2), p=d/(n-1). It is not a new
tree-based generator. At n=1000,d=4, rejection is impractical; default d starts
at 6. Degree 4 can be explicitly requested for smaller n in a separate config.
Generation errors/timeouts are reported as generation failures, not epidemic
censoring, and no fallback graph ensemble is substituted. On POSIX, generation
has a configurable 180-second wall-time limit; on platforms without SIGALRM,
the existing generator's attempt limit still applies but wall-time enforcement
is unavailable. This runner is tested in the project's Linux/WSL environment.

Basic production has alpha_mode=off and correction_mode=off. In the runner only,
the two unused spectral/boundary descriptor functions can be temporarily skipped
to avoid repeated dense eigendecompositions. Their neutral metadata are explicitly
placeholders, NOT measured graph statistics. Both functions are restored after
generation. A gate checks identical generated edges/W and pathwise independent-MM
output against normal generation. Actual per-node internal and external degrees
are saved separately. Set skip_unused_spectral_descriptors=false to retain the
original expensive computations.

The default is 6 worker processes, one BLAS thread each. Complete-graph jobs with
n>=250 run alone after the other worker pool closes, to avoid simultaneous
multi-GB graph copies. Processes are recycled after every task. Large complete
graphs can still be expensive in RAM and wall time. Calibrating the model's time
units does not eliminate the computational cost of a given normalized horizon.

## Commands

From the repository root, using an environment with numpy and networkx:

```bash
python extrascripts/runners/run_topology_size_matrix.py
```

Resume interrupted collection (completed calibration is reused; there are no
automatic repeated smoke runs or regression runs on a matching resume):

```bash
python extrascripts/runners/run_topology_size_matrix.py --resume
```

Calibration only, then continue main collection:

```bash
python extrascripts/runners/run_topology_size_matrix.py --calibrate-only
python extrascripts/runners/run_topology_size_matrix.py --resume
```

Preview configuration without simulations:

```bash
python extrascripts/runners/run_topology_size_matrix.py --plan-only
```

Edit a copy of `extrascripts/runners/configs/topology_size_matrix.json`, then use:

```bash
python extrascripts/runners/run_topology_size_matrix.py --config extrascripts/runners/configs/my_matrix.json --output output/my_matrix
```

`--runs N` overrides the replication count; `--workers N` overrides worker count.
Increase the run count later without rerunning existing replications:

```bash
python extrascripts/runners/run_topology_size_matrix.py --resume --runs 150
```

With default refinement_runs=null, this extends ALL three MM steps and Full to
150 replications. Explicit --refinement-runs N can increase a separately capped
refinement count. Counts may increase, but not decrease; retain the increased
--runs setting on subsequent resume/export calls or update the config accordingly.
Workers may change on resume, but scientific settings, Python/dependency versions
and production/runner source hashes must match. Calibration is reused when only
replication counts increase. Count extensions are recorded in the manifest.
Changing scientific settings requires a new output directory. Cached successful
individual models are reused even if their paired partner was interrupted.

`--self-test` runs only the small regression/RNG/scaling gates. `--export-only`
rebuilds CSV indexes from existing checkpoints without simulations.

## Outputs

Default directory: `output/topology_size_matrix/`.

- `benchmark_manifest.json`: immutable config/source fingerprints, gates, status.
- `calibrated_parameters.json` and `.csv`: beta, tau, C, physical/normalized horizons.
- `calibration/<cell>/full_*.json`: Full-only pilot observables, seeds and validation.
- `checkpoints/<cell>/rep_XXXXX/<model>.json`: exact arrivals, T50/T99/Tall,
  completion, runtime, all seeds, network checks, trajectory path.
- `checkpoints/<cell>/rep_XXXXX/<model>.npz`: compressed grid trajectories AND
  exact SI infection event times/node/community arrays. At most L*n infections
  occur in SI, so exact infection history remains compact even for long horizons.
- `checkpoints/<cell>/rep_XXXXX/network.npz`: node/community, internal degree,
  external degree by community, W. Full edge lists are not duplicated on disk;
  network seed and frozen source fingerprints permit reconstruction.
- `runs.csv`, `community_arrivals.csv`, `runtime.csv`: indexes of individual runs,
  created after collection or using --export-only. Censored CSV times are blank,
  JSON times null. Numerical NPZ arrays contain only observed infection events.
- `completion.json`: expected/saved simulation counts and epidemic endpoint counts.
- `completion_by_model.csv`: expected/saved replications for every graph/model/step
  parameter set. Collection success and epidemic completion are separate fields.
- `calibration_failures.json`, `run_failures.json`: explicit operational failures.

NPZ grid arrays S/I/R have shape [community, time_index]. Grid time is identical
for all models in a cell; counts use right-continuous sample-and-hold of actual
infection timestamps. Exact infection arrays are `infection_time`, `infection_u`,
`infection_node`, `infection_community`. They include the seed at t=0. Events
beyond the requested horizon are excluded (Full can internally sample one event
past the horizon). The full exact SI curves can be reconstructed without reruns:

```python
import numpy as np
z = np.load("output/topology_size_matrix/checkpoints/L10_n100_d6_k5/rep_00000/full.npz")
t, I = z["time_u"], z["I"]
# I[community] is the infected count on the shared normalized time grid.
```

Do not treat a missing/failed simulation as epidemic censoring. Do not compute
unqualified full-distribution W1/KS or uncensored medians by dropping censored
runs. There are no plots or publication conclusions generated by this runner.
