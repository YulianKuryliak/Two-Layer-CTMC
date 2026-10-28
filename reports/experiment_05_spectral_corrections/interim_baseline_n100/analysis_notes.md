# Experiment 05 interim analysis — baseline, paired replications 1–100

**Status:** DEVELOPMENT RESULTS — NOT FINAL. The active final runner is still
writing later replications. This analysis uses an immutable snapshot of the first
100 complete baseline groups only and does not read the moving frontier.

## Integrity and scope

- 100 paired replications and five variants (500 raw files).
- All 500 checkpoint SHA-256 values match their raw Parquet files.
- All Parquet metadata are readable and non-empty.
- Adjacency, inter-community edges, initial node, network seed, and
  initialization seed agree across V0–V4 within every replication.
- Bootstrap uncertainty uses 1,000 replication-level resamples with seed
  `72087832`; event rows are never treated as independent observations.

## Terminal results

| Variant | Completion [95% bootstrap CI] | relW1(T_last), completed | Delta RMST vs V0 [95% CI] |
|---|---:|---:|---:|
| V0 Full Micro | 0.99 [0.97, 1.00] | 0.0000 | 0.0 [0.0, 0.0] |
| V1 baseline MicroMacro | 0.98 [0.95, 1.00] | 0.0694 | 200.6 [98.2, 307.7] |
| V2 lag | 0.00 [0.00, 0.00] | undefined | 1070.9 [995.9, 1152.0] |
| V3 latent boundary | 0.44 [0.34, 0.55] | 0.2421 | 905.8 [807.4, 989.9] |
| V4 combined | 0.00 [0.00, 0.00] | undefined | 1070.9 [991.0, 1146.3] |

Positive Delta RMST means later/censored terminal arrival than Full Micro.
Completed-only Wasserstein values for V2 and V4 are undefined because neither
variant completes by the horizon.

## Correction effects relative to V1

All contrasts below are paired and use the convention corrected variant minus V1.

| Contrast | Delta RMST vs V1 [95% CI] | Completion difference [95% CI] |
|---|---:|---:|
| V2 - V1 | 870.3 [792.0, 954.5] | -0.98 [-1.00, -0.95] |
| V3 - V1 | 705.2 [613.9, 790.5] | -0.54 [-0.64, -0.43] |
| V4 - V1 | 870.3 [792.0, 954.5] | -0.98 [-1.00, -0.95] |

The literal corrections therefore do not reduce baseline error in this cohort.
They over-delay inter-community propagation, with the lag term producing complete
terminal censoring.

## H1/H2 additivity

On layers 1–5, before severe censoring, the factorial interaction
`V4 - V2 - V3 + V1` has a 95% bootstrap interval containing zero. The combined
delay is therefore compatible with approximate additivity over these early layers.
For example, at layer 5 the mean effects are H1 = 1252.3, H2 = 474.1, combined =
1722.1, and interaction = -4.2 [-124.1, 119.9].

From layer 6 onward the interaction becomes increasingly negative. This coincides
with horizon censoring (V2 has only 90 observed layer-6 arrivals and V4 only 31;
both have zero terminal arrivals). The terminal interaction of -705.2 is therefore
a ceiling/saturation contrast and is not evidence that one correction
scientifically subsumes the other.

## Interpretation boundary

The snapshot supports a framework-level conclusion for the baseline only: the
implementation is paired and reproducible, but the proposed descriptor formulas
are too strong at their literal scale. It does not yet support the planned density
or interface-width robustness claims because `k10`, `d4`, and `d10` final cells
have not been produced, and the existing `k1` files contain only the earlier
20-replication validation cohort.

The active command requests 1,000 replications per scenario and variant (25,000
runs), not 1,000 total simulations. Based on the validation output, final hazard
CSV aggregation is projected at roughly 418 GB and may require a second similarly
sized temporary file during atomic replacement.

## Source files

- `output/experiment_05_spectral_corrections/interim_baseline_n100/manifest.json`
- `summaries/terminal_metrics.csv`
- `summaries/effects_vs_v1.csv`
- `summaries/completion_probability_ci.csv`
- `summaries/layer_factorial_effects.csv`
- `reports/experiment_05_spectral_corrections/interim_baseline_n100/report.pdf`
