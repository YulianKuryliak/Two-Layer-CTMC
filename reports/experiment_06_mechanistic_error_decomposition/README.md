# Experiment 06 — Mechanistic Error Decomposition

Purpose: identify which information losses produce the discrepancy between the event-driven two-layer CTMC approximation and the Full Micro CTMC, while keeping the intended separation between macro and micro levels.

The experiment reuses the 100 paired CE-01 networks and initial conditions and adds diagnostic oracle arms that change exactly one information channel at a time. Supplemental branches test repeated imports, boundary-only placement, and out-of-sample hidden gateway architectures (near/random/far).

## Main result

Two mechanisms are supported in the baseline sparse-chain SI regime:

1. **Source-side boundary infection state matters.** Replacing ordinary community prevalence with the infected cross-boundary weight improves RMST by +105.12 relative to Aggregate/Uniform (95% paired bootstrap CI +2.81 to +207.14).
2. **Import placement matters strongly once the hazard is sufficiently accurate.** Exact edge-conditioned placement improves RMST by +168.77 relative to exact-hazard/uniform placement (95% CI +62.04 to +275.35).

Destination-boundary susceptibility alone and edge-pair covariance are not detectably important in the baseline cell. Repeated imports also do not explain the hazard gap in the dedicated single-entry ablation.

A practical black-box interface candidate is: each source micro model exports an aggregate boundary-exposure scalar; each destination exports an aggregate boundary-susceptibility scalar; the destination micro model performs boundary-local placement internally. The macro layer does **not** need node identities or the internal edge list.

## Files

- `report.tex`, `report.pdf` — full report.
- `figures/` — report figures.
- `output/mechanistic_error_decomposition/final_100/` — main 2x2 source/destination decomposition.
- `output/mechanistic_error_decomposition/reimport_100/` — repeated-import ablation.
- `output/mechanistic_error_decomposition/boundary_placement_100/` — boundary-placement candidates.
- `output/mechanistic_error_decomposition/gateway_validation_100/` — near/random/far validation.
- `extrascripts/run_mechanistic_error_decomposition.py`
- `extrascripts/run_reimport_ablation.py`
- `extrascripts/run_boundary_placement_context.py`
- `extrascripts/run_gateway_interface_validation.py`

All new simulation branches use 100 paired replications and compact arrival-time outputs. Oracle arms are diagnostic; they are not automatically deployable until their information is exposed through a level-separation-preserving interface contract.
