# Claude Experiments Reports Index

All reports in English (PDF + TeX). Ukrainian summaries in `docs/`.

| Exp | Folder | PDF | Status | Key finding |
|---|---|---|---|---|
| CE-01 | `ce01_ablation/` | `report_ce01_ablation.pdf` | ✅ Done | Rate abstraction dominant (ΔRMST=−209); interaction effect significant |
| CE-03 | `ce03_beta_ratio/` | `report_ce03_beta_ratio.pdf` | ✅ Done | Export-ratio sign flip; see CE-11 for horizon-aware interpretation |
| CE-04 | `ce04_sir_validation/` | `report_ce04_sir_validation.pdf` | ✅ Done | Recovery worsens error; 44pp false-positive rate at R₀=1 |
| CE-06 | `ce06_regime_map/` | `report_ce06_regime_map.pdf` | Done | Error is strongest for fast export and sparse/narrow interfaces; sign-aware gating required |
| CE-07 | `ce07_calibration_floor/` | `report_ce07_calibration_floor.pdf` | Pilot | Scalar calibration helps but leaves a structural floor; full run pending |
| CE-08 | `ce08_negative_control/` | `report_ce08_negative_control.pdf` | Done | Dense negative control shows no detectable systematic RMST shift |
| CE-09 | `ce09_boundary_isolation/` | `report_ce09_boundary_isolation.pdf` | Done with limitation | Static boundary placement is not robust; modes were not network-paired |
| CE-10 | `ce10_correction_ladder/` | `report_ce10_correction_ladder.pdf` | Done | Spectral damping helps at high export ratio and regresses at low ratio |
| CE-12 | `ce12_power_calibration/` | `report_ce12_power_calibration_uk.pdf` | Done | Small 2.5%-of-horizon effects often require substantially more replications |
| Oracle bridge | `oracle_bridge_ceiling/` | `report_oracle_bridge_ceiling_uk.pdf` | Done | Source-side bridge state closes 47.4% of the point gap; degree proxy regresses |
| Sync audit | `synchronization_audit/` | `report_synchronization_audit_uk.pdf` | Done | Production/event-driven discrepancy remains above 1000 time units across tested tau |
| Gateway audit | `gateway_distance/` | `report_gateway_distance_uk.pdf` | Done | Hidden gateway architecture changes both magnitude and sign of timing bias |

## LaTeX Setup

All reports use `\documentclass[11pt,a4paper]{article}` then `\input{../preamble}`.  
Preamble provides: T1 encoding, geometry, amsmath, booktabs, xcolor, hyperref, float, microtype.  
Commands: `\SIG`, `\NS`, `\DRMS`, `\ci{lo}{hi}`.

**Note:** Ukrainian Cyrillic requires `texlive-lang-cyrillic` (not installed).  
Install: `sudo apt-get install texlive-lang-cyrillic texlive-fonts-extra`  
Then change preamble babel to `\usepackage[ukrainian]{babel}` and fontenc to `\usepackage[T2A]{fontenc}`.

## Ukrainian Summaries Location

- CE-01: `docs/experiment_ablation_hazard_placement/report.md`
- CE-03: `docs/ce03_beta_ratio/report.md`
- CE-04: `docs/ce04_sir_validation/report.md`
