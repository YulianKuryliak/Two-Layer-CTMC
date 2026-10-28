# METHODS — Error source decomposition

## Design principles

- All comparisons are paired by network replication whenever possible.
- Terminal time is restricted at T_end; RMST is the primary metric.
- Bootstrap confidence intervals resample replication IDs, preserving within-replication pairing.
- Percentages are ratios of mean signed corrections within a specified denominator and are bootstrapped as ratios.
- Negative or >100% signed contributions are allowed; they indicate compensation/overshoot, not arithmetic failure.

## Structural attribution

Uses Experiment 06 final_100. Source and destination boundary marginals are a genuine 2x2 factorial and use two-player Shapley attribution. Edge-pair and placement are conditional refinements. Exact placement is not crossed with aggregate hazard because it is undefined when aggregate fires without a real active edge.

## Calibration

Event-driven A/U only; candidate c = 0.7,0.8,0.85,0.9,1.0,1.1; 5 folds defined by replication_id mod 5. c* minimizes absolute mean RMST difference on 80 training networks, then is evaluated on 20 held-out networks.

## Production RNG study

Current Production is reproduced exactly using the synchronization-audit seeds. The intervention gives every MicroModel a private random.Random stream and makes midpoint probes use a copied stream state, preventing probe draws from mutating the source stream. The original macro scheduler otherwise remains unchanged.

## Re-import mediation

Single-entry zeros hazards into communities that already contain infection. It is a mediation/sensitivity intervention, not counted as a primary source. Difference-of-differences tests whether the RNG correction depends on repeated imports.

## RNG unit test

Two identical 2-node micro-processes are advanced sequentially to the same horizon. Shared-global and independent streams are compared over 50,000 seeds.
