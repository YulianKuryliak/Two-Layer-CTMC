## 2026-10-07 — Chain-only TwoScale rerun; original midpoint retained

- User confirmed the macro layer receives aggregates only at synchronization; withdrew the proposed removal of midpoint. The original black-box midpoint scheduler at tau_micro=1, macro_T=1 is retained. No event-driven production scheduler was created or installed; provisional event-driven imports/labels in the new runner were replaced before any execution.
- Keep the corrected native micro residual clock, fresh RNG use without rollback, and cache-safe private probe copies. These production fixes were not reverted.
- Added `extrascripts/runners/rerun_chain_twoscale.py` and `reports/chain_twoscale_rerun.md`: only 19,200 new TwoScale trajectories, 48 chain configurations (n=50,100,250,1000; ASPL=1..6; k=1,5; 400 each). Frozen graph/interface/initial-node/seed metadata from historical article pairs. Full rows are reused unchanged, including two saved horizon-40000 extensions, with hashes and separate provenance.
- New output: `output/chain_twoscale_residual_midpoint_20261007`. User launch command: `.venv/bin/python extrascripts/runners/rerun_chain_twoscale.py --run`. Resume verifies source/code/environment hashes and skips compatible completed tasks. Full simulator is never called. Journals record new TwoScale experiments, baseline reuse and failures separately from code changes.
- Static verification only: parsed source AST, checked absence of Full/event-driven simulator calls, checked saved extension labels, inspected provenance/resume logic and whitespace. No runner execution, preparation, module import, diagnostic or smoke simulation. User runs the simulations.
- Interpretation: local time/RNG corrected; midpoint synchronization error, aggregate-rate abstraction and uniform import placement remain part of this TwoScale model. Neither the revised manuscript nor historical results were edited.

## 2026-10-07 — Residual local clock after imports; reproducible SMPT rerun

- User clarified after preparation: do not launch scripts/experiments; user will run them. Further work is analysis and script preparation only. At clarification, all prior processes had finished; only two pilot pairs and the isolated diagnostic had been run, and no full suite was launched.

- User requested proportional rescheduling after imports. `MicroModel` now retains the unused aggregate exponential intensity budget: B = a_old * (t_pending - t_import), next wait = B / a_new. It discards the old event mark and selects a fresh mark from updated rates. Successive imports, inactive/zero-rate clocks and probe copies retain the correct clock state.
- Removed horizon RNG rollback/replay; each community and its disposable clone owns a private stream. Native clone/refresh copies pending proposal, rate, origin and RNG state. Explicit `micro_rngs` allows the paper runner to preserve historical per-community seeds without consuming the macro stream during initialization.
- Macro logs now include successful/repeat-import flags and destination S/I before the event, firing hazard and post-event hazard. Micro events are logged in chronological order so wait_time is nonnegative.
- Added `extrascripts/runners/rerun_smpt_paper.py`: 46,000 frozen matched pairs (10,000 clique–leaf, 16,800 source–leaf, 19,200 chains), no duplicate clique-chain cell. Preserves graphs, interfaces, initial nodes, event seeds and two historical Full extensions to 40,000; writes under `output/paper_rerun_residual_clock_20261007`. Raw typed events and all SI change points retained. Atomic checkpoints, provenance checks, source/graph snapshots, automatic resume and output locking.
- Added separate append-only journals `logs/code_changes.jsonl` (source snapshots) and `logs/experiments.jsonl` (experiment lifecycle). The rerunner also keeps its own experiment journal. `AGENTS.md` requires logging all future code edits and experiment invocations.
- Verification: 10 targeted regression tests passed. Native-engine diagnostic with 20,000 seeds, 14,887 conditioned runs: import-to-node-2 probability 0.506146 (expected 0.5), mean residual wait 0.740530, 95% CI [0.727381, 0.753679] (expected 0.75); exact residual formula checked in every accepted trajectory. Diagnostic recorded in `output/validation/residual_import_clock_20261007/manifest.json` and experiment journal.
- End-to-end verification: full manifest prepared (91 cells, 46,000 pairs, exactly two Full extensions). A clique–leaf pair and n=50/ASPL=6/k=1 chain pair completed; Full clique observables and the entire Full chain infection trajectory reproduced historical results exactly. Corrected MicroMacro chain had 22 typed repeat imports, 18 before T9; all event waits were nonnegative and raw repeat flags matched counts. Resume verification reused the completed clique checkpoint without simulation. Only these two bounded pilot pairs were simulated; the full suite awaits the launch command. Added narrow .gitignore exceptions so the regression file is visible in Git.
- Scientific limits: midpoint hazard integration remains an approximation and can use a probe state later than the scheduled import. Retaining a budget does not remove that numerical error or the surrogate model's uniform placement/interface abstraction. Legacy runners that monkey-patch clone/refresh must not be used for this corrected rerun. Historical data and revised manuscript were not edited. Other unrelated working-tree edits predated this task and are not claimed here.

## v1.0.3
- Cross-platform path handling: resolve config/output/plot paths via pathlib and normalize Windows-style separators so runs work on Linux and Windows.
- Config updates: added micro plot folder keys to `config.json` for first-infection and per-community outputs.
- Visualization and simulator outputs now anchor paths relative to the script location for consistent behavior across OSes.
- Run logging and network visualization now resolve paths relative to the repo to avoid OS-specific path issues.

## v1.0.2
- Updated Micro simulator:
	- Fixed bug in Microsimulator that blocked intra-infections of other communities till the end of outbreak in current one. 
	- Infection-source was incorrect. Fixed infection-source sampling to use β · Σw units (matches total_infection_rate); removes β-dependent bias.
	- On infection, set sum_of_weights_i = s (no accumulation drift).
	- On recovery, subtract node’s β · sum_of_weights_i from total_infection_rate (prevents overcounting).
	- SIS fix: add back weights to infected neighbors (not susceptibles) when a node recovers.
	- Added numeric clamps for tiny negatives in sums and total_infection_rate.
	- simulate_step now applies the event internally and returns wait time + event; kept sample-and-hold CSV output; minor plotting/legend tweaks; graceful fallback if sim_db.log_run is absent.

- Added macro/micro topology options and randomness controls to network generation (macro: complete/chain, micro: complete/random, edge probability).
- Renamed inter-community links setting to `inter_links` and updated config, docs, and callers.
- Added `network_visualization.py` with config-driven rendering and saved outputs.
- Renamed Solid simulator to Micro simulator, updated config/data paths, and plot folders.
- Enhanced Micro simulator plots: first-infection snapshots, per-community counts, and combined community curves with infection markers.

## v1.0.1
- Optimized MicroMacro simulator (micromacro2) for faster runtime with identical logic:
	- O(1) S/I/R counters (maintained counters instead of rescanning nodes).
	- Cached neighbor lists/weights to avoid repeated NetworkX neighbor lookups.
	- Lightweight cloning that copies only mutable state and reuses static topology.
	- Hazard buffer reuse with cached flattened cumulative sums for sampling.
	- Cached next-event sampling per community, invalidated on state changes.

## v1.0 (main)
- Initial baseline release.
