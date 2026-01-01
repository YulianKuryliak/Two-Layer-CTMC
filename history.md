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
