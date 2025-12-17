## v1.0.1
- Optimized MicroMacro simulator (micromacro2) for faster runtime with identical logic:
	- O(1) S/I/R counters (maintained counters instead of rescanning nodes).
	- Cached neighbor lists/weights to avoid repeated NetworkX neighbor lookups.
	- Lightweight cloning that copies only mutable state and reuses static topology.
	- Hazard buffer reuse with cached flattened cumulative sums for sampling.
	- Cached next-event sampling per community, invalidated on state changes.

## v1.0 (main)
- Initial baseline release.
