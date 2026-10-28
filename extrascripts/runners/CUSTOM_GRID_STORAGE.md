# Custom-grid checkpoints

Run as before:

```bash
.venv/bin/python3 extrascripts/runners/run_custom_grid_independent_rng.py
```

The editable parameter arrays and worker settings remain at the top of that
script. Output remains `output/data/<scenario>/checkpoints/rep_XXXX.json.gz`.
The scenario manifest records the effective configuration and storage policy.

## Storage-only changes

- Each worker writes its own complete paired result, then returns a small
  completion receipt. Temporary files are ignored by resume. The file contents
  are flushed and fsynced before atomic replacement of the final checkpoint.
- There is no automatic CSV consolidation in custom grid. Historical CSVs are
  untouched; they are not updated and must not be used as current results.
- Failed simulation attempts are saved separately in `failures/`. They do not
  count as reusable successful pairs and will be retried on the next launch.
- New checkpoints have `format_version: 4`. `trajectories.format` is
  `numeric_columns_v1`: common family/benchmark/replication metadata once, then
  one block per simulator with numeric community/time/S/I/R columns. In the
  worker these are float64 and signed-int64 arrays; on disk they are JSON numeric
  arrays inside gzip. Float values round-trip exactly, counts are not converted
  to floating point. The original row order and all reporting points remain.
- The custom runner stops at a natural absorbing/no-event state (for its SI
  grid, normally when every node is infected) or at `T_END`, whichever comes
  first. It does **not** write a held plateau after that stop. `termination`
  records `stop_time`, `stop_reason`, and the final per-community S/I/R state,
  so an analyzer may reconstruct a plateau in memory only when it explicitly
  needs one. `T_END` is a maximum observation horizon, not a required output
  duration.
- Legacy version-2 row-dictionary checkpoints remain readable and reusable.
  Existing data is neither converted nor deleted automatically.
- A stored longer horizon is reused. If `T_END` is increased, checkpoints where
  **both** models already have `completion=true` are retained. Only censored or
  partial pairs are rerun from zero at the new horizon with the exact
  `network_seed` and `event_seed` recorded in the old checkpoint (falling back
  to deterministic derivation only if an old file lacks those fields). The old
  checkpoint is replaced only after that new paired run has been
  written successfully. This is not engine-state continuation: it is exact
  replay from the same initial seeds. Existing completed trajectories remain
  shorter, so the size plotter shows only the common exact trajectory prefix and
  labels it as such; arrivals remain valid through the new maximum horizon.

`checkpoint_io.py` provides `read_checkpoint`, `has_trajectories`,
`model_trajectory_columns`, and `iter_trajectory_rows`. The last helper restores
the old row schema lazily if a downstream consumer needs CSV export. Do not
iterate over `payload['trajectories']` as rows without format detection.

## Plotting

`extrascripts/visualization/plot_trajectory_runtime_by_size.py` reads trajectories,
arrivals and runtimes from checkpoints, including mixed legacy/compact folders.
It ignores stale CSVs. Medians, IQRs, pairing and plotting time grids are unchanged.
The older `plot_main_clean_independent_rng.py` and original benchmark runner keep
their existing CSV workflow; this change does not remove their output files.
Their legacy readers are not migrated to version 3: do not point those readers
directly at new compact custom-grid checkpoints. Use the updated size plotter
or the shared `checkpoint_io` reader for these files.

## Simulation invariants

The custom runner asks MicroMacro for change-point logs and resamples those
directly on the original reporting grid. The regular Orchestrator API still
returns dense logs by default. Midpoint probes omit only their unused event-list
allocations; all probe events are still applied. The scheduler, hazards, RNG
ownership/state transitions, repeated imports, placement and cache policy are
unchanged. Compact output is opt-in in the shared benchmark task function, so
other callers keep their old output shape. Runtime measurements naturally change.

## Reusable verification

```bash
.venv/bin/python3 extrascripts/validation/check_memory_optimizations.py
.venv/bin/python3 extrascripts/validation/check_checkpoint_optimizations.py
```

Both use fixed saved small networks instead of generating new networks. The
second check compares the compact/sparse path against the same immutable
reference hashes, including events, per-community and macro RNG states, final
states and restored output values. It also checks non-integral reporting times,
mixed-format plotting inputs, ignored stale CSVs, two spawned writing workers,
atomic-write failure, and resume without CSV consolidation. Temporary IO test
files are removed; experiment outputs are not modified.

An optional real-data lossless round trip is available without touching source:

```bash
.venv/bin/python3 extrascripts/validation/check_checkpoint_optimizations.py \
  --existing-checkpoint output/data/<scenario>/checkpoints/rep_0001.json.gz
```
