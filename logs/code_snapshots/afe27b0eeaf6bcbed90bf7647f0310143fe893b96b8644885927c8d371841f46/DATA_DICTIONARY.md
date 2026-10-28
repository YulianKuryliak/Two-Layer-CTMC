# Data dictionary

All times are nonnegative continuous simulation times unless a signed delay/shift
is explicitly indicated. Percentages use a scale of 0–100. Full is full-network
Gillespie simulation. `mm`/MicroMacro denotes the TwoScale approximation. SI has
no recovery (gamma=0). See `table_schemas.json` for every CSV column, inferred type,
row count and missing-value count.

## Run-level measurements

`data/clique_leaf/observations.csv.gz`: one model observation per replication ID;
the key is (replication_id, simulator). Two rows form a matched pair. Export time
is first leaf infection; bridge infection time is the first infected real bridge;
bridge_to_export_delay = export minus first-bridge time. A negative delay denotes
a bridge bypass. It is impossible in Full and occurs in 98 TwoScale records.

`data/source_leaf/paired_runs.csv.gz`: one matched pair per
(target_aspl, k, lineage_id, interface_id, event_id). `full_export`/`mm_export` are
leaf infection times; `full_bridge`/`mm_bridge` are first-bridge times; `_observed`
flags indicate measured exports. `_bypass` means export before first-bridge
infection. `actual_aspl` is the measured graph ASPL. `initial_node` and `bridges_json`
record actual placements. Event seeds are stored separately for the two models.
`raw_member` identifies the original record inside raw/source_leaf.tar; graph_path
is resolved using provenance/path_aliases.json.gz. `target_aspl` is a design target,
not an exact measured ASPL. Replicates sharing a lineage are not independent graphs.

`data/chains/per_run.csv.gz`: one matched pair per
(n, aspl, k, network_id, replicate_id). `n` is community size; there are ten
communities. `aspl` is the integer target ASPL. `mean_degree` is the design degree.
`full_T0`...`full_T9` and `mm_T0`...`mm_T9` are first-arrival times for communities
0...9. `full_TB0`...`full_TB8` and `mm_TB0`...`mm_TB8` are first real source-bridge
infection times at the nine interfaces. `full_sd` averages the within-community
standard deviations of Full infection times over completed source communities.
`full_horizon`, `mm_horizon` and `extended` describe the actual observation windows.
`repeat_imports` and `repeat_imports_to_endpoint` are saved TwoScale diagnostics.
`*_runtime_s` are retained provenance measurements, not a controlled speed benchmark.

## Summaries and plot inputs

`cell_summary.csv` files have one row per design cell. `R`, `paired_runs`,
`checkpoint_count` and `pairs_available` are counts. Source-leaf completion columns
are percentages; chain observed/completion counts are explicitly named. `mean_*`
and `median_*` give arrival or bridge-time means/medians. `_ci_low`/`_ci_high` are
saved 95% bootstrap bounds, not trajectory IQR. `_sem` denotes standard error of
the mean (sample SD divided by square root of the sample count).

`rel_w1_pct`, `relative_w1_pct`, `relative_w1_percent`, `e_W` and plot `w1` all use
100 * mean(abs(sorted(TwoScale times)-sorted(Full times))) / mean(Full times).
`median_shift_pct` and `signed_relative_median_shift_percent` use
100 * (median(TwoScale)-median(Full)) / median(Full). A negative sign means earlier
TwoScale arrival. `signed_relative_bias_pct` and `mean_shift_pct` similarly compare
means. `first_cross_*` concerns the first crossing; `later_cross_*` concerns the
duration from the first to the ninth crossing; `later_minus_first_*` is their
contrast. `_perm_p` is the saved label-swap permutation p-value. `_null95_pct`
is the saved 95th percentile of the permutation statistic; `_null_median_pct`
is its median. No multiplicity adjustment was applied to the cellwise tests.

`sigma_full` is the saved average within-community timing SD. For the article use
rho = .003*sigma_full, stored as `plotrho` in descriptor tables. Alternative saved
`rho`/`rho_sigma` fields are preserved for provenance and are not the plotted rho.
`mean_access_full` is mean delay from community arrival to its first infected bridge.
`*_chain_bypass_pct` averages bridge-bypass indicators over the nine crossings.

Layer summaries concern n=1,000 only: one row per (target_ASPL, k, community),
community=1...9. `local_delta_relative_mean_shift_percent` concerns the transition
duration between adjacent communities; other layer shifts concern cumulative
arrival from time zero. Terminal summaries cover all 48 cells at community 9.

Most `.dat` files are whitespace-delimited numeric tables with a header.
The `_path.dat` files instead have a comment followed by two unnamed numeric columns. ECDF
`x`/`y` are time/probability with an explicit zero-probability anchor. `_path.dat`
contains the same vertices transformed to axis coordinates for streamed TeX
rendering. Heatmaps use zero-based x/y grid indices, `value` for the displayed
metric, and `hatch=1` only for the Wasserstein cells whose saved p-value is >=.05.
Descriptor `errlo`/`errhi` are distances to confidence limits, not endpoints.

## Raw archives and provenance

Tar member bytes match the source SHA-256 records. Gzipped JSON checkpoints can
be read with Python gzip/json; graph snapshots specify nodes and edges. Chain
checkpoints contain task, rerun_task, event seeds, comparison horizons and two
model runs, including node-level infection_events (time, community, node). Some
TwoScale records additionally retain typed events, aggregate change points and
import diagnostics. `None`/JSON null means unobserved; never replace it by T_max.
Source-leaf checkpoints store task, seeds, RNG policy and per-model export/bridge
measurements. Clique-leaf checkpoints store measured observables and per-run
network/event seeds; their empty trajectory arrays reflect the recorded policy.

`provenance/raw_inventory.csv.gz` gives archive, member, logical source_path,
resolved_source_path, bytes and sha256. The logical and resolved paths differ for
the assembled source-leaf dataset. These columns are provenance identifiers;
archive+member is the portable address. `source_files.csv` similarly documents
ordinary copied files. The release inventory and SHA256SUMS cover the large tar
containers as well as ordinary data/documents/scripts.
