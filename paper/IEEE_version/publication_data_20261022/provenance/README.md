# Reading the frozen provenance records

The release coverage is established by `../validation.json`, the 46,000 paired
checkpoint records, and the compact tables. Historical metadata are preserved
verbatim, including status labels written at different stages:

- The chain design `manifest.json` inside `raw/chains.tar` says `prepared` because
  it was frozen before execution. Its companion `run_status.json` says `complete`
  and `full_design_complete=true`. The release contains all 19,200 chain pairs.
- `clique_leaf_parent_manifest.json` describes a broader 16-scenario benchmark
  collection and retains the status `running`. This release selects only its
  completed clique–leaf scenario: 10,000 pairs, checked individually. The other
  scenarios and unfinished temporary files are outside this article's dataset.
- The source–leaf experiment originally assembled low- and high-ASPL directories
  with symbolic links. `raw/source_leaf.tar` stores ordinary bytes for every
  selected graph, checkpoint and configuration; the links themselves are not
  required after download. Logical and resolved source paths are both recorded.
- Original absolute machine paths or historical baseline references are provenance
  identifiers. Use archive/member addresses and `path_aliases.json.gz` for portable
  access. Chain checkpoints embed both models' results; historical baseline files
  are not required to inspect those results or reproduce the compact-table checks.
- Original analysis/export scripts are preserved as source snapshots. Their default
  directories refer to the original repository. The portable scripts at the release
  root's `scripts/` directory read this dataset without importing either simulator.
- Saved bootstrap/permutation outputs are unchanged. Portable verification repeats
  point calculations and checks file integrity; it does not generate new bootstrap
  samples or new epidemic trajectories. Full seed-level reruns also require the
  appropriate historical code/environment and are not claimed by a checksum pass.

The article sources and PDF are snapshots of the manuscript at collection time.
This dataset preparation did not edit the working manuscript or publish a DOI.
