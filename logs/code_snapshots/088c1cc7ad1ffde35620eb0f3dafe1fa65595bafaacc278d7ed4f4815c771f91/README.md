# Supporting data for the two-scale community-network study

This release supports the manuscript *Two-scale stochastic simulation of epidemics
on community networks with aggregate inter-community coupling*, by Yulian Kuryliak
and Michael T. M. Emmerich. It was assembled from the IEEE-format manuscript and
the saved experimental outputs on 2026-10-22. It does not claim that the article
has been accepted or assigned a DOI.

## Start here

- `data/` contains the observations and saved statistical summaries in CSV/JSON.
- `plot_inputs/` contains all 37 numeric tables read by the manuscript's figures.
  These preserve the original precision, including ECDF observations outside the
  displayed axis ranges. The four original time/probability ECDF tables are also
  included in `data/clique_leaf/ecdf/`. `figure_sources/` contains the seven actual figure blocks.
- `FIGURE_DATA_MAP.csv` connects every figure, table and appendix to these files.
- `raw/` contains three tar archives of original gzip/JSON checkpoints, graphs,
  seeds, parameters and available frozen code. The tar members are ordinary files;
  no symbolic links or external linked directories are required to read them.
- `provenance/raw_inventory.csv.gz` gives each raw member's source, byte count and
  SHA-256. `provenance/path_aliases.json.gz` maps graph/checkpoint source paths to
  locations inside this release. Original internal path strings remain provenance
  identifiers, not instructions to download anything.
- `DATA_DICTIONARY.md`, `table_schemas.json` and `metadata.json` explain the data.
- `article/` snapshots the exact manuscript sources and PDF used for this release.
  It identifies which manuscript version the data support; it is not a new paper.

## Experimental coverage

1. Clique–leaf: 10,000 matched pairs, 20,000 model observations; n=100, k=2,
   beta=0.005, SI, observation horizon 1,000. The saved configuration fixes the
   seed node at 0 and samples the bridge contacts. The raw checkpoints retain
   export/bridge measurements and network/event seeds. This benchmark did not
   save full node-event trajectories; the plotted data are export/bridge times.
2. Source–leaf: 42 cells, 400 pairs each = 16,800 pairs. n=1,000; ASPL 1–6 in
   steps of 0.25; k=1 or 5; beta=0.003; horizon 30,000. Each cell contains 40 graph
   lineages and 10 interface/event replicates per lineage. All measured exports
   were observed. The assembled experiment's linked directories were materialised.
3. Chains: 48 cells, 400 pairs each = 19,200 pairs. Ten communities; n=50,100,250,
   1,000; ASPL 1–6 in steps of 1; k=1 or 5; beta=0.003. The clique-chain example
   is the n=1,000, ASPL=1, k=1 cell, not 400 additional pairs. Raw checkpoints
   include both Full and TwoScale results, node infection events and seed metadata.
4. Appendix A/Table 3: eight ASPL=1 cells of the same chain dataset, not a separate
   experiment. The CSV includes exact expectations and simulated means/SEMs.
   The post-bridge waiting-time column applies only when k=1.

## Horizons, uncertainty and the descriptor

At the common horizon 30,000, terminal arrivals were observed in 19,198 Full and
19,200 TwoScale chain runs. Two Full runs (n=50, ASPL=6, k=1) were extended to
40,000; their actual observed arrival times enter unrestricted arrival summaries.
Never replace an unobserved arrival by the horizon. Both horizons are explicit
in `per_run.csv.gz` and the raw records. All 19,200 arrival pairs used in the
saved unrestricted chain summaries are observed with these recorded horizons.

In Figure 4, shading denotes trajectory interquartile ranges, not confidence
intervals. Saved bootstrap intervals and permutation results are copied without
resampling. The manuscript's descriptor is **rho = 0.003 * sigma_full**. The legacy
columns named `rho` and `rho_sigma` in the saved summary are alternative descriptors;
use the `plotrho` field in plot inputs or the documented formula for this paper.
Time and rates use the model's simulation units, not an assumed number of days.

## Verify and reproduce the reported numbers

Python 3, NumPy and pandas are sufficient for the release's portable checks:

```bash
python -m pip install -r requirements.txt
python scripts/verify_dataset.py .
python scripts/verify_dataset.py . --full
python scripts/reproduce_tables.py . --out reproduced_tables
```

The fast check verifies all compact file hashes, pairing/coverage, arrival order,
completion rules, means, Wasserstein distances, median shifts, Appendix A values,
and numeric plot projections. `--full` also reads every raw tar member and verifies
its checksum; it reads several GB. The final command creates a NEW output folder
with point estimates and key article numbers. It does not rerun simulations,
bootstrap confidence intervals or permutation tests. Saved intervals/tests and
their original analysis scripts are included for inspection. `inspect_data.ipynb`
provides a small companion notebook for exploring these same tables/checks.

To inspect raw records without extracting the whole archive:

```python
import tarfile, gzip, json
with tarfile.open('raw/chains.tar') as archive:
    member = next(m for m in archive if m.name.startswith('checkpoints/') and m.name.endswith('.json.gz'))
    record = json.loads(gzip.decompress(archive.extractfile(member).read()))
    print(record['comparison_horizons'])
```

`analysis_sources/` preserves original analysis/export scripts, with their original
repository paths. They are provenance snapshots: their default ROOT/output paths
target the original repository and should not be executed directly against this
release. Use the portable scripts above. The chain raw archive also includes its
frozen simulation source and graph snapshots. Historical baseline-file references
are audit identifiers; the Full results themselves are embedded in the released
paired checkpoints. Unused older model outputs, development comparisons, unrelated
benchmarks, temporary files and the unrelated multi-GB trajectory table are omitted.

## Rights and citation

**All rights reserved.** The authors explicitly requested retaining the project's
current conditions. See `LICENSE.txt`; no Creative Commons licence is granted by
this package. Publication of the files does not itself grant unrestricted reuse.
`CITATION.cff` contains draft dataset citation metadata. After depositing the data,
add the repository's assigned DOI and version to that citation record and to the
article's Data availability statement. This task only prepared local files; it did
not upload or publish them or promise an existing public data location.
