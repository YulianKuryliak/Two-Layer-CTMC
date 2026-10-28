#!/usr/bin/env python3
"""Collect the current IEEE article's saved evidence into a new publication folder.

No simulator, experiment runner, plotting process, bootstrap or permutation test
is executed. Original output folders are read-only inputs. Existing releases are
never overwritten. Large raw gzip checkpoints are stored in uncompressed tar
containers and in ZIP64 without redundant recompression.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
ARTICLE = ROOT / 'paper/IEEE_version'
CHAIN = ROOT / 'output/chain_twoscale_residual_midpoint_20261007'
ANALYSIS = ROOT / 'output/analysis/chain_corrected_figures_20261008'
LEAF = ROOT / 'output/data/n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial'
CLIQUE = ROOT / 'output/main_clean_benchmarks_independent_rng/.checkpoints/clique_leaf_n100_k2'
SOURCE_SUMMARY = ROOT / 'output/analysis/n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial/figures/aspl_export_summary.csv'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def files_under(root):
    # Deliberately follow the leaf experiment's assembled scenario/graph links.
    for current, dirs, names in os.walk(root, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for name in sorted(names):
            path = Path(current) / name
            if path.is_file() and not name.endswith(('.tmp', '.pyc')):
                yield path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


README = '''# Supporting data for the two-scale community-network study

This release supports the manuscript *Two-scale stochastic simulation of epidemics
on community networks with aggregate inter-community coupling*, by Yulian Kuryliak
and Michael T. M. Emmerich. It was assembled from the IEEE-format manuscript and
the saved experimental outputs on 2026-10-22. It does not claim that the article
has been accepted or assigned a DOI.

## Start here

- `data/` contains the observations and saved statistical summaries in CSV/JSON.
- `plot_inputs/` contains all 37 numeric tables read by the manuscript's figures.
  These preserve the original precision, including ECDF observations outside the
  displayed axis ranges. `figure_sources/` contains the seven actual figure blocks.
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
'''

DICTIONARY = '''# Data dictionary

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

All `.dat` files are whitespace-delimited numeric tables with a header. ECDF
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
'''


class Collector:
    def __init__(self, out):
        self.out = out
        self.raw = []
        self.copied = []
        self.leaf_rows = []
        self.clique_ids = set()
        self.chain_keys = set()
        self.chain_samples = set()

    def copy(self, source, target):
        source = Path(source)
        target = self.out / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = digest(source)
        assert digest(target) == expected
        self.copied.append(dict(source_path=str(source.relative_to(ROOT)),
                                file=str(target.relative_to(self.out)), sha256=expected,
                                bytes=target.stat().st_size))

    def archive(self, name, entries, callback=None):
        path = self.out / 'raw' / f'{name}.tar'
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = list(entries)
        total = 0
        seen = set()
        with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as archive:
            for i, (source, member) in enumerate(entries, 1):
                assert member not in seen
                seen.add(member)
                data = source.read_bytes()
                h = hashlib.sha256(data).hexdigest()
                info = tarfile.TarInfo(member)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
                self.raw.append(dict(archive=str(path.relative_to(self.out)), member=member,
                                     source_path=str(source.relative_to(ROOT)),
                                     resolved_source_path=str(source.resolve().relative_to(ROOT)),
                                     bytes=len(data), sha256=h))
                total += len(data)
                if callback:
                    callback(source, member, data)
                if i % 3000 == 0 or i == len(entries):
                    print(f'{name}: {i}/{len(entries)} files; {total/2**20:.1f} MiB', flush=True)

    def source_leaf(self, source, member, data):
        if '/checkpoints/' not in member or not member.endswith('.json.gz'):
            return
        payload = json.loads(gzip.decompress(data))
        task = payload['task']
        runs = {r['simulator']: r for r in payload['runs']}
        full, mm = runs['Full Micro'], runs['Production MicroMacro + independent RNG']
        seeds = payload['seeds']
        self.leaf_rows.append(dict(target_aspl=task['target_ASPL'], k=task['k'],
                                  lineage_id=task['lineage_id'], interface_id=task['interface_id'],
                                  event_id=task['event_id'], actual_aspl=task['actual_ASPL'],
                                  initial_node=task['initial_node'],
                                  bridges_json=json.dumps(task['bridges'], separators=(',', ':')),
                                  graph_path=task['graph_path'],
                                  full_event_seed=seeds['full_event_seed'],
                                  mm_event_seed=seeds['micromacro_event_seed'],
                                  full_export=full['T_export'], mm_export=mm['T_export'],
                                  full_bridge=full['T_bridge_first'], mm_bridge=mm['T_bridge_first'],
                                  full_bypass=full['bypass'], mm_bypass=mm['bypass'],
                                  full_observed=full['export_observed'], mm_observed=mm['export_observed'],
                                  raw_member=member))

    def clique_leaf(self, source, member, data):
        payload = json.loads(gzip.decompress(data))
        assert payload['benchmark'] == 'clique_leaf_n100_k2'
        rep = payload['replication_id']
        assert rep not in self.clique_ids
        self.clique_ids.add(rep)
        assert len(payload['benchmark_observables']) == 2
        assert all(r['completion'] for r in payload['runs'])
        assert payload['benchmark_config']['store_trajectories'] is False

    def chain(self, source, member, data):
        if not member.startswith('checkpoints/'):
            return
        match = re.search(r'__n(\d+)__.*__aspl(\d+)__k(\d+)__.*network(\d+)_replicate(\d+)', member)
        assert match, member
        key = tuple(int(x) for x in match.groups())
        assert key not in self.chain_keys
        self.chain_keys.add(key)
        cell = key[:3]
        # Read a complete checkpoint per cell and both known long-horizon records.
        extended = key[0] == 50 and key[1] == 6 and key[2] == 1
        if cell not in self.chain_samples or extended:
            payload = json.loads(gzip.decompress(data))
            assert payload['status'] == 'complete' and len(payload['runs']) == 2
            assert all(len(r['arrivals']) == 10 for r in payload['runs'])
            assert all('infection_events' in r for r in payload['runs'])
            self.chain_samples.add(cell)


def make_table3(chain):
    result = []
    for n in (50, 100, 250, 1000):
        for k in (1, 5):
            rows = chain[chain.n.eq(n) & chain.aspl.eq(1) & chain.k.eq(k)]
            p = (n-k)/n
            exact = 0.
            for m in range(1, n-k+1):
                exact += p / (.003*m*(n-m))
                p *= (n-k-m)/(n-m)
            delay = rows.full_T1-rows.full_TB0
            result.append(dict(n=n, k=k, pairs=len(rows), beta=.003,
                               exact_bridge_mean=exact, full_bridge_mean=rows.full_TB0.mean(),
                               full_bridge_sem=rows.full_TB0.sem(),
                               exact_wait_mean=1/.003 if k == 1 else np.nan,
                               full_wait_mean=delay.mean() if k == 1 else np.nan,
                               full_wait_sem=delay.sem() if k == 1 else np.nan))
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    c = Collector(out)
    print(f'Building {out}', flush=True)
    for directory in ('data/clique_leaf', 'data/source_leaf', 'data/chains', 'data/tables', 'provenance', 'scripts'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    c.copy(ROOT / 'LICENSE', 'LICENSE.txt')
    assert (out / 'LICENSE.txt').read_text().strip() == 'All rights reserved.'
    for name in ('main.tex', 'appendices.tex', 'main.pdf'):
        c.copy(ARTICLE / name, 'article/' + name)
    main_text = (ARTICLE / 'main.tex').read_text()
    figure_rows = []
    for i, match in enumerate(re.finditer(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', main_text, re.S), 1):
        label = re.search(r'\\label\{([^}]+)\}', match[0])[1]
        dest = out / 'figure_sources' / f'figure_{i:02}.tex'
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(match[0] + '\n')
        figure_rows.append(dict(item=f'Figure {i}', label=label,
                                plot_inputs=';'.join(sorted(set(re.findall(r'figures_pgfplots/([^}]+\.dat)', match[0])))),
                                supporting_data=['conceptual schematic; no empirical dataset',
                                                 'data/clique_leaf', 'data/source_leaf', 'data/chains',
                                                 'data/chains/median_layer_metrics.csv',
                                                 'data/chains/cell_summary.csv',
                                                 'data/chains/cell_summary.csv'][i-1]))
    assert len(figure_rows) == 7
    all_tables = main_text + (ARTICLE / 'appendices.tex').read_text()
    for i, match in enumerate(re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', all_tables, re.S), 1):
        (out / 'figure_sources' / f'table_{i:02}.tex').write_text(match[0] + '\n')
    style = main_text[main_text.index('% BEGIN EDITABLE PLOT STYLE'):main_text.index('% END EDITABLE PLOT STYLE')]
    (out / 'figure_sources/shared_plot_style.tex').write_text(style + '\n')
    for path in sorted((ARTICLE / 'figures_pgfplots').glob('*.dat')):
        c.copy(path, 'plot_inputs/' + path.name)
    assert len(list((out / 'plot_inputs').glob('*.dat'))) == 37
    for name in ('leaf_manifest.json', 'chain_manifest.json', 'pdftex_ecdf_manifest.json'):
        c.copy(ROOT / 'paper/SMPT paper/figures_pgfplots' / name, 'provenance/' + name)
    obs = pd.read_csv(ROOT / 'output/main_clean_benchmarks_independent_rng/benchmark_observables.csv')
    obs = obs[obs.benchmark.eq('clique_leaf_n100_k2')]
    obs.to_csv(out / 'data/clique_leaf/observations.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    c.copy(ROOT / 'output/analysis/clique_leaf/summary_metrics.json', 'data/clique_leaf/summary_metrics.json')
    c.copy(ROOT / 'output/main_clean_benchmarks_independent_rng/benchmark_manifest.json', 'provenance/clique_leaf_parent_manifest.json')
    c.copy(SOURCE_SUMMARY, 'data/source_leaf/cell_summary.csv')
    for name in ('cell_summary.csv', 'per_run.csv.gz', 'layer_metrics.csv', 'median_layer_metrics.csv', 'terminal_metrics.csv'):
        c.copy(ANALYSIS / name, 'data/chains/' + name)
    for path in (ANALYSIS / 'trajectory_cdf_cache').glob('*.npz'):
        c.copy(path, 'data/chains/trajectory_cdf_cache/' + path.name)
    c.copy(ANALYSIS / 'analysis_manifest.json', 'provenance/chain_analysis_manifest.json')
    c.copy(ANALYSIS / 'figures/figure_manifest.json', 'provenance/chain_figure_manifest.json')
    scripts = ['analysis/analyze_corrected_chain_saved.py', 'analysis/smpt_twoscale_paper_metrics.py',
               'visualization/export_leaf_pgfplots.py', 'visualization/export_chain_pgfplots.py',
               'visualization/plot_n1000_aspl_export.py', 'visualization/plot_clique_leaf_results.py',
               'visualization/smpt_twoscale_paper_figures.py', 'visualization/plot_l10_aspl_chain_random_initial.py',
               'visualization/plot_l10_aspl_chain_median_heatmap.py', 'visualization/plot_l10_aspl_chain_size_heatmap.py',
               'visualization/plot_l10_aspl_chain_trajectory_cdf.py']
    for name in scripts:
        frozen = ANALYSIS / 'sources/extrascripts' / name
        c.copy(frozen if frozen.is_file() else ROOT / 'extrascripts' / name, 'analysis_sources/extrascripts/' + name)
    for name, target in [('verify_ieee_article_dataset.py', 'verify_dataset.py'),
                         ('reproduce_ieee_article_tables.py', 'reproduce_tables.py'),
                         ('package_ieee_article_data.py', 'build_release.py')]:
        c.copy(ROOT / 'extrascripts/analysis' / name, 'scripts/' + target)

    c.archive('clique_leaf', ((p, 'checkpoints/' + p.name) for p in sorted(CLIQUE.glob('*.json.gz'))), c.clique_leaf)
    assert c.clique_ids == set(range(1, 10001))
    leaf_entries = [(p, str(p.relative_to(LEAF))) for p in files_under(LEAF)
                    if p.name.endswith(('.json', '.json.gz', '.md'))]
    c.archive('source_leaf', leaf_entries, c.source_leaf)
    source = pd.DataFrame(c.leaf_rows).sort_values(['target_aspl', 'k', 'lineage_id', 'interface_id', 'event_id'])
    source.to_csv(out / 'data/source_leaf/paired_runs.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    chain_entries = []
    for name in ('manifest.json', 'tasks.json.gz', 'run_status.json'):
        chain_entries.append((CHAIN / name, name))
    for folder in ('checkpoints', 'inputs', 'sources'):
        chain_entries.extend((p, str(p.relative_to(CHAIN))) for p in files_under(CHAIN / folder))
    c.archive('chains', chain_entries, c.chain)
    chain = pd.read_csv(out / 'data/chains/per_run.csv.gz')
    expected_keys = set(chain[['n', 'aspl', 'k', 'network_id', 'replicate_id']].itertuples(index=False, name=None))
    assert c.chain_keys == expected_keys and len(c.chain_samples) == 48
    make_table3(chain).to_csv(out / 'data/tables/table_3_clique_verification.csv', index=False)

    raw = pd.DataFrame(c.raw)
    raw.to_csv(out / 'provenance/raw_inventory.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(c.copied).to_csv(out / 'provenance/source_files.csv', index=False)
    aliases = {row['source_path']: row['archive'] + '::' + row['member'] for row in c.raw}
    aliases.update({row['resolved_source_path']: row['archive'] + '::' + row['member'] for row in c.raw})
    original = json.loads((CHAIN / 'manifest.json').read_text())
    for graph in original['graphs']:
        aliases[graph['source']] = 'raw/chains.tar::' + graph['snapshot']
    with gzip.GzipFile(out / 'provenance/path_aliases.json.gz', 'wb', mtime=0) as stream:
        stream.write(json.dumps(aliases, sort_keys=True).encode())
    assert all(p in aliases for p in source.graph_path.unique())
    figure_rows.extend([dict(item='Table 1', label='tab:information', plot_inputs='', supporting_data='figure_sources/table_01.tex; model definitions'),
                        dict(item='Table 2', label='tab:design', plot_inputs='', supporting_data='figure_sources/table_02.tex; metadata.json'),
                        dict(item='Table 3 / Appendix A', label='tab:verification', plot_inputs='', supporting_data='data/tables/table_3_clique_verification.csv; data/chains/per_run.csv.gz'),
                        dict(item='Appendix B', label='app:floor', plot_inputs='', supporting_data='analytical derivation in article/appendices.tex; saved chain permutation floors in data/chains/cell_summary.csv'),
                        dict(item='Appendix C', label='app:repro', plot_inputs='', supporting_data='data/chains/per_run.csv.gz; raw/chains.tar')])
    pd.DataFrame(figure_rows).to_csv(out / 'FIGURE_DATA_MAP.csv', index=False)
    (out / 'README.md').write_text(README)
    (out / 'DATA_DICTIONARY.md').write_text(DICTIONARY)
    (out / 'requirements.txt').write_text('numpy\npandas\n')
    (out / 'CITATION.cff').write_text('''cff-version: 1.2.0
message: "Cite this dataset together with the associated article; add the deposit DOI when assigned."
type: dataset
title: "Supporting data: Two-scale stochastic simulation of epidemics on community networks with aggregate inter-community coupling"
authors:
  - family-names: Kuryliak
    given-names: Yulian
  - family-names: Emmerich
    given-names: Michael T. M.
date-released: 2026-10-22
version: "2026-10-22"
''')
    schemas = {}
    for path in sorted((out / 'data').rglob('*.csv*')):
        frame = pd.read_csv(path)
        schemas[str(path.relative_to(out))] = dict(rows=len(frame), columns=[dict(name=col, dtype=str(frame[col].dtype),
                                                                               missing=int(frame[col].isna().sum())) for col in frame])
    json_write(out / 'table_schemas.json', schemas)
    json_write(out / 'metadata.json', dict(title='Supporting data for the two-scale community-network study',
               created_utc=datetime.now(timezone.utc).isoformat(), article_sha256=digest(ARTICLE / 'main.tex'),
               authors=['Yulian Kuryliak', 'Michael T. M. Emmerich'], licence='All rights reserved',
               deposition_status='local preparation only; no DOI or public deposit assigned',
               clique_leaf=dict(pairs=10000, n=100, k=2, beta=.005, horizon=1000),
               source_leaf=dict(cells=42, pairs=16800, n=1000, beta=.003, horizon=30000),
               chains=dict(cells=48, pairs=19200, L=10, sizes=[50,100,250,1000], beta=.003,
                           common_horizon=30000, extended_full_runs=2, extended_full_horizon=40000),
               raw_members=len(c.raw), raw_archives=['raw/clique_leaf.tar','raw/source_leaf.tar','raw/chains.tar'],
               source_copy_hashes_verified=True, raw_symlinks_materialised=True, simulations_run=False))
    cells = [dict(cell_type='markdown', metadata={}, source=['# Inspect the article data\n',
              'Run these cells from this release folder. No simulation is executed.'])]
    code_cells = ["from pathlib import Path\nimport importlib.util\nimport pandas as pd\nROOT = Path.cwd()\nspec = importlib.util.spec_from_file_location('check', ROOT/'scripts/verify_dataset.py')\ncheck = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(check)\ncheck.check_data(ROOT)",
                  "pd.read_csv(ROOT/'data/source_leaf/cell_summary.csv').head()",
                  "pd.read_csv(ROOT/'data/chains/cell_summary.csv')[['n','aspl','k','rel_w1_pct','median_shift_pct']].head()",
                  "pd.read_csv(ROOT/'data/tables/table_3_clique_verification.csv')"]
    cells.extend(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=s.splitlines(keepends=True)) for s in code_cells)
    for index, cell in enumerate(cells):
        cell['id'] = f'cell-{index}'
    json_write(out / 'inspect_data.ipynb', dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, nbformat=4, nbformat_minor=5))
    verifier = load_module(out / 'scripts/verify_dataset.py', 'release_verifier')
    validation = verifier.check_data(out)
    validation.update(raw_members=len(c.raw), raw_checkpoint_counts=Counter(row['archive'] for row in c.raw if '/checkpoints/' in row['source_path'] or '/.checkpoints/' in row['source_path']),
                      raw_chain_cells_sampled=len(c.chain_samples),
                      raw_source_hashes_verified_while_writing=True)
    json_write(out / 'validation.json', validation)
    reproduce = load_module(out / 'scripts/reproduce_tables.py', 'release_point_estimates')
    _, _, claims = reproduce.point_estimates(out)
    json_write(out / 'article_numbers.json', claims)
    # Hash every ordinary release file and the tar containers. SHA256SUMS excludes
    # itself to avoid a recursive checksum dependency.
    all_files = sorted(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes = [(digest(p), str(p.relative_to(out))) for p in all_files]
    (out / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for sha, name in hashes))
    print(f'Dataset complete: {len(all_files)+1} files; {sum(p.stat().st_size for p in all_files)/2**30:.3f} GiB', flush=True)
    print('Verifying every raw archive member...', flush=True)
    full_audit = verifier.check_hashes(out, full=True)
    print(json.dumps(full_audit), flush=True)
    json_write(out / 'full_verification.json', dict(status='passed', **full_audit,
               completed_utc=datetime.now(timezone.utc).isoformat()))
    with (out / 'SHA256SUMS').open('a') as stream:
        stream.write(digest(out / 'full_verification.json') + '  full_verification.json\n')
    zpath = out.with_suffix('.zip')
    assert not zpath.exists()
    with zipfile.ZipFile(zpath, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                archive.write(p, str(Path(out.name) / p.relative_to(out)),
                              compress_type=zipfile.ZIP_STORED if p.suffix == '.tar' else zipfile.ZIP_DEFLATED)
    print(f'ZIP64 archive ready: {zpath} ({zpath.stat().st_size/2**30:.3f} GiB)', flush=True)


if __name__ == '__main__':
    main()
