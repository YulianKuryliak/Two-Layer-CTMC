#!/usr/bin/env python3
"""Check the publication dataset from saved files; never import a simulator."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_data(root):
    root = Path(root)
    leaf = pd.read_csv(root / 'data/clique_leaf/observations.csv.gz')
    key = ['replication_id', 'simulator']
    assert len(leaf) == 20000 and not leaf.duplicated(key).any()
    assert leaf.groupby('simulator').size().tolist() == [10000, 10000]
    full_name = 'Full Micro'
    mm_name = 'Production MicroMacro + independent RNG'
    full = leaf[leaf.simulator.eq(full_name)].sort_values('replication_id')
    mm = leaf[leaf.simulator.eq(mm_name)].sort_values('replication_id')
    np.testing.assert_array_equal(full.replication_id, mm.replication_id)
    assert np.isfinite(leaf[['bridge_infection_time', 'bridge_to_export_delay', 'total_export_time']]).all().all()
    np.testing.assert_allclose(leaf.total_export_time - leaf.bridge_infection_time,
                               leaf.bridge_to_export_delay, atol=1e-10)
    assert (full.bridge_to_export_delay < 0).sum() == 0
    assert (mm.bridge_to_export_delay < 0).sum() == 98
    metrics = json.loads((root / 'data/clique_leaf/summary_metrics.json').read_text())
    np.testing.assert_allclose(full.total_export_time.mean(), metrics['mean_export_full'], rtol=1e-10)
    np.testing.assert_allclose(mm.total_export_time.mean(), metrics['mean_export_micromacro'], rtol=1e-10)
    plots = root / 'plot_inputs'
    for stem, column in [('export', 'total_export_time'), ('signed_delay', 'bridge_to_export_delay')]:
        for label, frame in [('full', full), ('twoscale', mm)]:
            curve = pd.read_csv(plots / f'leaf_clique_{stem}_{label}.dat', sep=r'\s+')
            assert len(curve) == 10001
            np.testing.assert_allclose(curve.x.iloc[1:], np.sort(frame[column]), rtol=2e-14, atol=1e-12)
            np.testing.assert_allclose(curve.y.iloc[1:], np.arange(1, 10001) / 10000)

    source = pd.read_csv(root / 'data/source_leaf/paired_runs.csv.gz')
    skey = ['target_aspl', 'k', 'lineage_id', 'interface_id', 'event_id']
    assert len(source) == 16800 and not source.duplicated(skey).any()
    expected = {(float(a), k) for a in np.arange(1, 6.01, .25) for k in (1, 5)}
    assert set(source.groupby(['target_aspl', 'k']).groups) == expected
    saved = pd.read_csv(root / 'data/source_leaf/cell_summary.csv').set_index(['target_aspl', 'k'])
    assert len(saved) == 42 and set(saved.index) == expected
    assert source.full_observed.all() and source.mm_observed.all()
    assert source.full_bypass.sum() == 0
    for key, rows in source.groupby(['target_aspl', 'k']):
        assert len(rows) == 400 and rows.lineage_id.nunique() == 40
        assert rows.groupby('lineage_id').size().eq(10).all()
        reference = saved.loc[key]
        w1 = np.abs(np.sort(rows.full_export) - np.sort(rows.mm_export)).mean()
        for field, value in [('full_export_mean', rows.full_export.mean()),
                             ('mm_export_mean', rows.mm_export.mean()),
                             ('relative_w1_pct', 100 * w1 / rows.full_export.mean()),
                             ('mm_bypass_pct', 100 * rows.mm_bypass.mean())]:
            np.testing.assert_allclose(value, reference[field], rtol=1e-10, atol=1e-10)
    for k in (1, 5):
        table = pd.read_csv(plots / f'leaf_source_sweep_k{k}.dat', sep=r'\s+')
        reference = saved.xs(k, level='k').sort_index()
        np.testing.assert_allclose(table.aspl, reference.index)
        for field, ref in [('full_mean', 'full_export_mean'), ('twoscale_mean', 'mm_export_mean'),
                           ('w1', 'relative_w1_pct'), ('bypass', 'mm_bypass_pct')]:
            np.testing.assert_allclose(table[field], reference[ref], rtol=2e-14, atol=1e-12)

    chain = pd.read_csv(root / 'data/chains/per_run.csv.gz')
    ckey = ['n', 'aspl', 'k', 'network_id', 'replicate_id']
    assert len(chain) == 19200 and not chain.duplicated(ckey).any()
    expected = {(n, a, k) for n in (50, 100, 250, 1000) for a in range(1, 7) for k in (1, 5)}
    assert set(chain.groupby(['n', 'aspl', 'k']).groups) == expected
    assert chain.groupby(['n', 'aspl', 'k']).size().eq(400).all()
    for model in ('full', 'mm'):
        times = chain[[f'{model}_T{i}' for i in range(10)]].to_numpy()
        assert np.isfinite(times).all() and (np.diff(times, axis=1) >= 0).all()
        assert (chain[f'{model}_T9'] <= chain[f'{model}_horizon']).all()
    assert chain.extended.sum() == 2
    assert chain.full_horizon.value_counts().to_dict() == {30000.0: 19198, 40000.0: 2}
    assert chain.mm_horizon.eq(30000).all()
    assert chain.full_T9.gt(30000).sum() == 2 and not chain.mm_T9.gt(30000).any()
    summary = pd.read_csv(root / 'data/chains/cell_summary.csv').set_index(['n', 'aspl', 'k'])
    assert len(summary) == 48 and set(summary.index) == expected
    for key, rows in chain.groupby(['n', 'aspl', 'k']):
        reference = summary.loc[key]
        w1 = np.abs(np.sort(rows.full_T9) - np.sort(rows.mm_T9)).mean()
        np.testing.assert_allclose(100 * w1 / rows.full_T9.mean(), reference.rel_w1_pct, rtol=1e-10)
        shift = 100 * (rows.mm_T9.median() - rows.full_T9.median()) / rows.full_T9.median()
        np.testing.assert_allclose(shift, reference.median_shift_pct, atol=1e-10)
    for k in (1, 5):
        for metric, col in [('median', 'median_shift_pct'), ('w1', 'rel_w1_pct')]:
            table = pd.read_csv(plots / f'accuracy_k{k}_{metric}.dat', sep=r'\s+')
            for row in table.itertuples():
                key = ((50, 100, 250, 1000)[int(row.x)], int(row.y) + 1, k)
                np.testing.assert_allclose(row.value, summary.loc[key, col], rtol=2e-14)
    for n in (50, 100, 250, 1000):
        for k in (1, 5):
            table = pd.read_csv(plots / f'descriptor_n{n}_k{k}.dat', sep=r'\s+')
            reference = summary.xs((n, k), level=('n', 'k')).sort_values('sigma_full')
            np.testing.assert_allclose(table.plotrho, .003 * reference.sigma_full, rtol=2e-14)
    layers = pd.read_csv(root / 'data/chains/median_layer_metrics.csv').set_index(['target_ASPL', 'k', 'community'])
    for k in (1, 5):
        for metric, col in [('median', 'signed_relative_median_shift_percent'), ('w1', 'relative_w1_percent')]:
            table = pd.read_csv(plots / f'chain_heatmap_k{k}_{metric}.dat', sep=r'\s+')
            assert len(table) == 54
            for row in table.itertuples():
                np.testing.assert_allclose(row.value, layers.loc[(int(row.y) + 1, k, int(row.x) + 1), col], rtol=2e-14)
    verified = pd.read_csv(root / 'data/tables/table_3_clique_verification.csv')
    assert len(verified) == 8
    for row in verified.itertuples():
        rows = chain[chain.n.eq(row.n) & chain.aspl.eq(1) & chain.k.eq(row.k)]
        np.testing.assert_allclose(rows.full_TB0.mean(), row.full_bridge_mean)
        np.testing.assert_allclose(rows.full_TB0.sem(), row.full_bridge_sem)
        if row.k == 1:
            delay = rows.full_T1 - rows.full_TB0
            np.testing.assert_allclose(delay.mean(), row.full_wait_mean)
            np.testing.assert_allclose(delay.sem(), row.full_wait_sem)
    with np.load(root / 'data/chains/trajectory_cdf_cache/aspl1_k1.npz', allow_pickle=False) as cache:
        assert int(cache['checkpoint_count']) == 400
        for model in ('full', 'mm'):
            table = pd.read_csv(plots / f'chain_{model}_trajectory.dat', sep=r'\s+')
            visible = cache['grid'] <= 6000
            np.testing.assert_allclose(table.time, cache['grid'][visible])
            np.testing.assert_allclose(table[['lower', 'median', 'upper']].to_numpy().T,
                                       cache[f'{model}_quantiles'][:, visible])
            cdf = pd.read_csv(plots / f'chain_{model}_cdf.dat', sep=r'\s+')
            np.testing.assert_allclose(cdf.time.iloc[1:], np.sort(cache[f'{model}_arrivals']))
    return {'clique_leaf_observations': len(leaf), 'clique_leaf_pairs': 10000,
            'source_leaf_pairs': len(source), 'source_leaf_cells': 42,
            'chain_pairs': len(chain), 'chain_cells': 48, 'table_3_rows': 8,
            'common_30000_completed': {'Full': 19198, 'TwoScale': 19200},
            'extended_full_runs': 2, 'plot_tables_checked': 37,
            'simulations_run': False, 'bootstrap_or_permutation_recomputed': False}


def check_hashes(root, full=False):
    root = Path(root)
    checked = 0
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = root / relative
        if path.suffix == '.tar' and not full:
            continue
        assert sha256(path) == expected, relative
        checked += 1
    raw_count = 0
    if full:
        with gzip.open(root / 'provenance/raw_inventory.csv.gz', 'rt') as stream:
            entries = list(csv.DictReader(stream))
        by_archive = {}
        for row in entries:
            by_archive.setdefault(row['archive'], {})[row['member']] = row
        for archive, inventory in by_archive.items():
            found = set()
            with tarfile.open(root / archive, 'r|') as stream:
                for member in stream:
                    assert member.isfile() and member.name in inventory
                    row = inventory[member.name]
                    data = stream.extractfile(member).read()
                    assert len(data) == int(row['bytes'])
                    assert hashlib.sha256(data).hexdigest() == row['sha256']
                    assert member.name not in found
                    found.add(member.name)
            assert found == set(inventory), archive
            raw_count += len(found)
    return {'release_files_hashed': checked, 'raw_members_hashed': raw_count,
            'raw_archive_hashes_checked': full}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--full', action='store_true', help='also hash all large raw archives and each member')
    args = parser.parse_args()
    report = check_data(args.dataset)
    report.update(check_hashes(args.dataset, args.full))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
