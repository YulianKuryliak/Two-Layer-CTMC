#!/usr/bin/env python3
"""Recover point estimates from the released observations, without simulations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def point_estimates(root):
    root = Path(root)
    source = pd.read_csv(root / 'data/source_leaf/paired_runs.csv.gz')
    source_rows = []
    for (aspl, k), rows in source.groupby(['target_aspl', 'k']):
        f, m = rows.full_export, rows.mm_export
        source_rows.append(dict(target_aspl=aspl, k=k, pairs=len(rows),
                                full_export_mean=f.mean(), mm_export_mean=m.mean(),
                                relative_w1_pct=100 * np.abs(np.sort(f) - np.sort(m)).mean() / f.mean(),
                                mm_bypass_pct=100 * rows.mm_bypass.mean()))
    chain = pd.read_csv(root / 'data/chains/per_run.csv.gz')
    chain_rows = []
    for (n, aspl, k), rows in chain.groupby(['n', 'aspl', 'k']):
        f, m = rows.full_T9, rows.mm_T9
        chain_rows.append(dict(n=n, aspl=aspl, k=k, pairs=len(rows),
                               mean_full_T9=f.mean(), mean_mm_T9=m.mean(),
                               median_full_T9=f.median(), median_mm_T9=m.median(),
                               rel_w1_pct=100 * np.abs(np.sort(f) - np.sort(m)).mean() / f.mean(),
                               median_shift_pct=100 * (m.median() - f.median()) / f.median()))
    summary = pd.read_csv(root / 'data/chains/cell_summary.csv')
    summary['paper_rho'] = .003 * summary.sigma_full
    ranks = summary[['paper_rho', 'rel_w1_pct']].rank()
    claims = dict(chain_pairs=len(chain), chain_cells=len(summary),
                  common_30000_full_observed=int(chain.full_T9.le(30000).sum()),
                  common_30000_twoscale_observed=int(chain.mm_T9.le(30000).sum()),
                  largest_terminal_w1_pct=float(summary.rel_w1_pct.max()),
                  spearman_rho_vs_w1=float(ranks.paper_rho.corr(ranks.rel_w1_pct)),
                  configurations_below_rho_0p25=int(summary.paper_rho.lt(.25).sum()),
                  largest_w1_below_rho_0p25=float(summary.loc[summary.paper_rho.lt(.25), 'rel_w1_pct'].max()),
                  chain_clique_w1_pct=float(summary.loc[summary.n.eq(1000) & summary.aspl.eq(1) & summary.k.eq(1), 'rel_w1_pct'].iloc[0]),
                  simulations_run=False, bootstrap_or_permutation_recomputed=False)
    return pd.DataFrame(source_rows), pd.DataFrame(chain_rows), claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, help='new directory for reproduced point estimates')
    args = parser.parse_args()
    source, chains, claims = point_estimates(args.dataset)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        source.to_csv(args.out / 'source_leaf_point_estimates.csv', index=False)
        chains.to_csv(args.out / 'chain_point_estimates.csv', index=False)
        (args.out / 'article_numbers.json').write_text(json.dumps(claims, indent=2) + '\n')
    print(json.dumps(claims, indent=2))


if __name__ == '__main__':
    main()
