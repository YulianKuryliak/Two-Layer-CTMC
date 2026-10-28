"""Render separate accuracy metrics from the saved corrected-chain summary only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extrascripts.visualization import smpt_twoscale_paper_figures as figures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'output/analysis/chain_corrected_figures_20261008/cell_summary.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'paper/SMPT paper/Images/twoscale/corrected_chain_20261008')
    args = parser.parse_args()
    summary_hash = sha(args.summary)
    summary = figures.pd.read_csv(args.summary)
    if not summary.R.eq(400).all():
        raise ValueError('Expected 400 observed pairs in each configuration')
    figures.OUT = args.out
    name = 'fig_accuracy_map_separate_metrics'
    figures.accuracy_map(summary, name=name)
    assert sha(args.summary) == summary_hash
    metadata = dict(
        kind='Figure rendering from saved data; no simulation or statistical recomputation',
        source=str(args.summary.relative_to(ROOT)), source_sha256=summary_hash,
        cells=len(summary), pairs_per_cell=400,
        panel_layout={'rows': ['k=1', 'k=5'], 'columns': ['median_shift_pct', 'rel_w1_pct']},
        significance='Wasserstein panels only: hatching for rel_w1_perm_p >= 0.05',
        colour_scales='Zero-centred diverging median scale; nonnegative Wasserstein scale; shared within each metric',
        scripts={str(path.relative_to(ROOT)): sha(path) for path in
                 [Path(__file__).resolve(), Path(figures.__file__).resolve()]},
        outputs={name + ext: sha(args.out / (name + ext)) for ext in ('.pdf', '.png')},
        simulations_run=False,
    )
    (args.out / (name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
