#!/usr/bin/env python3
"""Prepare all-observation ECDF drawing files for bounded-memory pdfLaTeX.

Only display coordinates are normalised. No observation, ECDF jump, statistic,
or confidence interval is removed or recomputed. The raw .dat files remain the
scientific sources. PGF streams the projected paths without PGFPlots' retained
per-point analysis metadata. Axis bounds are declared in the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'paper/SMPT paper/figures_pgfplots'
BOUNDS = {
    'leaf_clique_export_full': (0., 600., 0., 1.01),
    'leaf_clique_export_twoscale': (0., 600., 0., 1.01),
    'leaf_clique_signed_delay_full': (-20., 100., 0., .55),
    'leaf_clique_signed_delay_twoscale': (-20., 100., 0., .55),
}


def main():
    rows = []
    for stem, (xmin, xmax, ymin, ymax) in BOUNDS.items():
        source = FIGURES / (stem+'.dat')
        values = np.loadtxt(source, skiprows=1)
        assert values.shape == (10001, 2) and np.isfinite(values).all()
        assert values[0, 1] == 0 and values[-1, 1] == 1
        assert (np.diff(values[:, 0]) >= 0).all()
        projected = (values-np.array([xmin, ymin])) / np.array([xmax-xmin, ymax-ymin])
        target = FIGURES / (stem+'_path.dat')
        np.savetxt(target, projected, fmt='%.17g', header='normalised x y; all 10000 observations and zero anchor')
        check = np.loadtxt(target)
        restored = check*np.array([xmax-xmin, ymax-ymin])+np.array([xmin, ymin])
        np.testing.assert_allclose(restored, values, rtol=2e-13, atol=2e-13)
        assert check.shape == values.shape
        rows.append({'source':str(source.relative_to(ROOT)), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'drawing_file':str(target.relative_to(ROOT)), 'drawing_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                     'points':len(values), 'bounds':[xmin,xmax,ymin,ymax],
                     'max_inverse_coordinate_error':float(np.max(np.abs(restored-values)))})
    manifest = {'purpose':'pdfLaTeX-compatible streaming ECDF drawing coordinates',
                'all_observations_retained':True, 'right_continuous_steps':True,
                'no_statistics_recomputed':True, 'files':rows,
                'editing':'Style/width/height remain editable in the main. If axis bounds or raw observations change, regenerate these display coordinates with this script.'}
    (FIGURES/'pdftex_ecdf_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Prepared four complete ECDF paths: 10001 points each, inverse-coordinate parity checked.')


if __name__ == '__main__':
    main()
