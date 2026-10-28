# Editable article figures

The six result figures are native TikZ/PGFPlots. The Ukrainian main contains
their plotting code directly. The `.dat` files in this folder contain the
numeric values used by the plots; they are dependencies of the main document.
Heatmap colors and printed cell values read the same table. A `hatch=1` marks
a Wasserstein cell with saved permutation p >= 0.05, without multiplicity
adjustment. Hatching does not apply to the median metric.

`preview.tex` displays all six figures separately, using the same plotting
snippets and style. The individual `.tex` snippets are convenient reusable
copies; editing them alone does not change the inline plots in the main.

## Compilation

From `paper/SMPT paper`, use `latexmk -lualatex main-SMPT_UK.tex`.
The local `.latexmkrc` adds the vendored PGFPlots fallback to TeX/Lua search
paths. With a normal TeX distribution including PGFPlots (including Overleaf),
the document requires only the main, template/bibliography and this folder's
`.dat` files. The main document uses the article's existing CAS template.

The standalone gallery can be built from the same directory with:

```sh
TEXINPUTS=.:figures_pgfplots/vendor//: LUAINPUTS=.:figures_pgfplots/vendor//: \
  lualatex -output-directory=figures_pgfplots figures_pgfplots/preview.tex
```

## Data provenance and regeneration

`chain_manifest.json` and `leaf_manifest.json` record source hashes, definitions
and validation. Saved observations and summaries are used directly. No
simulations or bootstrap/permutation resampling are run by the exporters.
Clique-chain shading is the interquartile range, not a confidence interval.
Leaf bypass intervals reproduce the original deterministic Wilson formula.

To regenerate the reusable plot sources and data (this does not overwrite the
inline manuscript figures), from the repository root:

```sh
.venv/bin/python extrascripts/visualization/export_chain_pgfplots.py
.venv/bin/python extrascripts/visualization/export_leaf_pgfplots.py
```

Backups of the pre-conversion manuscripts, PDFs, figures, template and plotting
scripts are in `../backups/before_editable_figures_20261013/`, with a SHA256
manifest. Original image assets and scientific outputs remain available.

## Local dependency

`vendor/` contains unmodified PGFPlots 1.18.3 runtime files from the official
CTAN TeX-directory archive:
https://mirrors.ctan.org/install/graphics/pgf/contrib/pgfplots.tds.zip

See `vendor/CTAN_README.txt` and file headers for authorship and GPL licensing.
