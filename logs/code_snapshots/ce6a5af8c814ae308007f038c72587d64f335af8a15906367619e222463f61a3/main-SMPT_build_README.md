# Final English SMPT main

`main-SMPT.tex` contains Sections 1–6 directly. Section 6 translates the **first**
Results block in the current Ukrainian main: five subsections and six figures.
The existing title, Sections 1–5 and the `test` abstract are preserved.

Use **LuaLaTeX**, including in Overleaf's compiler settings. The editable ECDFs
retain all 10,000 observations per model; pdfLaTeX's default memory limit is
insufficient for these plots.

From `paper/SMPT paper`:

```sh
latexmk main-SMPT.tex
```

The local `.latexmkrc` selects LuaLaTeX and supplies the vendored PGFPlots path.
The resulting files are `main-SMPT.pdf` and, after text export, `main-SMPT.txt`.
`main-SMPT_editable_figures.zip` contains the main, template, bibliography,
editable plot data and local plotting dependency for uploading or copying.

Edit the plotting code in the main and its numeric tables in
`figures_pgfplots/`. The plot sources and data match the first Ukrainian Results
block; no simulations or statistics were recomputed during translation.

The prior English source/PDF/TXT, the selected Ukrainian source, translation
and verification records are saved in `results_first_block_en_20261013/`.
References to later appendices and Discussion keep the existing reserved
full-paper numbers; those sections are outside this partial document.
