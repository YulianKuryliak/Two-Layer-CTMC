# Final English SMPT main

`main-SMPT.tex` contains Sections1-8 directly, synchronized with the current
Ukrainian manuscript. Results retain subsections6.1-6.4 and six figures;
subsection6.5 stays excluded. All three descriptor panels, including(c),
and their caption are retained. Discussion and Conclusions are translated
in full from active Ukrainian prose; commented-out text is excluded.
The original English title is preserved. The abstract and keywords are
translated from the current Ukrainian version.

Use **pdfLaTeX**, including in Overleaf. The four large ECDFs retain every
10000-observation sample and use streamed native PGF paths, with no custom
memory limit or shell escape. From `paper/SMPT paper`:

```sh
latexmk -pdf main-SMPT.tex
```

The local `.latexmkrc` supplies the vendored PGFPlots path. The resulting
PDF is `main-SMPT.pdf`; extracted text is `main-SMPT.txt`.
`main-SMPT_editable_figures.zip` includes the main, template, bibliography,
editable data tables, icons and plotting dependency for Overleaf.

Plot code remains inline; numeric tables are in `figures_pgfplots/`.
The four `*_path.dat` tables are verified display-coordinate copies of the
original ECDF samples. Edit width/height/colour/style directly in the main.
After changing ECDF axis bounds or raw observations, regenerate the display
copies from the repository root with:

```sh
.venv/bin/python extrascripts/visualization/prepare_pdftex_ecdf.py
```

The Ukrainian fontspec main still compiles with LuaLaTeX explicitly:
`latexmk -lualatex main-SMPT_UK.tex`.

The body translation and audit are in `en_sync_uk_20261014T161350Z/`.
The latest abstract/keyword source, English backup and verification are in
`en_abstract_keywords_20261014T163247Z/`.
Appendices A-C remain outside this document, with their existing reserved
references. Discussion now uses its real section label and hyperlink.
No simulations or scientific statistics were recomputed.
