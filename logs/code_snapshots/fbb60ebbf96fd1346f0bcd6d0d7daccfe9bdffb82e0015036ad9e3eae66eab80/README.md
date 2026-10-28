# Overleaf Figure 2 repair

The uploaded project lacked four `*_path.dat` tables referenced by panels (a)
and (b). The original raw ECDF tables are present and byte-identical to the
current project. The four added tables retain all 10000 observations and a zero
anchor per curve. They are existing display-coordinate copies, not new results.

Upload the four files from `figure2_missing_files.zip` into the existing
`SMPT paper/figures_pgfplots/` directory in Overleaf, then Recompile from scratch
with pdfLaTeX. Keep the `_path` suffixes. No main TEX change is needed.

Alternatively import `Kuryliak_and_Emmerich_2026_Overleaf_figure2_fixed.zip` as a
new Overleaf project. Every original ZIP entry is preserved byte-for-byte; only
four missing files were added. This archive preserves the uploaded manuscript,
which may differ from the latest English main in the working repository.
The original Downloads ZIP was not modified.

Local verification uses a copy of the project .latexmkrc for the vendored
PGFPlots search path; this extra verification config is not added to the fixed ZIP.

Verification: pdfLaTeX produced the article PDF and Figure2 panels (a) and (b)
were visually checked on page15; all four curves are present. Only figure
repair was audited. The isolated build initially omitted the archive-root
references_fixed.bib; that file exists in the full original/fixed archives.
Main source and all original entries in the corrected archive are unchanged.
