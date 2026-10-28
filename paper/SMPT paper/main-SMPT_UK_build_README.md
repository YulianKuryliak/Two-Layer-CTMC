# Current Ukrainian SMPT main

`main-SMPT_UK.tex` retains Sections1-5 and the first Results block through
subsection6.4, including all three descriptor panels. The bibliography is
retained. Subsection6.5 and the second Results block are archived in
`uk_through_6_4_backup_20261014/after_6_4_UK.tex`; the complete prior source
and PDF are in that folder.

Compile with **LuaLaTeX**, required by the Ukrainian fontspec setup:

```sh
latexmk -lualatex main-SMPT_UK.tex
```

For Overleaf, select LuaLaTeX and `main-SMPT_UK.tex` as the main document.
Plot code remains inline and editable; numeric tables are in
`figures_pgfplots/`. No data or statistics were recomputed.
