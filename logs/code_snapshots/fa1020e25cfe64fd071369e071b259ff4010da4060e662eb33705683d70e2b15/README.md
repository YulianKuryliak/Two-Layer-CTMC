# Overleaf folder figure audit

The modern editable Figure2 is in `overleaf_project/test.tex`.
`overleaf_project/main-SMPT.tex` is a different older article version.
Four missing `leaf_clique_*_path.dat` files were copied into the existing
figures_pgfplots folder from the already verified saved display tables.
All existing project files, raw observations and article source bytes remain
unchanged. All37data tables referenced by test.tex are now present.

For Overleaf, upload the four new path tables into figures_pgfplots and select
test.tex as the main document if that is the intended manuscript. Use pdfLaTeX
and Recompile from scratch. This check does not replace either manuscript.

figure2_check.tex contains the exact preamble and Figure2 block from test.tex,
with only minimal document wrappers for an isolated rendering check.

Additional blocker: cas-sc.cls held a sample article, cas-common.sty held a
standalone article, and cas-model2-names.bst held a class definition. These three
files were backed up to template_backup and replaced with the verified template
files from the working paper. No article text or existing numeric data changed.
All37referenced plot tables now match the working project byte-for-byte.

Verified: stable pdfLaTeX1page Figure2 render, all4ECDFcurves visible.
The7-file repair ZIP contains only4path tables and3restored template files.
Article sources remain unchanged. Full article bibliography/images outside
Figure2 were not built in this isolated check.
