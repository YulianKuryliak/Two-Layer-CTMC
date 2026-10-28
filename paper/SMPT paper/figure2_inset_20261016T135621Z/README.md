# Figure2(b) near-zero inset

Source: the user-provided Downloads/main-SMPT (4).tex, preserved as
main-SMPT.before.tex. The updated main-SMPT.tex is a copy of that supplied article
with only an inset in Figure2(b) and a matching caption sentence added.
No working main manuscript, template, raw data or projected path table was edited.

For an Overleaf project with newer prose edits, replace only the complete
Figure2 environment (label fig:clique_leaf) with figure2_replacement.tex content.
The replacement is meant to be pasted into the main, not loaded as a dependency.
Alternatively use main-SMPT.tex if the supplied original is still authoritative.

The inset uses the existing leaf_clique_signed_delay_full.dat and
leaf_clique_signed_delay_twoscale.dat (without the _path suffix). Keep the
four _path.dat tables for the unchanged main curves. All these files already
exist in overleaf_project/figures_pgfplots. No new data files are needed.
Inset range: time -22 to3; CDF0 to2.5percent. The main axes remain unchanged.
All10000observations are read; restrict x to domain only controls display.
The inset probability axis is multiplied by100 and explicitly labelled CDF(%).

Verification: exact source-scope comparison; original raw files matched working
data byte-for-byte; stable pdfLaTeX1page isolated figure build at standard
5000000-word memory and visual review. Full article not rebuilt in this task.
Preview: figure2_preview.pdf; cropped image: figure2_zoom.png.
