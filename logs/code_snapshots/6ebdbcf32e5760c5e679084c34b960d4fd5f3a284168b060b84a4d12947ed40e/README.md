# Final English SMPT manuscript: Sections 1–3

Main editable document: `../SMPT_final_EN_sections_1_3.tex`.
Compiled PDF: `../SMPT_final_EN_sections_1_3.pdf`.

The original article title, authors, affiliations and Elsevier `cas-sc`
format are retained from `../claude-version_merged_parallel_EN.tex`.
This partial document contains Sections 1–3 and the cited bibliography;
the abstract, highlights, keywords and later sections are omitted.

The English text mirrors the saved user-edited source
`../readability_revision_parallel_black_20261008/sections_1_3_UK.tex`.
An exact snapshot is stored as `user_sections_1_3_UK_snapshot.tex`.
No independent scientific additions or alterations are included.
The original English and Ukrainian sources remain intact.

`sections_1_3_EN.tex` contains the translated section text used to generate
the main document. `english_vs_original.diff` records changes from the old
English article; `english_vs_previous.diff` records changes from the previous
English readability revision. `verification.json` records input fingerprints
and targeted checks. `document.txt` is the extracted PDF text.

References to later sections and propositions retain the numbering of the
original full article using `later_sections_labels.tex`. These references
are displayed without hyperlinks because their targets are outside this
partial document. Update them when adding the later sections.

To regenerate from the maintained section source, run from the repository
root in WSL:

```sh
.venv/bin/python extrascripts/analysis/build_smpt_final_en_sections_1_3.py
```

The builder checks exact citation and label sequences, identical display
and numbered equations, the same heading structure and an identical TikZ
figure against the Ukrainian snapshot. It then builds with LuaLaTeX and
BibTeX and checks for undefined references, missing glyphs and overflow.
No simulations or diagnostic experiments are run.

If editing the main TEX directly, compile it with LuaLaTeX/BibTeX rather
than regenerating from the section source. The regeneration script refuses
to overwrite a directly edited main TEX once its build fingerprint has
been recorded.
