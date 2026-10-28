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

## Code availability and remaining submission checks (2026-10-15)

An unnumbered `Code availability` section now follows Conclusions and links
only to the simulator source at
https://github.com/YulianKuryliak/Two-Layer-CTMC.
It does not claim that all experimental data are public or identify a release
as the source of the reported results. For reproducibility, identify and archive
the actual code version and results/data used in the paper before submission.

The CAS class already supplies the article title, author affiliations, ORCIDs,
corresponding-author details, abstract, keywords, sections, figures and references.
The following items remain to be completed or checked. Unnumbered declaration
sections have been inserted with explicit author-completion placeholders; factual
author declarations have not been invented:

- Restore Appendices A-C or supply them as explicitly identified supplementary
  material and update the references. The current preamble reserves labels for
  absent appendices, so a successful TeX build does not establish completeness.
- Prepare a Declaration of competing interest based on all authors' information,
  and state funding sources and grant numbers, or the factual absence of specific
  funding. See Elsevier's declaration guidance:
  https://www.elsevier.support/publishing/answer/what-are-conflict-of-interest-statements-funding-source-declarations-author-agreementsdeclarations-and-permission-notes
- Prepare CRediT contributions with roles confirmed by the authors:
  https://www.elsevier.com/researcher/author/policies-and-guidelines/credit-author-statement
- State the availability of the experimental data, analysis inputs and scripts,
  including any repository/archive identifiers or restrictions. The simulator
  GitHub link alone does not document data availability. Journal-specific data
  requirements must be checked:
  https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines
- Finalize the draft declaration of AI assistance before References. Translation
  and substantive language/structure editing occurred during manuscript preparation;
  Elsevier requires disclosure of these uses, beyond basic spelling/grammar checks.
  Confirm the actual tools and scope and authors' review before finalizing wording:
  https://www.elsevier.com/en-gb/about/policies-and-standards/generative-ai-policies-for-journals
- Check the current journal limits for the abstract (currently about 262 words)
  and keywords (currently 8 entries). Neither has been shortened in this task.
- Verify bibliography style against current SMPT instructions: this manuscript
  uses author-year natbib and cas-model2-names. Recent published SMPT articles
  use numeric citations; this is a format check, not a change made here.
- Prepare Highlights as a separate editable file if required by the journal:
  Elsevier specifies 3-5 bullets of at most 85 characters including spaces, but
  explicitly says journal-specific requirements determine whether they are mandatory.
  https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript
- Check the submission portal for a cover letter, declaration uploads and any
  graphical abstract requirement. A CAS environment's existence alone does not
  make that item mandatory for SMPT.

Current SMPT Guide for Authors:
https://www.sciencedirect.com/journal/simulation-modelling-practice-and-theory/publish/guide-for-authors
The journal guide could not be retrieved in this check (HTTP 403 / access error).
Thus the above distinguishes observed manuscript gaps and verified general
Elsevier guidance from journal-specific requirements that remain unconfirmed.

## Declaration sections added (2026-10-15)

The main now contains Data availability, Funding, CRediT authorship contribution
statement, Declaration of competing interest, and Declaration of generative AI
and AI-assisted technologies in the manuscript preparation process. Code
availability remains immediately after Conclusions. AI disclosure is the last
section before References. Unknown author-specific details are visible italic
placeholders, including each author's CRediT roles. The AI draft names the known
OpenAI Codex assistance with translation, language and structure; the complete
list of tools/uses and author review/responsibility still require confirmation.
These placeholders must be completed before submission. No missing appendices
or scientific claims were invented, and no Ukrainian text was edited.

## Declaration wording updated after author instructions (2026-10-15)

The user confirmed that the authors have no competing interests and no funding.
Funding and Declaration of competing interest now contain complete statements.
The AI declaration names OpenAI Codex and Anthropic Claude for drafting,
translation, language editing and text organisation, followed by author review
and responsibility wording. This is a manuscript-preparation declaration; it
does not assert that AI was unused for code or research workflows.

The CRediT section now contains a proposed allocation, explicitly marked for
both authors' confirmation: Kuryliak -- Conceptualization, Methodology, Software,
Investigation, Formal analysis, Visualization, Writing - original draft;
Emmerich -- Conceptualization, Methodology, Supervision, Writing - review & editing.
This proposal follows the earlier local twoscale draft and is not treated as a
confirmed account. Data availability remains an author-completion placeholder.
The earlier dated checklist entries record the state at the time they were written.
