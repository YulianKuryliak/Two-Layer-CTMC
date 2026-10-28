# English main, current Sections 1--4

Main document: ../main-SMPT.tex; PDF: ../main-SMPT.pdf. Native SMPT CAS formatting retained.

Source of truth: current Ukrainian main, archived exactly in user_main-SMPT_UK_snapshot.tex. Sections 1--3 were checked against the previously translated text; their figure already matches. Section 4 was replaced with a translation of its active current text. Section 5 and both commented temporal-descriptor drafts are excluded. The editorial question in the remark heading is retained only as a comment; literal escaped X_0 is typeset correctly.

Build directly in the SMPT paper folder: pdflatex main-SMPT.tex; bibtex main-SMPT; pdflatex twice. After direct main edits, do not regenerate without reconciling them. The update script guards source hashes. References to Sections 5--7 retain full-paper numbers; these sections are not yet included.
