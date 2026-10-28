# Ukrainian Research Synthesis

`report.tex` and `report.pdf` synthesize the completed Two-Layer-CTMC experiment
reports around one question: when MicroMacro differs from Full Micro and how that
evidence should constrain calibration.

Build from this directory:

```bash
lualatex -interaction=nonstopmode -halt-on-error report.tex
lualatex -interaction=nonstopmode -halt-on-error report.tex
```

The report treats Experiment 05 and CE-07 as development evidence and excludes
partial ER-01 pilot outputs from final conclusions.
