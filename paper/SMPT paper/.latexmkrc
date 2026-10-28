# Project-local PGFPlots fallback for TeX installations without that package.
# LuaLaTeX for main-SMPT_UK.tex: latexmk -lualatex main-SMPT_UK.tex
# English main uses streamed complete ECDFs compatible with pdfLaTeX.
$pdf_mode = 1;
my $plot_sep = ($^O eq 'MSWin32') ? ';' : ':';
$ENV{'TEXINPUTS'} = '.' . $plot_sep . 'figures_pgfplots/vendor//' . $plot_sep . ($ENV{'TEXINPUTS'} // '');
$ENV{'LUAINPUTS'} = '.' . $plot_sep . 'figures_pgfplots/vendor//' . $plot_sep . ($ENV{'LUAINPUTS'} // '');
