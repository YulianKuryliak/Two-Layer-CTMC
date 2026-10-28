# Project-local PGFPlots fallback for TeX installations without that package.
# IEEE Access uses the original complete streamed ECDFs with pdfLaTeX.
$pdf_mode = 1;
my $plot_sep = ($^O eq 'MSWin32') ? ';' : ':';
$ENV{'TEXINPUTS'} = '.' . $plot_sep . 'figures_pgfplots/vendor//' . $plot_sep . ($ENV{'TEXINPUTS'} // '');
$ENV{'LUAINPUTS'} = '.' . $plot_sep . 'figures_pgfplots/vendor//' . $plot_sep . ($ENV{'LUAINPUTS'} // '');

@default_files = ("main.tex");
