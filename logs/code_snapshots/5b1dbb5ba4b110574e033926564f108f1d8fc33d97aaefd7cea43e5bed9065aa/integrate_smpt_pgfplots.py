#!/usr/bin/env python3
"""Install editable figure code inline in the current Ukrainian SMPT main.

Only known figure bodies, PGFPlots setup and two obsolete legend explanations
are changed. Refuses unexpected figure counts. Re-running safely refreshes
marked plotting blocks; scientific prose and captions otherwise stay intact.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/SMPT paper'
FIGURES = PAPER / 'figures_pgfplots'
MAIN = PAPER / 'main-SMPT_UK.tex'
FIGURES_BY_IMAGE = {
    'clique_leaf_endpoint_and_mechanism.png': 'clique_leaf',
    'fig_aspl_export_fidelity.png': 'source_sweep',
    'chain_cliques_n1000_k1_propagation_cdf_random_initial.pdf': 'chain_cliques',
    'aspl_chain_median_arrival_error_map_labeled.pdf': 'chain_heatmap',
    'fig_accuracy_map_separate_metrics.pdf': 'accuracy',
    'fig_descriptor.pdf': 'descriptor',
}


def main():
    original = MAIN.read_text(encoding='utf-8')
    updated = original
    counts = {}
    for image, stem in FIGURES_BY_IMAGE.items():
        start, end = f'% BEGIN EDITABLE FIGURE {stem}', f'% END EDITABLE FIGURE {stem}'
        block = start+'\n'+(FIGURES/(stem+'.tex')).read_text(encoding='utf-8').strip()+'\n'+end
        if start in updated:
            pattern = re.escape(start)+r'.*?'+re.escape(end)
        else:
            pattern = r'\\includegraphics\[[^\]]+\]\{[^\n}]*'+re.escape(image)+r'\}'
        updated, count = re.subn(pattern, lambda _: block, updated, flags=re.DOTALL)
        if count != 2:
            raise ValueError(f'Expected both existing figure inclusions for {stem}; found {count}')
        counts[stem] = count
    if r'\usepackage{pgfplots}' not in updated:
        updated = updated.replace(r'\usepackage{tikz}', '\n'.join([
            r'\usepackage{tikz}',r'\usepackage{pgfplots}',
            r'\usepgfplotslibrary{groupplots,fillbetween}', r'\usetikzlibrary{patterns}']),1)
    style_start, style_end = '% BEGIN EDITABLE PLOT STYLE', '% END EDITABLE PLOT STYLE'
    style = style_start+'\n'+(FIGURES/'style.tex').read_text().strip()+'\n'+style_end+'\n'
    if style_start in updated:
        updated = re.sub(re.escape(style_start)+r'.*?'+re.escape(style_end)+'\n?',lambda _:style,updated,flags=re.DOTALL)
    else:
        updated = updated.replace(r'\begin{document}',style+r'\begin{document}',1)
    obsolete = " ``Full Micro'' позначає \\Full{}, а ``MicroMacro'' --- \\TS{}."
    updated = updated.replace(obsolete,'')
    # The first diagram, all equations, labels and surrounding prose are kept.
    for command in ('section','subsection','label','citep','citet'):
        pattern = r'\\'+command+r'\*?(?:\[[^\]]*\])?\{[^}]*\}'
        assert re.findall(pattern,original)==re.findall(pattern,updated), command
    # Ignore newly inserted plot mathematics; manuscript equation environments
    # must remain identical.
    pattern = r'\\begin\{(?:equation|align)\*?\}.*?\\end\{(?:equation|align)\*?\}'
    assert re.findall(pattern,original,re.DOTALL)==re.findall(pattern,updated,re.DOTALL)
    MAIN.write_text(updated,encoding='utf-8')
    previous_record = (json.loads((FIGURES/'integration_manifest.json').read_text())
                       if (FIGURES/'integration_manifest.json').exists() else {})
    record = {'replaced_figure_bodies':counts,'plot_code':'inline in main-SMPT_UK.tex',
              'numeric_tables':'figures_pgfplots/*.dat',
              'preconversion_main_sha256': previous_record.get('preconversion_main_sha256', previous_record.get('previous_main_sha256', hashlib.sha256(original.encode()).hexdigest())),
              'previous_main_sha256':hashlib.sha256(original.encode()).hexdigest(),
              'current_main_sha256':hashlib.sha256(updated.encode()).hexdigest(),
              'scope':'PGFPlots setup, figure bodies, obsolete legacy legend explanations only; inherited duplicate Results retained'}
    (FIGURES/'integration_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Updated six result figure types inline (12 existing inclusions); labels, citations, headings and equations unchanged.')


if __name__=='__main__':
    main()
