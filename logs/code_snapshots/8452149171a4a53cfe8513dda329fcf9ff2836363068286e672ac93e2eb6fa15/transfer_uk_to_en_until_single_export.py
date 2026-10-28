from pathlib import Path
import re
import json
import hashlib
import subprocess
import difflib

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/SMPT paper'
OUT = PAPER / 'final_en_until_single_export_20261009'
path = PAPER / 'main-SMPT.tex'
before = path.read_text(encoding='utf-8-sig')
uk = (PAPER / 'main-SMPT_UK.tex').read_text(encoding='utf-8-sig')
uk_prefix = uk[uk.index(r'\section{Вступ}'):uk.index(r'\subsection{Одне передавання інфекції назовні}')]
(OUT / 'main-SMPT_EN_before_transfer.tex').write_bytes(path.read_bytes())
(OUT / 'user_UK_until_single_export_snapshot.tex').write_text(uk_prefix, encoding='utf-8')
changes = {
    r'It is exact for a \emph{surrogate network}: the real cross-community edges are replaced by weak connections between all node pairs, preserving the total weight':
    r'The ideal process is exact for a \emph{surrogate network with uniform inter-community connections}: for each connected community pair, every node in one community is connected to every node in the other, and the total coupling weight is distributed equally among these node pairs',
    'Our surrogate network is related to this idea':
    'Our surrogate network with uniform inter-community connections is related to this idea',
    'exact for the surrogate network defined later':
    'exact for the surrogate network with uniform inter-community connections defined later',
    'equivalent to a surrogate network in which every ordered node pair across the two communities has a weak connection':
    'equivalent to a process on a surrogate network with uniform inter-community connections: for each connected community pair, all ordered node pairs have the same contact weight',
    'exact for the surrogate network.':
    'exact for the surrogate network with uniform inter-community connections.',
    'exact for a surrogate network with uniform inter-community coupling.':
    'exact for a surrogate network with uniform inter-community connections.',
}
after = before
for old, new in changes.items():
    assert old in after, old
    after = after.replace(old, new)
tail = (OUT / 'section_4_start_EN.tex').read_text()
assert r'\label{sec:analysis}' not in after
after = after.replace(r'\bibliographystyle{cas-model2-names}', tail + '\n\n' + r'\bibliographystyle{cas-model2-names}')
# References to omitted material retain their full-article numbers without dead links.
after = re.sub(r'% Later-section reference numbers.*?\\AtBeginDocument\{.*?\n\}\n', '', after, flags=re.S)
labels = re.findall(r'\\label\{([^}]+)\}', after)
assert len(labels) == len(set(labels))
refs = set(re.findall(r'\\(?:ref\*?|eqref)\{([^}]+)\}', after))
missing = sorted(refs - set(labels))
aux = (PAPER / 'claude-version_merged_parallel_EN.aux').read_text()
external = []
for label in missing:
    match = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{([^}]+)\}\{([^}]+)\}', aux)
    assert match, label
    external.append('\\newlabel{' + label + '}{{' + match[1] + '}{' + match[2] + '}{}{}{}}')
for label in labels:
    after = after.replace('\\ref*{' + label + '}', '\\ref{' + label + '}')
for label in missing:
    after = after.replace('\\ref{' + label + '}', '\\ref*{' + label + '}')
after = after.replace(r'\begin{document}', '% Full-paper numbers for material not yet translated into this partial main.\n\\AtBeginDocument{\n' + '\n'.join(external) + '\n}\n' + r'\begin{document}', 1)
en_prefix = after[after.index(r'\section{Introduction}'):after.index(r'\bibliographystyle')]
assert re.findall(r'\\label\{([^}]+)\}', uk_prefix) == re.findall(r'\\label\{([^}]+)\}', en_prefix)
assert re.findall(r'\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}', uk_prefix) == re.findall(r'\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}', en_prefix)
for pattern in (r'\\begin\{equation\}(.*?)\\end\{equation\}', r'\\\[(.*?)\\\]',
                r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}'):
    normalize = lambda text: [re.sub(r'\s+', '', item) for item in re.findall(pattern, text, re.S)]
    assert normalize(uk_prefix) == normalize(en_prefix), pattern
assert r'\label{sec:single_export}' not in after
path.write_text(after, encoding='utf-8')
(OUT / 'english_main_changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='main-SMPT.tex before', tofile='main-SMPT.tex after')), encoding='utf-8')
for args in [('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
             ('bibtex', 'main-SMPT'),
             ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
             ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex')]:
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-2000:]
log = (PAPER / 'main-SMPT.log').read_text()
titleline = next(i for i, line in enumerate(after.splitlines(), 1) if line == r'\maketitle')
warnings = [line for line in log.splitlines() if any(x in line.lower() for x in
           ('undefined', 'missing character', 'multiply defined', 'overfull'))]
assert not [x for x in warnings if x != 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(titleline)], warnings
fls = (PAPER / 'main-SMPT.fls').read_text()
assert all('INPUT ./' + x in fls for x in ['cas-sc.cls', 'cas-common.sty'])
result = dict(scope='Sections 1-3 and 4.1-4.2, stopping before single export',
              root_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              ukrainian_main_sha256=hashlib.sha256((PAPER / 'main-SMPT_UK.tex').read_bytes()).hexdigest(),
              labels=len(labels), mathematical_and_citation_parity=True,
              external_references=missing, warnings=warnings, simulations_run=False)
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
