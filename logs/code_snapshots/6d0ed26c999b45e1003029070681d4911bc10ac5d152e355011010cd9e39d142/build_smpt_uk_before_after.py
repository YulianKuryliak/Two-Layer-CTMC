"""Build aligned Ukrainian before/after sections 1--3 PDFs; no experiments."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/SMPT paper'
FOLDER = PAPER / 'plain_uk_before_after_20261009'
ORIGINAL = PAPER / 'readability_revision_parallel_black_20261008/sections_1_3_UK.tex'
BASE = 'claude-version_merged_plain_sections_1_3_black_UK'


def run(*args):
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(' '.join(args) + '\n' + result.stdout[-5000:] + result.stderr[-1500:])


def blocks(text):
    out, current, depth = [], [], 0
    for line in text.splitlines():
        if line.startswith('%') or line in (r'\begingroup\color{black}', r'\endgroup', r'\clearpage') or line.startswith(r'\hypersetup'):
            continue
        if not line.strip() and depth == 0:
            if current:
                out.append('\n'.join(current))
                current = []
            continue
        current.append(line)
        depth += len(re.findall(r'\\begin\{', line)) - len(re.findall(r'\\end\{', line))
        assert depth >= 0
    if current:
        out.append('\n'.join(current))
    assert depth == 0
    return out


def main():
    before = (FOLDER / 'before.tex').read_text(encoding='utf-8-sig')
    after = (FOLDER / 'after.tex').read_text(encoding='utf-8-sig')
    assert (FOLDER / 'before.tex').read_bytes() == ORIGINAL.read_bytes(), 'Original changed after snapshot'
    labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
    cites = lambda s: re.findall(r'\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}', s)
    equations = lambda s: re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', s, re.S)
    assert labels(before) == labels(after)
    assert cites(before) == cites(after)
    assert equations(before) == equations(after)
    tikz = lambda s: re.search(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', s, re.S).group()
    assert tikz(before) == tikz(after)
    aux = (PAPER / (BASE + '.aux')).read_text(encoding='utf-8-sig')
    numbers = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
    bbl = (PAPER / (BASE + '.bbl')).read_text(encoding='utf-8-sig')

    def display(text):
        def heading(m):
            kind, title, label = m.groups()
            return '\\' + kind + '*{' + numbers[label] + '. ' + title + '}'
        text = re.sub(r'\\(section|subsection)\{([^}]+)\}\s*\\label\{([^}]+)\}', heading, text)
        text = re.sub(r'\\eqref\{([^}]+)\}', lambda m: '(' + numbers[m[1]] + ')', text)
        text = re.sub(r'\\ref\{([^}]+)\}', lambda m: numbers[m[1]], text)
        def equation(m):
            body = m[1]
            label = re.search(r'\\label\{([^}]+)\}', body)[1]
            body = re.sub(r'\\label\{[^}]+\}', '', body).strip()
            return r'\begin{equation*}' + body + '\\tag{' + numbers[label] + '}\n' + r'\end{equation*}'
        text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', equation, text, flags=re.S)
        text = re.sub(r'\\label\{[^}]+\}', '', text)
        for kind in ('table', 'figure', 'algorithm'):
            text = re.sub(r'\\begin\{' + kind + r'\}(?:\[[^\]]*\])?', r'\\par\\noindent\\begin{minipage}{\\linewidth}', text)
            text = text.replace('\\end{' + kind + '}', r'\end{minipage}\par')
        # Display captions without duplicate counters or labels in the parallel copy.
        text = text.replace(r'\caption{', r'\par\small\textbf{')
        text = text.replace(r'\begin{tabular}', r'\setlength{\tabcolsep}{3pt}\begin{tabular}')
        return text

    a, b = blocks(before), blocks(after)
    assert len(a) == len(b), (len(a), len(b))
    common = r'''
\usepackage{fontspec}
\setmainfont{Nimbus Roman}
\setsansfont{Nimbus Sans}
\setmonofont{Nimbus Mono PS}
\usepackage[authoryear]{natbib}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,xcolor,caption,tikz,longtable}
\usetikzlibrary{positioning,fit,arrows.meta}
\usepackage{microtype}
\microtypesetup{expansion=false}
\usepackage{url}
\setlength{\emergencystretch}{3em}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\newcommand{\ibar}{\bar{y}}
\newcommand{\sbar}{\bar{s}}
\newcommand{\Exp}{\mathrm{Exp}}
\newcommand{\Full}{\textup{\textsc{Full}}}
\newcommand{\TS}{\textup{\textsc{TwoScale}}}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\eW}{e_W}
'''
    compare = [r'\documentclass[11pt]{article}', r'\usepackage[a3paper,landscape,margin=15mm]{geometry}', common,
               r'\begin{document}', r'\begin{center}\Large Розділи 1--3: простіша українська редакція\end{center}',
               r'Відповідні фрагменти вирівняно поруч. Нумерація розділів, формул і посилань відповідає вихідній статті.',
               r'\setlength{\tabcolsep}{6mm}',
               r'\begin{longtable}{@{}p{0.475\textwidth}|p{0.475\textwidth}@{}}',
               r'\Large\textbf{До} & \Large\textbf{Після}\\\hline\endhead']
    for left, right in zip(a, b):
        compare.append(r'\begin{minipage}[t]{\linewidth}' + '\n' + display(left) + '\n' + r'\end{minipage} &')
        compare.append(r'\begin{minipage}[t]{\linewidth}' + '\n' + display(right) + '\n' + r'\end{minipage} \\[5mm]')
    compare += [r'\end{longtable}', r'\clearpage', r'\section*{Література (спільна для обох колонок)}', bbl, r'\end{document}']
    standalone = [r'\documentclass[11pt,a4paper]{article}', r'\usepackage[margin=22mm]{geometry}', common, r'\begin{document}']
    for block in b:
        if block.startswith(r'\section') and len(standalone) > 4:
            standalone.append(r'\clearpage')
        standalone.append(display(block))
    standalone += [r'\clearpage', bbl, r'\end{document}']
    names = {'sections_1_3_UK_before_after': compare, 'sections_1_3_UK_simple': standalone}
    checks = {}
    for stem, parts in names.items():
        source = FOLDER / (stem + '.tex')
        source.write_text('\n'.join(parts) + '\n', encoding='utf-8-sig')
        rel = str(source.relative_to(PAPER))
        for _ in range(2):
            run('lualatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory=' + str(FOLDER.relative_to(PAPER)), rel)
        log = (FOLDER / (stem + '.log')).read_text(encoding='utf-8-sig')
        for problem in ('undefined citations', 'undefined references', 'Missing character:', 'Overfull \\vbox', 'Overfull \\hbox'):
            assert problem not in log, stem + ': ' + problem
        output = subprocess.check_output(['pdftotext', '-layout', str(FOLDER / (stem + '.pdf')), '-'], text=True)
        assert '???' not in output
        checks[stem] = {'pages': output.count('\f'), 'text_chars': len(output), 'no_unresolved_references_or_overfull_boxes': True}
        (FOLDER / (stem + '.txt')).write_text(output, encoding='utf-8-sig')
    (FOLDER / 'changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='before.tex', tofile='after.tex')), encoding='utf-8-sig')
    manifest = {'scope': 'Ukrainian sections 1--3 only', 'base': str(ORIGINAL.relative_to(ROOT)),
                'original_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                'preserved': {'labels': len(labels(after)), 'citation_commands': len(cites(after)), 'equations': len(equations(after)), 'tikz_unchanged': True},
                'aligned_blocks': len(a), 'pdfs': checks,
                'limitations': 'Editorial revision of the existing paper; no experiments, new data or validation of corrected rerun results.'}
    (FOLDER / 'verification.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8-sig')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()





