"""Build aligned Ukrainian before/after sections 1--5 PDFs; no experiments.

The extended document uses sections_1_5 output filenames; previous PDFs
remain intact and have additional copies in prior_sections_1_3.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/SMPT paper'
FOLDER = PAPER / 'plain_uk_before_after_20261009'
BASE = 'claude-version_merged_plain_sections_1_3_black_UK'
SEGMENTS = (
    ('before.tex', 'after.tex', PAPER / 'readability_revision_parallel_black_20261008/sections_1_3_UK.tex'),
    ('before_4_5.tex', 'after_4_5.tex', PAPER / 'claude-version_merged_parallel_uk_middle.tex'),
)


def run(*args):
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(' '.join(args) + '\n' + result.stdout[-5000:] + result.stderr[-1500:])


def blocks(text):
    out, current, depth = [], [], 0
    for line in text.splitlines():
        if line.startswith('%') or line in (r'\begingroup\color{black}', r'\endgroup', r'\clearpage') or line.startswith((r'\hypersetup', r'\enlargethispage')):
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


def heading_numbers(text):
    section, subsection, numbers = 0, 0, {}
    for kind, title in re.findall(r'\\(section|subsection)\{([^}]+)\}', text):
        if kind == 'section':
            section += 1
            subsection = 0
            number = str(section)
        else:
            subsection += 1
            number = f'{section}.{subsection}'
        assert (kind, title) not in numbers
        numbers[kind, title] = number
    return numbers


def main():
    labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
    cites = lambda s: re.findall(r'\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}', s)
    equations = lambda s: re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', s, re.S)
    old_parts, new_parts, originals, paired = [], [], {}, []
    segment_blocks = {}
    for old_name, new_name, original in SEGMENTS:
        snapshot = FOLDER / old_name
        assert snapshot.read_bytes() == original.read_bytes(), f'Original changed after snapshot: {original}'
        before = snapshot.read_text(encoding='utf-8-sig')
        after = (FOLDER / new_name).read_text(encoding='utf-8-sig')
        assert labels(before) == labels(after), new_name + ': label change'
        assert cites(before) == cites(after), new_name + ': citation change'
        assert equations(before) == equations(after), new_name + ': numbered equation change'
        a, b = blocks(before), blocks(after)
        assert len(a) == len(b), (old_name, len(a), len(b))
        paired.extend(zip(a, b))
        segment_blocks[old_name] = len(a)
        old_parts.append(before)
        new_parts.append(after)
        originals[str(original.relative_to(ROOT))] = hashlib.sha256(original.read_bytes()).hexdigest()
    before, after = '\n'.join(old_parts), '\n'.join(new_parts)
    tikz = lambda s: re.search(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', s, re.S).group()
    assert tikz(before) == tikz(after)
    assert re.findall(r'\\(section|subsection)\{', before) == re.findall(r'\\(section|subsection)\{', after)
    heads = heading_numbers(before)
    heads.update(heading_numbers(after))
    aux_path, bbl_path = PAPER / (BASE + '.aux'), PAPER / (BASE + '.bbl')
    aux = aux_path.read_text(encoding='utf-8-sig')
    numbers = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
    bbl = bbl_path.read_text(encoding='utf-8-sig')

    def display(text):
        for kind, translated in (('proposition', 'Твердження'), ('corollary', 'Наслідок'), ('remark', 'Зауваження')):
            pattern = r'\\begin\{' + kind + r'\}(?:\[([^\]]*)\])?\s*\\label\{([^}]+)\}'
            def theorem(m):
                title = ' (' + m[1] + ')' if m[1] else ''
                return r'\par\noindent\textbf{' + translated + ' ' + numbers[m[2]] + title + r'.}\par\noindent '
            text = re.sub(pattern, theorem, text)
            text = text.replace('\\end{' + kind + '}', r'\par')
        text = text.replace(r'\begin{proof}', r'\par\noindent\textit{Доведення.}\par\noindent ')
        text = text.replace(r'\end{proof}', r'\par')
        text = re.sub(r'\\(section|subsection)\{([^}]+)\}',
                      lambda m: '\\' + m[1] + '*{' + heads[m[1], m[2]] + '. ' + m[2] + '}', text)
        text = re.sub(r'\\eqref\{([^}]+)\}', lambda m: '(' + numbers[m[1]] + ')', text)
        text = re.sub(r'\\ref\{([^}]+)\}', lambda m: numbers[m[1]], text)
        def equation(m):
            body = m[1]
            label = re.search(r'\\label\{([^}]+)\}', body)[1]
            body = re.sub(r'\\label\{[^}]+\}', '', body).strip()
            return r'\begin{equation*}' + body + '\\tag{' + numbers[label] + '}\n' + r'\end{equation*}'
        text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', equation, text, flags=re.S)
        # Long formulas from the original column may exceed its width; scale only
        # these unnumbered displays. Revised proofs use short aligned formulas.
        text = re.sub(r'\\\[(.*?)\\\]', lambda m: r'\[\fitmath{' + m[1].strip() + r'}\]', text, flags=re.S)
        captions = {'table': 'Таблиця', 'figure': 'Рисунок', 'algorithm': 'Алгоритм'}
        for kind, translated in captions.items():
            if '\\begin{' + kind + '}' in text:
                match = re.search(r'\\label\{([^}]+)\}', text)
                label = match[1]
                text = text.replace(r'\caption{', r'\par\small\textbf{' + translated + ' ' + numbers[label] + '. ')
            text = re.sub(r'\\begin\{' + kind + r'\}(?:\[[^\]]*\])?', r'\\par\\noindent\\begin{minipage}{\\linewidth}', text)
            text = text.replace('\\end{' + kind + '}', r'\end{minipage}\par')
        text = re.sub(r'\\label\{[^}]+\}', '', text)
        text = text.replace(r'\begin{tabular}', r'\setlength{\tabcolsep}{3pt}\begin{tabular}')
        return text

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
\newcommand{\Full}{\textnormal{Full}}
\newcommand{\TS}{\textnormal{TwoScale}}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\eW}{e_W}
\newcommand{\qed}{\hfill\(\square\)}
\newsavebox{\mathbox}
\newcommand{\fitmath}[1]{%
  \sbox{\mathbox}{$\displaystyle #1$}%
  \ifdim\wd\mathbox>\linewidth
    \resizebox{\linewidth}{!}{\usebox{\mathbox}}%
  \else\usebox{\mathbox}\fi}
'''
    compare = [r'\documentclass[11pt]{article}', r'\usepackage[a3paper,landscape,margin=15mm]{geometry}', common,
               r'\begin{document}', r'\begin{center}\Large Розділи 1--5: простіша українська редакція\end{center}',
               r'Відповідні фрагменти вирівняно поруч. Нумерація розділів, формул і посилань відповідає вихідній статті.',
               r'\setlength{\tabcolsep}{6mm}',
               r'\begin{longtable}{@{}p{0.475\textwidth}|p{0.475\textwidth}@{}}',
               r'\Large\textbf{До} & \Large\textbf{Після}\\\hline\endhead']
    for left, right in paired:
        compare.append(r'\begin{minipage}[t]{\linewidth}' + '\n' + display(left) + '\n' + r'\end{minipage} &')
        row_end = r'\\*[5mm]' if left.startswith((r'\section', r'\subsection')) else r'\\[5mm]'
        compare.append(r'\begin{minipage}[t]{\linewidth}' + '\n' + display(right) + '\n' + r'\end{minipage} ' + row_end)
    compare += [r'\end{longtable}', r'\clearpage', r'\section*{Література (спільна для обох колонок)}', bbl, r'\end{document}']
    standalone = [r'\documentclass[11pt,a4paper]{article}', r'\usepackage[margin=22mm]{geometry}', common, r'\begin{document}']
    for _, block in paired:
        if block.startswith(r'\section') and len(standalone) > 4:
            standalone.append(r'\clearpage')
        standalone.append(display(block))
    standalone += [r'\clearpage', bbl, r'\end{document}']
    names = {'sections_1_5_UK_before_after': compare, 'sections_1_5_UK_simple': standalone}
    checks = {}
    for stem, parts in names.items():
        source = FOLDER / (stem + '.tex')
        source.write_text('\n'.join(parts) + '\n', encoding='utf-8')
        rel = str(source.relative_to(PAPER))
        for _ in range(2):
            run('lualatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory=' + str(FOLDER.relative_to(PAPER)), rel)
        log = (FOLDER / (stem + '.log')).read_text(encoding='utf-8-sig')
        for problem in ('undefined citations', 'undefined references', 'Missing character:', 'Overfull \\vbox', 'Overfull \\hbox'):
            assert problem not in log, stem + ': ' + problem
        output = subprocess.check_output(['pdftotext', '-layout', str(FOLDER / (stem + '.pdf')), '-'], text=True)
        assert '???' not in output
        for expected in ('4.1.', '4.2.', '4.3.', '4.4.', '5.1.', '5.2.', '5.3.', '5.4.', 'Твердження 1', 'Твердження 2'):
            assert expected in output, stem + ': missing ' + expected
        checks[stem] = {'pages': output.count('\f'), 'text_chars': len(output), 'no_unresolved_references_or_overfull_boxes': True}
        (FOLDER / (stem + '.txt')).write_text(output, encoding='utf-8')
    (FOLDER / 'changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='before_sections_1_5', tofile='after_sections_1_5')), encoding='utf-8')
    manifest = {'scope': 'Ukrainian sections 1--5; sections 1--3 prose retained',
                'originals_sha256': originals,
                'reference_inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (aux_path, bbl_path)},
                'preserved': {'labels': len(labels(after)), 'citation_commands': len(cites(after)), 'numbered_equations': len(equations(after)), 'tikz_unchanged': True},
                'aligned_blocks': len(paired), 'segment_blocks': segment_blocks, 'pdfs': checks,
                'limitations': 'Editorial revision of the current manuscript protocol; no experiments or numerical result replacement. Corrected chain-only outputs are a separate provenance.'}
    (FOLDER / 'verification.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
