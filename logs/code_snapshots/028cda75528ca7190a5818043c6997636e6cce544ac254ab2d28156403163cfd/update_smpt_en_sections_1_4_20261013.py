"""Update the English main from reviewed current Ukrainian Sections 1--4.

Direct main-file edits are protected; this is a document build, not an experiment.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/SMPT paper'
OUT = PAPER / 'final_en_sections_1_4_20261013'


def read(path):
    return path.read_text(encoding='utf-8-sig')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active(text):
    return '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))


def run(*args):
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:] + result.stderr[-2000:])


def main():
    uk_path = PAPER / 'main-SMPT_UK.tex'
    en_path = PAPER / 'main-SMPT.tex'
    uk = read(uk_path)
    uk_sha = sha(uk_path)
    before = read(en_path)
    backup = OUT / 'main-SMPT_EN_before_update.tex'
    snapshot = OUT / 'user_main-SMPT_UK_snapshot.tex'
    verification = OUT / 'verification.json'
    if backup.exists():
        assert verification.exists(), 'An earlier update exists; inspect it before regeneration.'
        previous = json.loads(read(verification))
        assert sha(en_path) == previous['english_root_sha256'], 'English main was edited directly.'
        assert uk_sha == previous['ukrainian_root_sha256'], 'Ukrainian main has new edits to translate.'
    else:
        backup.write_bytes(en_path.read_bytes())
        snapshot.write_bytes(uk_path.read_bytes())
    first_uk = uk[uk.index(r'\section{Вступ}'):uk.index(r'\section{Теоретичний')]
    # Sections 1--3 were already translated; their current Ukrainian prose matches
    # the last transfer snapshot, apart from the separately synchronised figure.
    baseline = read(PAPER / 'final_en_until_single_export_20261009/user_UK_until_single_export_snapshot.tex')
    baseline = baseline[:baseline.index(r'\section{Що насправді')]
    strip_figure = lambda text: re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.S)
    assert re.sub(r'\s+', '', strip_figure(first_uk)) == re.sub(r'\s+', '', strip_figure(baseline)), 'Review new Sections 1--3 prose before updating.'
    start = before.index(r'\section{What the two-scale method actually simulates}') if r'\section{What the two-scale method actually simulates}' in before else before.index(r'\section{Theoretical analysis')
    end = before.index(r'\bibliographystyle', start)
    after = before[:start] + read(OUT / 'section_4_EN.tex') + '\n\n' + before[end:]
    # Keep external numbers only for material absent from the current scope.
    after = re.sub(r'\\AtBeginDocument\{.*?\n\}\n', '', after, flags=re.S)
    body_uk = active(uk[uk.index(r'\section{Вступ}'):uk.index(r'\section{Як ми перевіряємо')])
    body_en = active(after[after.index(r'\section{Introduction}'):after.index(r'\bibliographystyle')])
    labels = re.findall(r'\\label\{([^}]+)\}', body_en)
    assert labels == re.findall(r'\\label\{([^}]+)\}', body_uk), 'Label sequence differs'
    assert len(labels) == len(set(labels)), 'Duplicate labels'
    citations = lambda text: re.findall(r'\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}', text)
    assert citations(body_uk) == citations(body_en), 'Citation sequence differs'
    for pattern in (r'\\begin\{equation\}(.*?)\\end\{equation\}', r'\\\[(.*?)\\\]',
                    r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}'):
        extract = lambda text: [re.sub(r'\s+', '', item) for item in re.findall(pattern, text, re.S)]
        assert extract(body_uk) == extract(body_en), 'Math/figure parity failed: ' + pattern
    headings = lambda text: re.findall(r'\\(section|subsection|paragraph)\{', text)
    assert headings(body_uk) == headings(body_en)
    assert len(re.findall(r'\\section\{', body_en)) == 4
    assert r'\label{sec:descriptor}' not in body_en
    refs = set(re.findall(r'\\(?:ref\*?|eqref)\{([^}]+)\}', body_en))
    missing = sorted(refs - set(labels))
    aux = read(PAPER / 'claude-version_merged_parallel_EN.aux')
    external = []
    for label in missing:
        match = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{([^}]+)\}\{([^}]+)\}', aux)
        assert match, label
        external.append('\\newlabel{' + label + '}{{' + match[1] + '}{' + match[2] + '}{}{}{}}')
    for label in labels:
        after = after.replace('\\ref*{' + label + '}', '\\ref{' + label + '}')
    for label in missing:
        after = after.replace('\\ref{' + label + '}', '\\ref*{' + label + '}')
    after = after.replace(r'\begin{document}', '\\AtBeginDocument{\n' + '\n'.join(external) + '\n}\n' + r'\begin{document}', 1)
    en_path.write_text(after, encoding='utf-8')
    (OUT / 'english_main_changes.diff').write_text(''.join(difflib.unified_diff(read(backup).splitlines(True), after.splitlines(True), fromfile='main-SMPT.tex before', tofile='main-SMPT.tex updated')), encoding='utf-8')
    for args in [('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
                 ('bibtex', 'main-SMPT'),
                 ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
                 ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex')]:
        run(*args)
    run('pdftotext', '-layout', 'main-SMPT.pdf', str(OUT / 'document.txt'))
    log = read(PAPER / 'main-SMPT.log')
    warnings = [line for line in log.splitlines() if any(x in line.lower() for x in
                ('undefined', 'missing character', 'multiply defined', 'overfull'))]
    title_line = next(i for i, line in enumerate(after.splitlines(), 1) if line == r'\maketitle')
    assert not [x for x in warnings if x != 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(title_line)], warnings
    fls = read(PAPER / 'main-SMPT.fls')
    assert all('INPUT ./' + name in fls for name in ('cas-sc.cls', 'cas-common.sty'))
    assert sha(uk_path) == uk_sha, 'Ukrainian main must remain unchanged'
    result = dict(scope='Current active Ukrainian Sections 1--4 only',
                  english_root_sha256=sha(en_path), ukrainian_root_sha256=uk_sha,
                  sections=4, labels=len(labels), citations=len(citations(body_en)),
                  numbered_equations=len(re.findall(r'\\begin\{equation\}', body_en)),
                  display_equation_and_tikz_parity=True, heading_parity=True,
                  excluded='Section 5 and commented temporal-descriptor drafts',
                  editorial_cleanup=['Literal escaped X_0 rendered in math', 'Remark editorial question kept as TEX comment', 'Repeated respectively removed'],
                  external_references=missing, warnings=warnings, simulations_run=False)
    verification.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# English main, current Sections 1--4\n\n'
        'Main document: ../main-SMPT.tex; PDF: ../main-SMPT.pdf. Native SMPT CAS formatting retained.\n\n'
        'Source of truth: current Ukrainian main, archived exactly in user_main-SMPT_UK_snapshot.tex. '
        'Sections 1--3 were checked against the previously translated text; their figure already matches. '
        'Section 4 was replaced with a translation of its active current text. '
        'Section 5 and both commented temporal-descriptor drafts are excluded. '
        'The editorial question in the remark heading is retained only as a comment; literal escaped X_0 is typeset correctly.\n\n'
        'Build directly in the SMPT paper folder: pdflatex main-SMPT.tex; bibtex main-SMPT; pdflatex twice. '
        'After direct main edits, do not regenerate without reconciling them. The update script guards source hashes. '
        'References to Sections 5--7 retain full-paper numbers; these sections are not yet included.\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
