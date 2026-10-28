"""Verify the direct-main translation of the user's current Sections 1--5."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/SMPT paper'
OUT = PAPER / 'final_en_sections_1_5_20261013'
EN = PAPER / 'main-SMPT.tex'
UK = PAPER / 'main-SMPT_UK.tex'
read = lambda path: path.read_text(encoding='utf-8-sig')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
active = lambda text: '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))

uk_hash = sha(UK)
assert UK.read_bytes() == (OUT / 'user_main-SMPT_UK_snapshot.tex').read_bytes()
before = read(OUT / 'main-SMPT_EN_before_update.tex')
after = read(EN)
uk = read(UK)
old_uk = read(PAPER / 'final_en_sections_1_4_20261013/user_main-SMPT_UK_snapshot.tex')
assert uk[uk.index(r'\section{Вступ}'):uk.index(r'\section{Методика експериментального')] == old_uk[old_uk.index(r'\section{Вступ}'):old_uk.index(r'\section{Як ми перевіряємо')]
# The earlier English prose is retained; only links to newly included material change.
first_before = active(before[before.index(r'\section{Introduction}'):before.index(r'\bibliographystyle')])
first_after = active(after[after.index(r'\section{Introduction}'):after.index(r'\section{Methodology for experimental evaluation}')])
assert first_before.strip() == first_after.strip(), 'Earlier English prose was rewritten'
after = re.sub(r'\\AtBeginDocument\{.*?\n\}\n', '', after, flags=re.S)
body_uk = active(uk[uk.index(r'\section{Вступ}'):uk.index(r'\bibliographystyle')])
body_en = active(after[after.index(r'\section{Introduction}'):after.index(r'\bibliographystyle')])
labels = re.findall(r'\\label\{([^}]+)\}', body_en)
assert labels == re.findall(r'\\label\{([^}]+)\}', body_uk)
assert len(labels) == len(set(labels))
for pattern in (r'\\begin\{equation\}(.*?)\\end\{equation\}', r'\\\[(.*?)\\\]',
                r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                r'\\cite[pt](?:\[[^\]]*\])*\{[^}]+\}',
                r'\\(?:section|subsection|paragraph)\{'):
    extract = lambda text: [re.sub(r'\s+', '', x) for x in re.findall(pattern, text, re.S)]
    assert extract(body_uk) == extract(body_en), pattern
assert len(re.findall(r'\\section\{', body_en)) == 5
assert r'\input{' not in after
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
EN.write_text(after, encoding='utf-8')
(OUT / 'main_changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='English main before Section 5', tofile='English main with user Section 5')), encoding='utf-8')
for args in [('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
             ('bibtex', 'main-SMPT'),
             ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'),
             ('pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex')]:
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
log = read(PAPER / 'main-SMPT.log')
titleline = next(i for i,line in enumerate(after.splitlines(),1) if line == r'\maketitle')
warnings = [line for line in log.splitlines() if any(x in line.lower() for x in
            ('undefined', 'missing character', 'multiply defined', 'overfull', 'rerun to'))]
assert not [x for x in warnings if x != 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(titleline)], warnings
fls = read(PAPER / 'main-SMPT.fls')
assert all('INPUT ./' + name in fls for name in ('cas-sc.cls', 'cas-common.sty'))
subprocess.run(['pdftotext', '-layout', 'main-SMPT.pdf', str(OUT / 'document.txt')], cwd=PAPER, check=True)
text = read(OUT / 'document.txt').translate(str.maketrans({chr(k):v for k,v in {11:'ff',12:'',27:'ff',28:'fi',29:'fl',30:'ffi',31:'ffl'}.items()}))
(PAPER / 'main-SMPT.txt').write_text(text, encoding='utf-8')
assert sha(UK) == uk_hash
result = dict(scope='User-edited active Sections 1--5 in current Ukrainian main',
              english_sha256=sha(EN), ukrainian_sha256=uk_hash,
              prior_english_sections_1_4_preserved=True,
              prior_reference_link_change='sec:design now links to included Section 5',
              sections=5, labels=len(labels), citations=len(re.findall(r'\\cite[pt]', body_en)),
              numbered_equations=len(re.findall(r'\\begin\{equation\}', body_en)),
              mathematical_and_heading_parity=True, tikz_unchanged=True,
              external_references=missing, warnings=warnings,
              editorial_notes='Two Ukrainian not-sure-needed headings preserved as invisible TEX comments',
              source_policy='Current user wording, including repeated rho paragraphs; no removed methodological details restored',
              ukrainian_unchanged=True, simulations_run=False)
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
