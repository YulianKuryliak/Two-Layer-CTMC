from pathlib import Path
import re
import json
import hashlib
import subprocess

root = Path(__file__).resolve().parents[1]
paper = root / 'paper/SMPT paper'
uk_path, en_path = paper / 'main-SMPT_UK.tex', paper / 'main-SMPT.tex'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
uk_hash = sha(uk_path)
uk = uk_path.read_text(encoding='utf-8-sig')
en = en_path.read_text(encoding='utf-8-sig')
uk = uk[uk.index(r'\section{Вступ}'):uk.index(r'\section{Як ми перевіряємо')]
en = en[en.index(r'\section{Introduction}'):en.index(r'\bibliographystyle')]
active = lambda text: '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))
uk, en = active(uk), active(en)
for pattern in [r'\\label\{[^}]+\}', r'\\cite[pt](?:\[[^\]]*\])*\{[^}]+\}',
                r'\\begin\{equation\}.*?\\end\{equation\}', r'\\\[.*?\\\]',
                r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                r'\\(?:section|subsection|paragraph)\{']:
    extract = lambda text: [re.sub(r'\s+', '', x) for x in re.findall(pattern, text, re.S)]
    assert extract(uk) == extract(en), pattern
assert r'\input{' not in en_path.read_text(encoding='utf-8-sig')
for _ in range(2):
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT.tex'], cwd=paper, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-5000:]
log = (paper / 'main-SMPT.log').read_text()
assert not [line for line in log.splitlines() if any(x in line.lower() for x in ('undefined', 'missing character', 'multiply defined', 'rerun to'))]
assert sha(uk_path) == uk_hash
record = dict(date='2026-10-13', scope='Active Sections 1-4 in current Ukrainian main',
              ukrainian_sha256=uk_hash, english_sha256=sha(en_path),
              semantic_review={'sections_1_2_4': 'Independent paragraph-by-paragraph read-only review: no material omissions or distortions',
                               'section_3': 'Primary agent paragraph-by-paragraph review of model, table/caption, local clocks, macro layer, timing, seven scheduler steps and cost caveats'},
              refinements=['Gillespie method instead of Gillespie-type method',
                           'Counts of people in different states in both introductory and related-work passages',
                           'Co-simulation explanation follows Ukrainian sentence order and explicit random-event statement',
                           'Table caption mirrors data and rules wording'],
              preserved=['Headings and paragraph heading structure', '28 labels', '25 citations', '8 numbered equations', 'All displayed equations', 'TikZ drawing'],
              intentional_differences=['Editorial ??? remark note is invisible TEX comment',
                                     'Malformed escaped X_0 is properly typeset', 'Duplicated respectively is translated once',
                                     'References to absent later sections are non-linking'],
              ukrainian_unchanged=True, build='Two successful pdfLaTeX passes', simulations_run=False)
out = paper / 'final_en_sections_1_4_20261013/semantic_audit.json'
out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, indent=2))
