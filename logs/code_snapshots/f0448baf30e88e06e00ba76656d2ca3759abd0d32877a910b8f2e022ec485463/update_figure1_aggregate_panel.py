from pathlib import Path
import difflib
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/SMPT paper'
panel = r'''% (b) Aggregate community network
\node[font=\bfseries] at (5.0,2.25) {(b) Aggregate network};
\node[circle,draw=black!70,fill=black!5,line width=.8pt,
      minimum size=1.15cm,inner sep=0pt] (agg_i) at (3.65,1.0) {$V_i$};
\node[circle,draw=black!70,fill=black!5,line width=.8pt,
      minimum size=1.15cm,inner sep=0pt] (agg_j) at (6.35,1.0) {$V_j$};
\draw[-{Latex[length=3mm,width=3mm]},draw=black!65,line width=3pt]
      (agg_i.east) -- node[above=5pt,text=black] {$W_{ij}$} (agg_j.west);
\node at (3.65,0.15) {$\ibar_i$};
\node at (6.35,0.15) {$\sbar_j$};

'''
captions = {
    'main-SMPT.tex': (
        '(b) The two-scale simulator keeps a separate local simulator for each community. The macro layer receives community counts, uses $W_{ij}$ to generate a cross-community infection, and places it on a uniformly selected susceptible node.',
        '(b) In the aggregate network, each node represents a community, and the wide directed edge has weight $W_{ij}$, the total contact weight from $V_i$ to $V_j$. The macro layer uses this weight and the fractions $\\ibar_i$ and $\\sbar_j$; internal contact networks are retained by the local simulators.'),
    'main-SMPT_UK.tex': (
        '(b) Двомасштабний метод зберігає окремий локальний симулятор для кожної спільноти. Макрорівень отримує кількості вузлів у різних станах, використовує $W_{ij}$ для генерування міжспільнотного зараження та вибирає його ціль рівномірно серед сприйнятливих вузлів.',
        '(b) В агрегованій мережі кожен вузол представляє спільноту, а широке напрямлене ребро має вагу $W_{ij}$ --- загальну вагу контактів із $V_i$ до $V_j$. Макрорівень використовує цю вагу та частки $\\ibar_i$ і $\\sbar_j$; внутрішні контактні мережі зберігають локальні симулятори.'),
}
for name, (old_caption, new_caption) in captions.items():
    path = PAPER / name
    before = path.read_text(encoding='utf-8-sig')
    backup = ROOT / '.tmp' / (path.stem + '_before_aggregate_panel.tex')
    assert not backup.exists()
    backup.write_bytes(path.read_bytes())
    start = before.index('% (b) Two-scale representation')
    end = before.index('% (c) Surrogate interface', start)
    assert old_caption in before
    after = (before[:start] + panel + before[end:]).replace(old_caption, new_caption)
    # The surrounding panels and scientific equations are preserved exactly.
    assert before[before.index('% (a) Full microscopic interface'):start] == after[after.index('% (a) Full microscopic interface'):after.index('% (b) Aggregate community network')]
    old_c = before[before.index('% (c) Surrogate interface'):before.index(r'\end{tikzpicture}')]
    new_c = after[after.index('% (c) Surrogate interface'):after.index(r'\end{tikzpicture}')]
    assert old_c == new_c
    assert re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', before, re.S) == re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', after, re.S)
    path.write_text(after, encoding='utf-8')
    (ROOT / '.tmp' / (path.stem + '_aggregate_panel.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))), encoding='utf-8')
    engine = 'lualatex' if '_UK' in name else 'pdflatex'
    for _ in range(2):
        result = subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', name], cwd=PAPER, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-5000:]
    log = (PAPER / (path.stem + '.log')).read_text()
    title_line = next(i for i, line in enumerate(after.splitlines(), 1) if line == r'\maketitle')
    warnings = [line for line in log.splitlines() if any(x in line.lower() for x in ('undefined', 'missing character', 'multiply defined', 'overfull'))]
    assert not [x for x in warnings if x != 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(title_line)], warnings
    print(name + ': aggregate panel and caption updated, both LaTeX passes passed.')
