from pathlib import Path
import subprocess
import difflib

root = Path(__file__).resolve().parents[1]
paper = root / 'paper/SMPT paper'
panel = r'''% (b) Aggregate community network with local contacts retained
\node[font=\bfseries] at (5.0,2.25) {(b) Aggregate network};
\node[infected] (e1) at (3.35,1.20) {};
\node[person]   (e2) at (3.45,0.35) {};
\node[infected] (e3) at (4.25,0.82) {};
\node[person]   (e4) at (4.25,1.55) {};
\node[person]   (f1) at (5.65,1.15) {};
\node[person]   (f2) at (6.45,1.55) {};
\node[person]   (f3) at (6.55,0.55) {};
\node[person]   (f4) at (5.70,0.30) {};
\draw[link] (e1)--(e2)--(e3)--(e1) (e1)--(e4)--(e3);
\draw[link] (f1)--(f2)--(f3)--(f4)--(f1) (f1)--(f3);
\node[commbox,fit=(e1)(e2)(e3)(e4)] (agg_i) {};
\node[commbox,fit=(f1)(f2)(f3)(f4)] (agg_j) {};
\draw[-{Latex[length=2.5mm,width=2.5mm]},draw=black!65,line width=3pt]
      (agg_i.east) -- node[above=5pt,text=black] {$W_{ij}$} (agg_j.west);
\node at (3.8,-0.10) {$V_i$};
\node at (6.15,-0.10) {$V_j$};

'''
replacements = {
    'main-SMPT.tex': (
        '(b) In the aggregate network, each node represents a community, and the wide directed edge has weight $W_{ij}$, the total contact weight from $V_i$ to $V_j$. The macro layer uses this weight and the fractions $\\ibar_i$ and $\\sbar_j$; internal contact networks are retained by the local simulators.',
        '(b) Internal nodes and contacts are retained in each community. Individual inter-community contacts are represented by one wide directed edge between $V_i$ and $V_j$, labelled with their total weight $W_{ij}$.'),
    'main-SMPT_UK.tex': (
        '(b) В агрегованій мережі кожен вузол представляє спільноту, а широке напрямлене ребро має вагу $W_{ij}$ --- загальну вагу контактів із $V_i$ до $V_j$. Макрорівень використовує цю вагу та частки $\\ibar_i$ і $\\sbar_j$; внутрішні контактні мережі зберігають локальні симулятори.',
        '(b) Усередині кожної спільноти збережено вузли та локальні контакти. Окремі міжспільнотні контакти подано одним широким напрямленим ребром між $V_i$ та $V_j$ із загальною вагою $W_{ij}$.')
}
for name, (old_caption, new_caption) in replacements.items():
    path = paper / name
    before = path.read_text(encoding='utf-8-sig')
    backup = root / '.tmp' / (path.stem + '_before_local_aggregate_nodes.tex')
    assert not backup.exists()
    backup.write_bytes(path.read_bytes())
    start = before.index('% (b) Aggregate community network')
    end = before.index('% (c) Surrogate interface', start)
    assert old_caption in before
    after = (before[:start] + panel + before[end:]).replace(old_caption, new_caption)
    path.write_text(after, encoding='utf-8')
    (root / '.tmp' / (path.stem + '_local_aggregate_nodes.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))), encoding='utf-8')
    engine = 'lualatex' if '_UK' in name else 'pdflatex'
    for _ in range(2):
        result = subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', name], cwd=paper, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-5000:]
    log = (paper / (path.stem + '.log')).read_text()
    title_line = next(i for i,line in enumerate(after.splitlines(),1) if line == r'\maketitle')
    warnings = [line for line in log.splitlines() if any(x in line.lower() for x in ('undefined','missing character','multiply defined','overfull'))]
    assert not [x for x in warnings if x != 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(title_line)], warnings
    print(name + ': local nodes restored; build passed.')
