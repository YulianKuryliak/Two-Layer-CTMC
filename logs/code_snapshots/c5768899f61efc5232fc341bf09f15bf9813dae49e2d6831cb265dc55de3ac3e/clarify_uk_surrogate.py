from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
paper = root / 'paper/SMPT paper'
path = paper / 'main-SMPT_UK.tex'
before = path.read_text(encoding='utf-8-sig')
backup = root / '.tmp/main-SMPT_UK_before_surrogate_naming.tex'
if backup.exists():
    assert backup.read_bytes() == path.read_bytes(), 'Preserve the existing backup'
else:
    backup.write_bytes(path.read_bytes())
changes = {
    'сурогатну мережу з рівномірними міжспільнотними контактами':
    'сурогатну мережу з рівномірними міжспільнотними зв’язками',
    r'Він точний для \emph{сурогатної мережі}: справжні міжспільнотні ребра замінено слабкими зв\'язками між усіма парами вузлів зі збереженням загальної ваги':
    r'Ідеальний процес точний для \emph{сурогатної мережі з рівномірними міжспільнотними зв’язками}: між кожними двома зв’язаними спільнотами всі вузли однієї з’єднано з усіма вузлами іншої, а загальну вагу зв’язку розподілено порівну між цими парами',
    'Наша сурогатна мережа має подібну ідею':
    'Наша сурогатна мережа з рівномірними міжспільнотними зв’язками має подібну ідею',
    'точний для сурогатної мережі, визначеної далі':
    'точний для сурогатної мережі з рівномірними міжспільнотними зв’язками, визначеної далі',
    'на сурогатній мережі, де кожна напрямлена пара вузлів між двома спільнотами має слабкий зв\'язок':
    'на сурогатній мережі з рівномірними міжспільнотними зв’язками: між кожними двома зв’язаними спільнотами всі напрямлені пари вузлів мають однакову вагу контакту',
    'точний для сурогатної мережі.':
    'точний для сурогатної мережі з рівномірними міжспільнотними зв’язками.',
    r'\subsection{Точність для усередненої мережі}':
    r'\subsection{Точність для мережі з рівномірними міжспільнотними зв’язками}',
    r'Побудуємо сурогатну мережу $\tilde G$: залишимо внутрішні контакти без змін, а справжні міжспільнотні ребра замінимо зв\'язками між усіма парами вузлів відповідних спільнот зі збереженням сумарної ваги.':
    r'Побудуємо сурогатну мережу з рівномірними міжспільнотними зв’язками $\tilde G$: внутрішні контакти та їхні інтенсивності залишимо без змін, а між кожними двома зв’язаними спільнотами з’єднаємо кожен вузол однієї з кожним вузлом іншої. Сумарну вагу міжспільнотних контактів розподілимо порівну між усіма цими парами.',
    r'\begin{proposition}[Двомасштаба модель є точною для сурогатної мережі]':
    r'\begin{proposition}[Точність для мережі з рівномірними міжспільнотними зв’язками]',
    r'Кожному новому каналу надамо інтенсивність $\tilde\beta_{uv}=\beta_{\mathrm{out}}W_{ij}/(N_iN_j)$.':
    r'Кожному новому каналу надамо інтенсивність $\tilde\beta_{uv}=\beta_{\mathrm{out}}W_{ij}/(N_iN_j)$, де $N_i$ та $N_j$ --- кількості вузлів у відповідних спільнотах.',
    r'У сурогатній мережі контактів значно більше.':
    r'У $\tilde G$ між зв’язаними спільнотами контактів загалом більше, ніж у початковій мережі.',
}
after = before
for old, new in changes.items():
    old = old.replace("\\'", "'")
    assert old in after, old
    after = after.replace(old, new)
for pattern in (r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                r'\\begin\{equation\}.*?\\end\{equation\}',
                r'\\label\{[^}]+\}'):
    assert re.findall(pattern, before, re.S) == re.findall(pattern, after, re.S)
path.write_text(after, encoding='utf-8')
for args in [('lualatex', '-interaction=nonstopmode', '-halt-on-error', 'main-SMPT_UK.tex')]*2:
    result = subprocess.run(args, cwd=paper, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-5000:]
log = (paper / 'main-SMPT_UK.log').read_text()
bad = [line for line in log.splitlines() if any(x in line.lower() for x in
       ('undefined', 'missing character', 'multiply defined'))]
assert not bad, bad
print('Updated Ukrainian main only; TikZ, equations and labels unchanged; LuaLaTeX passed twice.')
