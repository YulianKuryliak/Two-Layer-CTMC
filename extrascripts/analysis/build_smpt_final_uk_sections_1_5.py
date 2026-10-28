"""Assemble Ukrainian Sections 1--5 in the local SMPT CAS template; no experiments."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/SMPT paper'
OUT = PAPER / 'final_uk_sections_1_5_20261009'
NAME = 'main-SMPT_UK'
UK = PAPER / 'readability_revision_parallel_black_20261008/sections_1_3_UK.tex'


def read(path):
    return path.read_text(encoding='utf-8-sig')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:] + result.stderr[-2000:])


def render(text):
    text = '\n'.join(line for line in text.splitlines() if not (
        line.startswith('% !TeX root') or line in (
            r'\clearpage', r'\begingroup\color{black}', r'\endgroup',
            r'\hypersetup{citecolor=black,linkcolor=black}')))
    return (text.replace(r'\begin{algorithm}[H]', r'\begin{algorithm}[tbp]')
            .replace(r'\begin{table}[H]', r'\begin{table}[tbp]')
            .replace(r'\fbox{\begin{minipage}{0.94\linewidth}', '')
            .replace(r'\end{minipage}}', ''))


def main():
    OUT.mkdir(exist_ok=True)
    root_path = PAPER / (NAME + '.tex')
    verification = OUT / 'verification.json'
    if root_path.exists() and verification.exists():
        assert digest(root_path) == json.loads(read(verification))['root_sha256'], (
            'The main document was edited directly; compile it rather than overwrite its edits.')
    source34 = PAPER / 'plain_uk_before_after_20261009/after_4_5.tex'
    sections45 = OUT / 'sections_4_5_UK.tex'
    if not sections45.exists():
        text = read(source34)
        text = text.replace('ідеальний двомасштабний процес має генератор $\\mathcal{L}_{\\tilde G}$.',
                            'ідеальний двомасштабний процес має ті самі інтенсивності всіх переходів, що й повний процес на $\\tilde G$.')
        text = text.replace('приєднаємо до спільноти $V_0$ з $n$ вузлів',
                            'приєднаємо до зв’язної спільноти $V_0$ з $n$ вузлів')
        text = text.replace('За заданої траєкторії $X_0$ імовірності ще не заразити листок дорівнюють відповідно',
                            'За заданих траєкторії $X_0$ та набору мостів $B$ імовірність ще не заразити листок у повній моделі дорівнює')
        text = text.replace('та $\\exp(-\\beta_{\\mathrm{out}}(k/n)\\sum_uA_u(t))$.',
                            'а в двомасштабній моделі за заданої $X_0$ --- $\\exp(-\\beta_{\\mathrm{out}}(k/n)\\sum_uA_u(t))$.')
        text = text.replace('За нерівністю Єнсена усереднення її значень',
                            'За нерівністю Єнсена усереднення першого виразу за $B$')
        sections45.write_text(text.replace('% !TeX root = sections_1_5_UK_simple.tex',
                                           '% !TeX root = ../main-SMPT_UK.tex'), encoding='utf-8')
    first, last = read(UK), read(sections45)
    corrected = last.replace('A_u(t))$ а в двомасштабній', 'A_u(t))$, а в двомасштабній')
    if corrected != last:
        sections45.write_text(corrected, encoding='utf-8')
        last = corrected
    original45 = read(source34)
    equations = lambda s: [re.sub(r'\s+', '', x) for x in re.findall(
        r'\\begin\{equation\}(.*?)\\end\{equation\}', s, re.S)]
    assert equations(last) == equations(original45), 'Numbered mathematical statements changed'
    labels = re.findall(r'\\label\{([^}]+)\}', first + last)
    assert len(labels) == len(set(labels)), 'Duplicate labels'
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', first + last))
    missing = sorted(refs - set(labels))
    aux = read(PAPER / 'claude-version_merged_parallel_EN.aux')
    external = []
    for label in missing:
        match = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{([^}]+)\}\{([^}]+)\}', aux)
        assert match, label
        external.append('\\newlabel{' + label + '}{{' + match[1] + '}{' + match[2] + '}{}{}{}}')
    body = render(first) + '\n\n' + render(last)
    for label in missing:
        body = body.replace('\\ref{' + label + '}', '\\ref*{' + label + '}')
    preamble = read(PAPER / 'claude-version_merged_plain_sections_1_3_black_UK.tex').split(r'\begin{document}')[0]
    # Nimbus has Cyrillic but no small-cap face; choose the regular face explicitly.
    preamble = preamble.replace(r'\setmainfont{Nimbus Roman}',
                                r'\setmainfont{Nimbus Roman}[SmallCapsFont={Nimbus Roman}]')
    preamble += '\n\\renewcommand{\\figurename}{Рисунок}\n\\renewcommand{\\tablename}{Таблиця}\n'
    preamble += '\\renewcommand{\\refname}{Література}\n\\RenewDocumentCommand{\\keywordtitle}{}{Ключові слова}\n'
    preamble += '\\AtBeginDocument{\n' + '\n'.join(external) + '\n}\n'
    metadata = read(PAPER / 'readability_revision_parallel_black_20261008/uk_front.tex').split(r'\begin{abstract}')[0]
    abstract = r'''
% Попередня анотація; розділи з результатами ще не включено.
\begin{abstract}[Анотація]
Епідемічні моделі часто поєднують докладні контактні мережі всередині спільнот із загальними даними про зв'язки між ними. Ми описуємо двомасштабний симулятор для такої ситуації. Кожна спільнота має локальний симулятор Gillespie, а макрорівень генерує міжспільнотні зараження за агрегованими інтенсивностями та вибирає їхні цілі рівномірно серед сприйнятливих вузлів. Ми описуємо локальні годинники, правило імпорту й синхронізацію через фіксовані часові інтервали. Окремо розглядаємо ідеальний процес із точним визначенням часу подій: він моделює сурогатну мережу з рівномірними міжспільнотними контактами. Порівняння з повномережевим Gillespie досліджує, як це наближення змінює часи прибуття інфекції та їхні розподіли. Мета --- з'ясувати, коли агреговані зв'язки відтворюють потрібні показники, та пояснити джерела відмінностей.
\end{abstract}
\begin{keywords}
Стохастичне моделювання \sep Алгоритм Gillespie \sep Багатомасштабне моделювання \sep Мережі спільнот \sep Моделювання епідемій \sep Похибка наближення
\end{keywords}
\maketitle
'''
    root = preamble + r'\begin{document}' + '\n' + r'\let\WriteBookmarks\relax' + '\n' + metadata + abstract + body
    root += '\n\\bibliographystyle{cas-model2-names}\n\\bibliography{twoscale_references}\n\\end{document}\n'
    assert len(re.findall(r'\\section\{', root)) == 5
    assert r'\input{' not in root
    root_path.write_text(root, encoding='utf-8')
    (OUT / 'user_sections_1_3_UK_snapshot.tex').write_bytes(UK.read_bytes())
    for command in [('lualatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', NAME + '.tex'),
                    ('bibtex', NAME),
                    ('lualatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', NAME + '.tex'),
                    ('lualatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', NAME + '.tex')]:
        run(*command)
    run('pdftotext', '-layout', NAME + '.pdf', str(OUT / 'document.txt'))
    log = read(PAPER / (NAME + '.log'))
    warnings = [line for line in log.splitlines() if any(x in line.lower() for x in
                ('undefined', 'missing character', 'multiply defined', 'overfull'))]
    title_line = next(i for i, line in enumerate(root.splitlines(), 1) if line == r'\maketitle')
    allowed = 'Overfull \\hbox (117.0831pt too wide) detected at line ' + str(title_line)
    assert not [x for x in warnings if x != allowed], '\n'.join(warnings)
    fls = read(PAPER / (NAME + '.fls'))
    for filename in ('cas-sc.cls', 'cas-common.sty'):
        assert 'INPUT ./' + filename in fls
    result = dict(root_sha256=digest(root_path), user_sections_1_3_sha256=digest(UK),
                  sections_4_5_sha256=digest(sections45), preserved_numbered_equations=True,
                  section_count=5, labels=len(labels), omitted_section_references=missing,
                  build='LuaLaTeX, BibTeX, LuaLaTeX twice', warnings=warnings,
                  simulations_run=False)
    verification.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# Українська фінальна версія, розділи 1–5\n\n'
        'Головний файл: ../main-SMPT_UK.tex; PDF поруч. Компілятор: LuaLaTeX. '
        'Локальні cas-sc.cls, cas-common.sty, cas-model2-names.bst та twoscale_references.bib обов’язкові.\n\n'
        'Розділи 1–3 перенесено зі збереженої користувачем української версії без зміни змісту; '
        'прибрано лише оформлення документа для порівняння. Розділи 4–5 походять із after_4_5.tex: '
        'уточнено рівність інтенсивностей замість невизначеного генератора, умовність доведення '
        'та зв’язність джерела. Формули й параметри збережено. Анотація попередня. '
        'Посилання на розділи 6–7 та додатки зберігають номери повної статті; ці частини ще не включено.\n\n'
        'Збірка: .venv/bin/python extrascripts/analysis/build_smpt_final_uk_sections_1_5.py\n'
        'Після безпосереднього редагування main-SMPT_UK.tex компілюйте його LuaLaTeX; '
        'генератор захищає такі правки від перезапису.\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
