# Підготовка реальної мережі для SMPT

Основний документ: [article.pdf](article.pdf), редагований текст: [article.tex](article.tex). Це українське пояснення процесу, рішень і перевірок, без анотації та висновків. Основну SMPT-статтю не змінено.

Джерело: [SocioPatterns High School 2013](https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/), сенсорні контакти; [Mastrandrea, Fournet, Barrat (2015)](https://doi.org/10.1371/journal.pone.0136497). Умови даних: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Цитуйте джерело та SocioPatterns; не підміняйте умови даних ліцензією журнальної статті. URL і хеші оригіналів — у `sources.json`.

Побудовано 327 вузлів, 5818 ребер, 9 спільнот за класами. Вага ребра — сумарні контактні секунди / 363580 календарних секунд. Параметр передачі має узгоджуватися з цією одиницею часу. Дані не визначають його біологічного значення.

Збережено два істотні питання якості: у метаданих є ще два ID без контактів; клас ID 28 суперечить класу в контактному файлі. Основний варіант використовує 327 контактних вузлів і окрему таблицю метаданих. Альтернативи завантажуються явно:

```python
from paper.network_preparation.prepare_network import load_prepared

micro_graphs, full_graph, W, comm_nodes = load_prepared()
# Ті самі контакти, але класи з контактних записів:
alternative = load_prepared(class_source="contacts")
# Усі 329 ID; два додаткові вузли ізольовані:
with_metadata_only = load_prepared(population="metadata")
# Інша модель: однакові ваги всіх спостережених пар:
binary = load_prepared(weight_mode="binary")
```

Запускати Python із кореня репозиторію. Для іншого каталогу даних передайте його першим аргументом `load_prepared(folder)`.

Відтворення з кореня проєкту, без епідемічних симуляцій:

```bash
.venv/bin/python paper/network_preparation/prepare_network.py
.venv/bin/python paper/network_preparation/analyze_network.py
.venv/bin/python -m unittest discover -s paper/network_preparation -p 'test_preparation.py' -v
latexmk -cd -lualatex -interaction=nonstopmode -halt-on-error paper/network_preparation/article.tex
```

Залежності вже наявні в проєкті: NumPy, NetworkX, Matplotlib; для PDF — LuaLaTeX, українська Babel та DejaVu. Нові пакети не встановлювалися. Рисунки будуються лише зі збереженої мережі. Повторний запуск перевіряє хеші та пропускає готові результати. Після зміни підготовки потрібен **новий** каталог: `prepare_network.py --output paper/network_preparation/prepared_v2`. Старі результати зберігаються. Рисунковий скрипт прив'язаний до основного `prepared/`; для іншої версії потрібна явна зміна входу та нового виходу.

`prepared/` містить CSV, GraphML, порядок спільнот, показники та маніфест. GraphML експортує топологію й ваги основної версії; службові списки метаданих відновлює Python-завантажувач. `figures/analysis.json` містить структурні показники й один точний контрприклад інтенсивностей, обчислений за фіксованим станом. Це не запуск CTMC і не оцінка частоти помилок.

`preparation_runs.jsonl` зберігає життєвий цикл підготовки, включно з першою зупинкою через суперечність класів до введення явного правила. `process.md` пояснює послідовність рішень. `verification.json` фіксує завершальні перевірки.

Базове підключення до `MicroMacroSimulator` передбачає `alpha_mode="off"`, `correction_mode="off"`; нейтральні `alphas=1` не є калібруванням спектральної поправки. Повний симулятор отримує `full_graph` та `comm_nodes`; двомасштабний — `micro_graphs` та `W`. Для SI встановлюють нульове одужання. Значення швидкості передачі, горизонту, кроку й початкового зараження тут не призначали; симуляції не запускали.
