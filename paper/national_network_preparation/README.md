# Велика мережа міст і громад: Франція

[Пояснення PDF](article.pdf) · [редагований LaTeX](article.tex). Окремий документ українською, без анотації та висновків. Попередню шкільну мережу й головну SMPT-статтю не змінено.

**Побудовано 34 858 комун, 743 711 неорієнтованих зважених зв'язків, 100 спільнот-департаментів.** Охоплення: метрополія та чотири заморські департаменти Франції, без Майотти. Географія — 01.01.2025; оцінки зайнятості — 2022.

Source: Insee. [Повний CSV потоків проживання → роботи](https://www.insee.fr/fr/statistiques/8582949), випуск 26.06.2025; [COG 2025](https://www.insee.fr/fr/information/8377162), сторінка оновлена 04.03.2025. [Умови повторного використання](https://www.insee.fr/fr/information/2008466): Licence Ouverte / Open Licence 2.0 за відсутності окремих вказівок. Незмінені оригінали — у `raw/`, URL і SHA-256 — у `sources.json`.

Тут вузол — **комуна, не людина**; спільнота — департамент. Первинна напрямлена матриця `F_domestic.npz` містить оцінки працівників за парою місць проживання й роботи. Для узгодженого входу поточних неорієнтованих симуляторів побудовано `A_proxy.npz`: `A[u,v]=(F[u,v]+F[v,u])/1000`, без петель. Це проксі просторової зв'язності, не виміряні контакти та не щоденна кількість поїздок. `W` отримана саме з міждепартаментних сум `A`; описова напрямлена матриця департаментів не підміняє її.

**Структура даних:** 167 ізольованих комун; 56 незв'язних локальних графів. Вони збережені, штучних ребер немає. Порівняння майбутніх симуляцій має враховувати досяжність, цензурування й допустимість маршрутів. Значення біологічної передачі не калібрувалися; симуляції не запускалися.

У `prepared/`:

- `nodes.csv`, `departments.csv` — узгоджені ID, географічні коди, належність і структурні показники.
- `F_domestic.npz` — напрямлені емпіричні оцінки з діагоналлю; `A_proxy.npz` — симетричний граф без петель.
- `W.npy`, `W.csv` — матриця 100×100 з нульовою діагоналлю.
- `department_directed_flows_descriptive.*` — описові напрямлені суми, **не макровхід узгодженого порівняння з A**.
- `geography_crosswalk.csv` — відповідність муніципальних округів батьківським містам; вихідні коди зберігаються як рядки.
- `outside_scope_flows.csv.gz` — усі 29 496 відокремлених рядків та причини відокремлення.
- `summary.json`, `components.csv`, `top_links.csv`, `manifest.json` — результати перевірок, версії, конфігурація, хеші.

Завантаження в корені проєкту:

```python
from paper.national_network_preparation.prepare_network import load_sparse
nodes, departments, A, W = load_sparse()

# Явне розгортання у NetworkX для наявних API:
from paper.national_network_preparation.prepare_network import load_networkx
micro_graphs, full_graph, W, comm_nodes = load_networkx()
```

NetworkX-завантажувач використовує один повний граф та локальні представлення без копій усіх ребер, додає `community_sizes` і нейтральні `alphas=1`. Базове підключення потребує `alpha_mode="off"`, `correction_mode="off"`; для SI — нульового одужання. `load_sparse` залишає дані компактними; перетворення в NetworkX збільшує пам'ять. Не передавайте 34 858×34 858 як щільну W поточному макрорушію.

Відтворення з кореня репозиторію:

```bash
.venv/bin/python paper/national_network_preparation/fetch_sources.py
.venv/bin/python paper/national_network_preparation/prepare_network.py
.venv/bin/python paper/national_network_preparation/analyze_network.py
.venv/bin/python -m unittest discover -s paper/national_network_preparation -p 'test_preparation.py' -v
latexmk -cd -lualatex -interaction=nonstopmode -halt-on-error paper/national_network_preparation/article.tex
```

Це завантаження, детермінована підготовка, аналіз збереженого графа, структурні тести й PDF. Жодна команда не запускає CTMC. Наявні дані та рисунки перевіряються за хешами й пропускаються. Після зміни коду, даних або масштабу використайте новий каталог: `prepare_network.py --output paper/national_network_preparation/prepared_v2 --flow-scale 1000`. Рисунковий скрипт свідомо прив'язаний до основного `prepared/`; для іншої версії треба явно змінити вхід і вибрати новий вихід. Конфігурація й початок/завершення/skip збережені у `preparation_runs.jsonl`.

Використані вже наявні NumPy, Pandas, SciPy, NetworkX, Matplotlib; нові пакети не встановлювалися. Для PDF потрібні LuaLaTeX, Babel із locale `ukrainian` та DejaVu.

Останній розділ пояснює іншу постановку: люди всередині міст, міста як макроспільноти. Вона потребує локальних контактних мереж, населення та моделі перетворення переміщень на контакти; наявний граф комун не видається за таку реконструкцію.
