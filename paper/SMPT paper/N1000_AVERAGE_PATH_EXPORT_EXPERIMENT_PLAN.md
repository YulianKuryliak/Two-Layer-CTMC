# Tree-augmented random-connected ASPL experiment, n=1000

Дата: 2026-09-19. Статус: runner підготовлено; повний експеримент не запускався. Це інструкція для виконання, а не частина TEX-рукопису.

## 1. Питання експерименту

Є одна random-connected спільнота на 1000 вузлів і один зовнішній leaf-вузол. У спільноті початково заражено вузол `0`. Leaf з'єднаний із `k` випадково вибраними bridge-вузлами.

Вимірюємо

```text
T_export = час першого зараження leaf.
```

На тому самому графі та з тим самим placement bridge-вузлів порівнюємо:

1. Full Gillespie;
2. Production MicroMacro з independent RNG для кожного `MicroModel`.

Мета — перевірити, як час експорту та похибка MicroMacro змінюються вздовж ASPL-сітки random-connected графів.

## 2. Як генерується random-connected graph

Для ASPL до 5 звичайний rejection-`G(n,m)` непридатний: при потрібній малій щільності майже всі графи незв'язні. Тому кожен граф будується так:

1. генерується випадкове labeled spanning tree на 1000 вузлах;
2. додаються випадкові відсутні ребра до заданої кількості `m`;
3. граф уже гарантовано connected;
4. приймається лише якщо його фактичний ASPL входить у допуск;
5. усі внутрішні ребра мають вагу 1.

Ця родина чесно називається `random_connected_tree_augmented`; це не тотожний `G(n,m)` із `network.py`.

ASPL — середня найкоротша відстань між усіма парами вузлів внутрішньої спільноти. Leaf у розрахунок ASPL не входить.

## 3. ASPL-сітка

Короткий topology pilot дав такі орієнтири:

| Target ASPL | Target mean degree `d` | Pilot ASPL |
|---:|---:|---:|
| 2.0 | 50 | ≈2.03 |
| 2.5 | 25 | ≈2.50 |
| 3.0 | 12 | ≈3.02 |
| 3.5 | 8 | ≈3.55 |
| 4.0 | 6 | ≈4.05–4.10 |
| 4.5 | 5 | ≈4.50 |
| 5.0 | 4.3 | ≈4.97 |

Runner приймає лише графи, фактичний ASPL яких входить у допуск:

```text
2.0 ± 0.10
2.5 ± 0.10
3.0 ± 0.10
3.5 ± 0.12
4.0 ± 0.15
4.5 ± 0.15
5.0 ± 0.15
```

Важлива межа інтерпретації: ASPL тут варіюється через зміну `m`, а отже разом з ASPL змінюється середній ступінь. Це ASPL/density sweep, а не доказ ізольованого причинного ефекту ASPL. У статті треба показувати і фактичний ASPL, і `d`.

## 4. Інші параметри

```text
n = 1000
k = 1, 5
beta_in = beta_out = 0.003
gamma = 0
model = SI
tau_micro = 1
macro_T = 1
T_max = 30000
paired replications per (ASPL, k) cell = 100
```

`T_max` — верхня межа спостереження. Якщо export не відбувся, записується `T_export = null`; значення 30000 не стає arrival time.

## 5. Replication design

Для кожного ASPL-рівня:

```text
20 незалежних tree-augmented random-connected graphs
× 5 незалежних interface placements на кожному графі
× 1 stochastic event replicate
= 100 paired runs на кожне k.
```

Для одного `(target_ASPL, graph_id, interface_id)`:

- `k=1` використовує перший bridge-вузол;
- `k=5` використовує перші п'ять bridge-вузлів;
- тому `k=1` є підмножиною `k=5`;
- Full і MicroMacro використовують той самий graph та interface.

Загальний обсяг:

```text
7 ASPL levels × 2 k values × 100 paired runs = 1400 pairs
1400 Full + 1400 MicroMacro = 2800 simulator runs.
```

## 6. Єдиний runner

```text
extrascripts/runners/run_n1000_aspl_export_independent_rng.py
```

Параметри, які користувач має бачити, розміщено на початку файла. Не створювати додаткових версій runner-а або прихованих конфігурацій.

Production `.py` не змінювати. Independent RNG змінює лише ownership RNG: scheduler, midpoint integration, aggregate hazard, uniform placement, cache `discard` та repeated imports `normal` залишаються незмінними.

## 7. Організація даних

```text
output/data/n1000_random_connected_tree_augmented_aspl_export/
  experiment_parameters.json
  graphs/
    random_connected_tree_augmented_aspl2/
    random_connected_tree_augmented_aspl2p5/
    ...
  run_manifest.json
  scenarios/
    topology_random_connected_tree_augmented__n1000__d50__aspl2__k1__beta0p003__tau1__independent_rng/
      parameters.json
      run_manifest.json
      checkpoints/
    topology_random_connected_tree_augmented__n1000__d4p3__aspl5__k5__beta0p003__tau1__independent_rng/
      parameters.json
      run_manifest.json
      checkpoints/
```

Назва scenario folder містить topology, `n`, target `d`, target ASPL, `k`, beta, tau та independent RNG. У `parameters.json` записані всі інші параметри, replication design і censoring policy.

Фактичні ASPL, degree distribution, clustering, diameter, generation attempts і graph seeds зберігаються у:

```text
output/analysis/n1000_random_connected_tree_augmented_aspl_export/topology_calibration_manifest.json
```

Один checkpoint відповідає одній парі Full/MicroMacro і записується атомарно. Повторний запуск пропускає валідні checkpoints.

## 8. Що Terra може виконувати

Terra не змінює scientific design. Її завдання — лише механічна перевірка й запуск за прямою вказівкою користувача.

Синтаксис:

```bash
.venv/bin/python3 -m py_compile extrascripts/runners/run_n1000_aspl_export_independent_rng.py
```

Topology stage:

```bash
.venv/bin/python3 extrascripts/runners/run_n1000_aspl_export_independent_rng.py --topology-only
```

PASS означає: 140 accepted graphs, по 20 на кожен ASPL-рівень; усі connected; фактичний ASPL у допуску; graph files читаються після запису.

Smoke після topology PASS:

```bash
.venv/bin/python3 extrascripts/runners/run_n1000_aspl_export_independent_rng.py --smoke-only --workers 1
```

Smoke запускає по два reusable `k=5` checkpoints на кожен ASPL-рівень, разом 14 пар. Перевірити regression gate, обидва simulators, independent RNG metadata, `Full bypass = false` і коректне censoring.

Повний запуск Terra не починає самостійно. Команда користувача:

```bash
.venv/bin/python3 extrascripts/runners/run_n1000_aspl_export_independent_rng.py --workers 2
```

## 9. Перевірка завершення

Очікується 14 scenario folders: сім ASPL-рівнів × два значення `k`. У кожній має бути 100 paired checkpoints і `run_manifest.json` зі status `complete`.

Перевірити:

- 20 distinct graph IDs та 5 interfaces на graph у кожній cell;
- однакові graph/interface/parameters для Full і MicroMacro в парі;
- `k=1` interface вкладений у `k=5`;
- Full bypass fraction дорівнює нулю;
- незавершені exports мають null time;
- shared-RNG або старі дані не підмішані.

Аналіз і рисунки — окремий наступний етап. Runner збору даних не повинен редагувати TEX статті.
