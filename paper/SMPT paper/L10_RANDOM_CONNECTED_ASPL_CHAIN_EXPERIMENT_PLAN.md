# ASPL experiment для ланцюга спільнот

Дата: 2026-09-19. Статус: завдання для реалізації; runner ще не створено, симуляції не запускалися. TEX-файл статті цим завданням не змінювати.

## 1. Мета

Перевірити, як час поширення вздовж ланцюга та похибка Production MicroMacro відносно Full Gillespie залежать від ASPL внутрішніх random-connected графів.

Порівнювати лише:

1. Full Gillespie;
2. Production MicroMacro з independent RNG для кожного `MicroModel`.

Independent RNG змінює лише ownership випадкових потоків. Production scheduler, midpoint integration, aggregate macro hazard, uniform placement, cache `discard` і normal repeated imports не змінювати.

## 2. Основна конфігурація

```text
macro topology = chain
communities L = 10
nodes per community n = 1000
initial infected node = node 0 у community 0
k = 1, 5 inter-community edges між кожною парою сусідніх communities
beta_in = beta_out = 0.003
gamma = 0
model = SI
tau_micro = 1
macro_T = 1
T_max = 30000
```

Production `.py` не змінювати.

## 3. Внутрішні графи

Кожна community має tree-augmented random-connected graph:

1. згенерувати випадкове labeled spanning tree на 1000 вузлах;
2. додати випадкові відсутні ребра до потрібного числа `m`;
3. точно обчислити ASPL;
4. прийняти граф лише тоді, коли фактичний ASPL входить у допуск.

| Target ASPL | Target mean degree `d` | Tolerance |
|---:|---:|---:|
| 2.0 | 50 | ±0.10 |
| 2.5 | 25 | ±0.10 |
| 3.0 | 12 | ±0.10 |
| 3.5 | 8 | ±0.12 |
| 4.0 | 6 | ±0.15 |
| 4.5 | 5 | ±0.15 |
| 5.0 | 4.3 | ±0.15 |

Після генерації кожного community graph перевірити:

- рівно 1000 вузлів із правильними labels;
- потрібне число ребер і mean degree;
- connectedness;
- simple graph без self-loops;
- одиничні ваги внутрішніх ребер;
- точний ASPL у допуску;
- успішний запис і повторне завантаження.

ASPL рахувати точно, а не за вибіркою BFS sources.

## 4. Replication design

Для кожного target ASPL:

```text
20 незалежних network realizations
× 5 interface/event replicates на кожній realization
= 100 paired runs на кожне k.
```

Одна network realization містить 10 незалежно згенерованих community graphs одного target ASPL.

Загальний обсяг:

```text
7 ASPL levels × 2 k values × 100 paired runs = 1400 pairs
1400 Full + 1400 MicroMacro = 2800 simulator runs.
```

## 5. Pairing, interfaces і seeds

Seeds задавати детерміновано:

```text
graph seed = (target_ASPL, network_id, community_id)
interface seed = (target_ASPL, network_id, replicate_id, chain_boundary)
event seed = (target_ASPL, network_id, replicate_id, simulator)
```

У межах однієї пари Full і MicroMacro повинні використовувати ті самі 10 community graphs, ті самі міжспільнотні ребра, той самий initial node та однакові epidemiological parameters. Full і MicroMacro мають окремі event RNG streams.

П'ять запусків на одній network realization не є дублікатами. Для `replicate_id=0..4` використовуються інші interface placements та інші event seeds.

Для кожної з дев'яти меж chain один детермінований впорядкований список inter-community edges задає обидва `k`:

```text
k=1 використовує перше edge;
k=5 використовує перші п'ять edges.
```

Отже, `k=1` є вкладеним у `k=5`. Не допускати duplicate edges або self-loops. Для кожної пари Full і MicroMacro interface повинен бути ідентичним.

## 6. Observables і збереження даних

Для кожної моделі в paired checkpoint зберігати:

```text
target_ASPL
actual_ASPL кожної community
network_id
replicate_id
k
graph seeds
interface seeds
event seed
T_0 ... T_9
delta_i = T_i - T_(i-1)
T_last
completion
termination reason
observation_end
runtime_seconds
```

`T_i` — перший справжній committed infection event у community `i`. Якщо community не заражена, `T_i=null`. `T_max` ніколи не записувати як arrival time.

Для SI також зберігати компактні infection events `(time, community, node)`. Цього має вистачати для відновлення `I_i(t)`, сумарної траєкторії та локальних кривих після alignment за arrival time без повторного моделювання. Не зберігати повну матрицю `time × node` і не створювати один великий consolidated CSV під час simulation run.

Якщо public simulator не підтримує ранню зупинку після досягнення останньої community, використовувати перевірений Production запуск до natural SI absorption або `T_max`. Не копіювати scheduler у runner лише для оптимізації.

## 7. Runner і структура output

Створити один runner:

```text
extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py
```

Дані записувати сюди:

```text
output/data/l10_n1000_random_connected_tree_augmented_aspl_chain/
  experiment_parameters.json
  topology_manifest.json
  graphs/
  run_manifest.json
  scenarios/
    macro_chain__L10__n1000__tree_augmented__d4p3__aspl5__k5__beta0p003__tau1__independent_rng/
      parameters.json
      run_manifest.json
      checkpoints/
```

Назва scenario folder має містити `macro_chain`, `L`, `n`, topology family, target `d`, target ASPL, `k`, beta, tau та independent RNG. Інші параметри записати у `parameters.json`.

Graph store не дублювати у checkpoints. Один checkpoint відповідає одній завершеній парі Full/MicroMacro і записується атомарно. Повторний запуск пропускає валідні checkpoints.

## 8. Режими запуску та gates

Runner повинен підтримувати:

```bash
python extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py --topology-only
python extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py --smoke-only --workers 1
python extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py --workers 2
```

Topology stage має бути resumable на рівні окремих graph files. Повний simulation run не запускати, якщо topology gate не пройшов.

Smoke виконати на reusable full-run keys для крайніх ASPL 2 і 5 та `k=1,5`. Перевірити regression gate, correctness усіх 10 community graphs і дев'яти interfaces, поширення вздовж chain, censoring, independent RNG metadata та memory usage.

При resume smoke автоматично не повторювати. Повний запуск не починати без прямої команди користувача.

## 9. Completion audit

Очікується 14 scenario folders: сім ASPL-рівнів × два значення `k`. У кожній має бути 100 paired checkpoints і `run_manifest.json` зі status `complete`.

Після збору перевірити:

- 20 distinct network IDs у кожній cell;
- по 10 валідних community graphs у кожній network realization;
- 5 distinct replicate IDs на network;
- `k=1` interfaces вкладені у `k=5` на кожній chain boundary;
- однакові graphs/interfaces для Full і MicroMacro всередині пари;
- censoring не перетворено на event;
- старі shared-RNG дані не підмішані.

## 10. Межа інтерпретації

ASPL тут змінюється через щільність, тому разом з ASPL змінюється mean degree. Це ASPL/density experiment, а не ізольований причинний тест ASPL.

Майбутній аналіз має показати arrival-time distributions по communities, signed error `T_i^MM-T_i^Full`, накопичення похибки вздовж chain, completion/censoring, локальні криві після arrival-time alignment та залежність похибки від фактичного ASPL і mean degree.

Аналіз і рисунки реалізувати окремими scripts після завершення збору. Simulation runner не повинен редагувати TEX статті.
