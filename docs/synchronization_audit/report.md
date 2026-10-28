# Synchronization audit — production MicroMacro проти event-driven aggregate scheduler

## Статус і походження даних

Повний аудит завершено для 100 парних реплікацій при `T_end=4000`; completion усіх варіантів дорівнює 1. Full Micro та event-driven aggregate/uniform варіант повторно використано з CE-01. Production MicroMacro виконано при `tau_micro ∈ {0.25, 0.5, 1}` на тих самих мережах і початкових вузлах. Збережено компактні часи прибуття без повних траєкторій. Provenance, контрольні суми та seed policy наведено в [manifest.json](../../output/missing_experiments/synchronization_audit/full/manifest.json).

## Наукове питання

Наскільки велика додаткова похибка production MicroMacro відносно event-driven реалізації того самого aggregate/uniform hazard model, і чи пояснюється вона вибором кроку синхронізації `tau_micro` у перевіреному діапазоні?

## Дизайн і оцінювана величина

Порівнюються дві реалізаційні семантики:

- `EventDrivenAggregate` — event-driven aggregate/uniform scheduler;
- `ProductionMM_tau*` — production midpoint/synchronization implementation при трьох значеннях `tau_micro`.

Full Gillespie CTMC є науковим референсом для повної апроксимаційної похибки; event-driven aggregate scheduler є внутрішнім референсом для різниці між реалізаціями aggregate model. Основний estimand: `ΔRMST = RMST_variant − RMST_reference`; від'ємне значення означає раніше прибуття варіанта. Наведені paired-bootstrap 95% CI та horizon-restricted `W1`. Через completion=1 цензурування не впливає на числові термінальні середні, але всі метрики визначено на горизонті 4000.

## Кількісні результати

### Відносно Full Micro

| Варіант | RMST | `ΔRMST` | 95% CI | `relW1` |
|---|---:|---:|---:|---:|
| FullMicro | 2936.73 | 0.00 | — | 0.0000 |
| EventDrivenAggregate | 2715.94 | -220.79 | [-314.05, -123.79] | 0.0752 |
| ProductionMM, `tau=0.25` | 1708.71 | -1228.03 | [-1384.49, -1069.44] | 0.4182 |
| ProductionMM, `tau=0.5` | 1695.17 | -1241.56 | [-1395.82, -1078.27] | 0.4228 |
| ProductionMM, `tau=1` | 1677.55 | -1259.18 | [-1417.17, -1100.70] | 0.4288 |

### Відносно event-driven aggregate scheduler

| Варіант | `ΔRMST` | 95% CI |
|---|---:|---:|
| ProductionMM, `tau=0.25` | -1007.23 | [-1152.95, -862.00] |
| ProductionMM, `tau=0.5` | -1020.77 | [-1175.33, -872.54] |
| ProductionMM, `tau=1` | -1038.39 | [-1216.63, -880.82] |

### Накопичення зсуву за шарами

| Шар | Event-driven проти Full | Production `tau=0.25` проти Full | Production `tau=1` проти Full |
|---:|---:|---:|---:|
| 1 | -47.2 | -46.7 | -46.7 |
| 3 | -103.8 | -235.7 | -238.3 |
| 5 | -121.0 | -513.7 | -539.1 |
| 7 | -162.5 | -831.8 | -863.1 |
| 9 | -220.8 | -1228.0 | -1259.2 |

## Аналіз і каузальна інтерпретація

Event-driven aggregate model має термінальне зміщення `-220.79`, тоді як production variants — приблизно `-1228`…`-1259`. Відносно event-driven scheduler додаткове раннє зміщення перевищує 1000 часових одиниць для кожного перевіреного `tau` і становить приблизно 80–82% signed production bias відносно Full Micro. Отже, у цій конфігурації production implementation додає велику похибку поверх уже наявної rate/placement abstraction error.

Мала відмінність point estimates між `tau=0.25`, `0.5` і `1` порівняно з їхньою спільною відстанню до event-driven variant вказує, що audited difference не можна інтерпретувати як звичайну похибку чисельної квадратури, яка очевидно зникає зі зменшенням кроку. Production і event-driven paths мають різні алгоритмічні семантики оновлення та синхронізації. Цей аудит вимірює їхню сукупну різницю; він не ізолює одну конкретну операцію.

Зсув майже однаковий у шарі 1, але різко накопичується після кількох переходів. Це узгоджується з повторним внесенням implementation discrepancy на кожному етапі ланцюга. Водночас наведені результати самі по собі не визначають, яка внутрішня складова production scheduler створює це накопичення.

## Значення для двошарової апроксимації та статті

- Похибку production MicroMacro не можна приписувати лише aggregate hazard/placement: реалізаційна семантика є окремим великим джерелом approximation error.
- Механістичні ablation-висновки з event-driven моделі не слід автоматично переносити на production implementation без цього розмежування.
- У статті доцільно розділити: (i) model abstraction error — Full проти event-driven aggregate; (ii) implementation/synchronization discrepancy — production проти event-driven aggregate.
- Для валідації виправлень потрібна не лише термінальна метрика, а й layerwise accumulation, оскільки ранні шари приховують подальше розходження.

## Обмеження і твердження, яких слід уникати

1. Перевірено лише `tau=0.25, 0.5, 1`; немає даних для послідовності, що прямує до нуля. Не слід заявляти ані збіжність, ані незбіжність production method при `tau→0`.
2. Між прямими контрастами `tau=0.25` проти `0.5` або `1` не обчислено paired CI. Перекривання окремих CI відносно Full чи event-driven не є тестом різниці між `tau`.
3. Порівнюються дві реалізаційні семантики, а не чиста quadrature rule за інших рівних умов; причинний внесок окремих операцій не ідентифіковано.
4. Одна SI-конфігурація і один горизонт не обґрунтовують універсальне твердження про production MicroMacro.
5. Completion=1 усуває цензурування лише в цьому наборі; за коротшого горизонту RMST і completion треба аналізувати спільно.
6. Аудит не оцінює runtime і не підтримує тверджень про швидкодію.

## Практичні рекомендації

1. Перед використанням production results у головних висновках узгодити semantics import scheduling, hazard accumulation і state synchronization з event-driven reference.
2. Додати контрольовані unit/trajectory tests на одному міжспільнотному переході, що фіксують момент накопичення hazard і момент застосування імпорту.
3. Після семантичного узгодження виконати окреме convergence study на ширшій, логарифмічній сітці `tau`, з прямими парними CI для різниць між сусідніми кроками.
4. У поточному рукописі подавати production discrepancy як окрему похибку реалізації, не як доказ неспроможності самої двошарової ідеї.

## Відтворюваність

Runner: [run_synchronization_audit.py](../../extrascripts/runners/run_synchronization_audit.py). Результати: [summary.csv](../../output/missing_experiments/synchronization_audit/full/summary.csv), [arrivals.csv](../../output/missing_experiments/synchronization_audit/full/arrivals.csv) і [реплікаційні записи](../../output/missing_experiments/synchronization_audit/full/replications/). Архівні референси: [CE-01 arrival_times.csv](../../output/ablation_hazard_placement/final_100/arrival_times.csv) та [runs.csv](../../output/ablation_hazard_placement/final_100/runs.csv). Параметри, pairing і SHA-256 runner/source наведено в [маніфесті](../../output/missing_experiments/synchronization_audit/full/manifest.json).
