# Oracle bridge ceiling — частковий oracle для source-side bridge hazard

## Статус і походження даних

Повний експеримент завершено: 100 парних реплікацій, `T_end=4000`, completion усіх чотирьох варіантів дорівнює 1. З архіву CE-01 повторно використано Full Micro і базовий aggregate/uniform MicroMacro; source-bridge oracle та degree proxy обчислено на тих самих мережах і початкових вузлах із детермінованими variant-specific потоками. Повні траєкторії не зберігалися, доступні компактні записи прибуття. Походження й контрольні суми зафіксовано в [manifest.json](../../output/missing_experiments/oracle_bridge_ceiling/full/manifest.json).

## Наукове питання

Яку частину похибки базового MicroMacro можна усунути, якщо aggregate hazard замінити інформацією про фактичну зараженість source-side bridge-вузлів, але залишити рівномірне розміщення імпорту в цільовій спільноті? Додатково перевіряється, чи може deployable proxy на основі внутрішнього ступеня відтворити цей ефект без oracle-доступу до bridge-стану.

## Дизайн і оцінювана величина

У ланцюгу з 10 спільнот порівняно:

- `FullMicro` — повна мережева Gillespie CTMC;
- `MM_uniform` — агрегований hazard і рівномірний susceptible-target placement;
- `MM_oracle_source_bridge` — hazard `beta ×` сума ваг міжспільнотних ребер із зараженим source endpoint `× S_j/N_j`, із тим самим рівномірним placement;
- `MM_degree_proxy` — `beta W_ij ×` зважена внутрішнім ступенем частка заражених у source community `× S_j/N_j`, також із рівномірним placement.

Основний estimand — парний `ΔRMST = RMST_variant − RMST_FullMicro`; від'ємне значення означає раніше прибуття варіанта. Також наведено horizon-restricted Wasserstein distance `W1` та `relW1 = W1/RMST_FullMicro`. Bootstrap 95% CI стосується `ΔRMST`. Оскільки completion=1, horizon restriction тут чисельно не замінює термінальні часи на `T_end`, але визначення estimand збережено.

## Кількісні результати

### Термінальний шар (`layer=9`)

| Варіант | RMST | `ΔRMST` проти Full | 95% CI | `relW1` | Point gap closure проти uniform |
|---|---:|---:|---:|---:|---:|
| FullMicro | 2936.73 | 0.00 | — | 0.00000 | — |
| MM_uniform | 2715.94 | -220.79 | [-315.00, -125.23] | 0.07518 | 0.0% |
| MM_oracle_source_bridge | 2820.63 | -116.10 | [-222.00, -15.15] | 0.04002 | 47.4% |
| MM_degree_proxy | 2594.81 | -341.93 | [-442.34, -241.38] | 0.11643 | -54.9% |

Point gap closure визначено як `(abs(error_uniform) − abs(error_variant))/abs(error_uniform)`. Для degree proxy від'ємне значення означає, що абсолютна point-похибка на 54.9% більша за похибку uniform baseline.

### Накопичення середнього зсуву за шарами

| Шар | Source-bridge oracle `ΔRMST` | Uniform `ΔRMST` | Degree proxy `ΔRMST` |
|---:|---:|---:|---:|
| 1 | -15.6 | -47.2 | -50.0 |
| 3 | -55.1 | -103.8 | -146.9 |
| 5 | -64.3 | -121.0 | -209.4 |
| 7 | -68.9 | -162.5 | -259.9 |
| 9 | -116.1 | -220.8 | -341.9 |

## Аналіз і каузальна інтерпретація

За point estimate source-side bridge oracle скорочує абсолютну термінальну похибку з 220.79 до 116.10, а `relW1` — з 0.07518 до 0.04002. Отже, фактична доступність заражених source gateways є важливою складовою rate-abstraction error. Водночас залишкова різниця `-116.10` має CI, що не включає нуль: виправлення лише source-side hazard не відтворює Full Micro у цьому дизайні.

Це **частковий**, а не повний oracle. Цільова susceptible fraction залишається агрегованою, а імпорт розміщується рівномірно, тому залишкова похибка сумісна щонайменше з target-side hazard abstraction і placement error. Експеримент не розділяє ці дві складові.

Degree proxy не наближає oracle ceiling: він посилює передчасне прибуття, причому відхилення накопичується вздовж ланцюга. Внутрішній ступінь, отже, не є достатнім surrogate для того, чи заражені саме endpoints міжспільнотних ребер. Це висновок для реалізованого proxy і цієї мережевої сім'ї, а не загальна неможливість deployable bridge-aware proxies.

## Значення для двошарової апроксимації та статті

- Bridge-aware source hazard є перспективним напрямом: він майже вдвічі зменшує point-похибку RMST і distributional error відносно uniform baseline.
- Результат задає емпіричну межу для **цього часткового виправлення**, а не загальну верхню межу якості MicroMacro.
- Для пояснення залишкової похибки потрібен факторний дизайн, що окремо змінює source hazard, target hazard і placement.
- Простий degree-weighted prevalence не слід пропонувати як готове виправлення: у наявному тесті він гірший за базовий aggregate hazard.

## Обмеження і твердження, яких слід уникати

1. Дизайн охоплює одну параметричну конфігурацію SI та один горизонт; переносимість на SIR, інші `k`, `d`, `beta` чи топології не перевірено.
2. Point gap closure 47.4% не має окремого bootstrap CI. Не слід стверджувати точну або статистично гарантовану частку «усуненої» похибки.
3. Oracle використовує мікроскопічний стан source bridge endpoints і тому не є безпосередньо deployable алгоритмом.
4. Однаковий uniform placement у MicroMacro arms ізолює відмінність hazard, але не оцінює виправлення placement.
5. Відсутність цензурування в цьому експерименті не гарантує її відсутності в інших горизонтах або режимах.
6. Не слід описувати degree proxy як доказ того, що центральність загалом неінформативна: перевірено лише конкретне зважування внутрішнім ступенем.

## Практичні рекомендації

1. Розробляти deployable state variable, що відстежує bridge exposure або зараженість gateway-підмножини, а не замінює її загальним ступенем.
2. Наступний ablation має додати target-side bridge availability та oracle placement, з парними контрастами між вкладеними варіантами.
3. У статті подавати разом `ΔRMST`, CI, completion і `relW1`; 47.4% називати лише point estimate gap closure.
4. Перевірити bridge-aware correction на режимній сітці, перш ніж формулювати загальне твердження про домінування цього механізму.

## Відтворюваність

Runner: [run_oracle_bridge_ceiling.py](../../extrascripts/runners/run_oracle_bridge_ceiling.py). Збережені результати: [summary.csv](../../output/missing_experiments/oracle_bridge_ceiling/full/summary.csv), [arrivals.csv](../../output/missing_experiments/oracle_bridge_ceiling/full/arrivals.csv) і [реплікаційні записи](../../output/missing_experiments/oracle_bridge_ceiling/full/replications/). Архівні baseline-дані: [CE-01 arrival_times.csv](../../output/ablation_hazard_placement/final_100/arrival_times.csv) та [runs.csv](../../output/ablation_hazard_placement/final_100/runs.csv). Seed policy, формули hazard, pairing і SHA-256 runner/source наведено в [маніфесті](../../output/missing_experiments/oracle_bridge_ceiling/full/manifest.json).
