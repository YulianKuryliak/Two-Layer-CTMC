# Gateway separation audit — near, random і far gateway architecture

## Статус і походження даних

Повний експеримент завершено без failed replications: 100 парних мережевих блоків, `L=5`, `n=100`, `d=6`, `k=5`, `beta=0.003`, SI, `T_end=2000`. Внутрішні графи й multiset внутрішніх ступенів gateway-вузлів спарені між режимами. MicroMacro не використовує gateway identities, тому його траєкторію повторно використано для `near`, `random` і `far` у межах реплікації. Збережено тільки часи прибуття. Повний опис дизайну містить [manifest.json](../../output/missing_experiments/gateway_distance/full/manifest.json).

## Наукове питання

Як архітектура розміщення вхідних і вихідних gateway-вузлів усередині проміжної спільноти змінює похибку двошарової апроксимації? Зокрема, чи зростає розходження між Full Micro і MicroMacro, коли source-to-next-layer gateways обираються далеко від gateways, через які інфекція входить у спільноту?

## Дизайн і оцінювана величина

У кожній із трьох проміжних спільнот фіксується множина left gateways. Множина right gateways із тим самим multiset внутрішніх ступенів обирається за однією з політик:

- `near` — кандидати з найменшою відстанню до left set;
- `random` — випадкові кандидати;
- `far` — кандидати з найбільшою відстанню до left set.

Кількість міжспільнотних ребер і всі внутрішні ребра незмінні. Дескриптори `D_min` та `D_mean` — середні за проміжними спільнотами мінімальна і середня попарна відстань між left/right gateway sets.

Основний estimand: `ΔRMST = RMST_MicroMacro − RMST_Micro`; від'ємне значення означає раніше прибуття MicroMacro. Незавершені до `T_end` прибуття замінено на 2000, тому всі оцінки є horizon-restricted. Міжрежимні контрасти — різниці цього error, ресемпловані за 100 paired network blocks.

## Кількісні результати

### Gateway descriptors

| Політика | Середнє `D_min` | Середнє `D_mean` | Перетин left/right sets у проміжних комірках |
|---|---:|---:|---:|
| near | 0.0000 | 2.2003 | 300/300 |
| random | 0.5567 | 2.6475 | 164/300 |
| far | 1.9033 | 2.9892 | 24/300 |

Знаменник 300 відповідає трьом проміжним спільнотам у кожній зі 100 мережевих реплікацій. `D_min=0` і 300/300 overlaps у near показують, що ця політика систематично допускає спільні left/right gateway-вузли.

### Термінальний шар (`layer=4`)

MicroMacro має однаковий restricted mean 983.06 в усіх режимах, бо aggregate model не бачить gateway placement.

| Політика | Restricted mean Micro | Completion Micro | Completion MM | `ΔRMST` MM−Micro | `relW1` |
|---|---:|---:|---:|---:|---:|
| near | 905.42 | 1.00 | 1.00 | +77.64 | 0.09235 |
| random | 1265.17 | 1.00 | 1.00 | -282.11 | 0.22298 |
| far | 1437.87 | 0.96 | 1.00 | -454.81 | 0.31631 |

Для far значення 1437.87 і `ΔRMST=-454.81` є restricted до 2000: чотири незавершені Micro-прогони входять як 2000. У шарі 3 completion Micro для far вже дорівнює 0.99, тоді як MicroMacro дорівнює 1.00. У шарі 1 результати однакові між політиками, оскільки endpoint gateway sets не змінювалися; маніпуляція починається у проміжних спільнотах.

### Парні міжрежимні контрасти термінальної похибки

| Контраст error | Оцінка | 95% CI |
|---|---:|---:|
| random − near | -359.74 | [-413.68, -305.50] |
| far − near | -532.45 | [-586.68, -478.01] |
| far − random | -172.70 | [-230.57, -114.84] |

Усі три CI не включають нуль у цьому наборі. Зі збільшенням separation/зменшенням overlap Full Micro прибуває пізніше, тоді як MicroMacro залишається незмінним, і тому error стає більш від'ємним.

## Аналіз і каузальна інтерпретація

Результат демонструє структурну сліпоту базової двошарової апроксимації: однакові community counts і однакова матриця `W` не кодують, наскільки легко інфекція може пройти від import gateways до export gateways усередині спільноти. Near architecture створює коротший внутрішній шлях і навіть може повторно використовувати ті самі вузли; far architecture вимагає більшого внутрішнього поширення перед наступним export. MicroMacro не розрізняє ці випадки, тому його термінальний mean незмінний.

Парування внутрішніх графів, кількості bridge edges і within-degree multiset усуває важливі confounds. Проте політики змінюють не лише скалярну відстань: вони різко змінюють overlap left/right sets і можуть змінювати інші позиційні властивості, зокрема centrality відносно gateway sets. Тому causal treatment слід називати **gateway separation/overlap architecture**, а не чистим distance-only effect.

Позитивний error у near (`+77.64`) і від'ємний у random/far показують, що одна aggregate параметризація може як запізнюватися, так і випереджати Full Micro залежно від прихованої gateway geometry. Це важливіше за єдину середню похибку: знак approximation bias не є інваріантним до мезоскопічної архітектури.

## Значення для двошарової апроксимації та статті

- Матриці community-level edge counts недостатньо для прогнозу arrival-time error, коли внутрішня відстань між import/export gateway sets змінюється.
- Gateway geometry є окремим джерелом rate-abstraction error: aggregate hazard активується за community prevalence, не враховуючи час досягнення export boundary.
- Для bridge-aware моделі варто додати стан доступності export gateways або crossing-delay surrogate, що залежить від import/export architecture.
- У статті цей експеримент краще подавати як paired structural stress test, а не як універсальну монотонну залежність від graph distance.

## Обмеження і твердження, яких слід уникати

1. Far terminal results мають 4% цензурування Micro і є чинними лише як restricted-at-2000 estimands. Їх не можна порівнювати з необмеженим mean arrival time або з експериментами іншого горизонту як однакову величину.
2. Overlap і потенційні centrality differences не контрольовано окремо від distance policy. Не слід заявляти ідентифікацію «чистого ефекту відстані».
3. Перевірено одну SI-конфігурацію, ER-like connected internal graphs, `L=5` і `k=5`; узагальнення на SIR чи інші топології потребує нових даних.
4. `relW1` та `ΔRMST` описують різні аспекти distributional error; жодна з них окремо не характеризує повний розподіл прибуття.
5. Bootstrap CI стосуються paired contrasts error між політиками, а не окремих terminal `ΔRMST` у таблиці.
6. Експеримент не оцінює runtime і не підтримує тверджень про прискорення.

## Практичні рекомендації

1. У наступному дизайні зробити disjoint gateway sets обов'язковими в усіх режимах або окремо факторизувати overlap і distance.
2. Додатково match/record centrality, distance quantiles і час першого зараження export gateway, щоб пов'язати архітектуру з механізмом crossing delay.
3. Для far режиму збільшити горизонт або використати survival/RMST analysis із наперед зафіксованим спільним горизонтом; не аналізувати лише completed arrivals.
4. Калібрувати bridge-aware state на layerwise error, а не лише на термінальному шарі, і перевіряти можливу зміну знаку bias.

## Відтворюваність

Runner: [run_gateway_distance_sweep.py](../../extrascripts/runners/run_gateway_distance_sweep.py). Зведені дані: [summary.csv](../../output/missing_experiments/gateway_distance/full/summary.csv), [contrasts.csv](../../output/missing_experiments/gateway_distance/full/contrasts.csv) і [реплікаційні записи](../../output/missing_experiments/gateway_distance/full/replications/). Параметри, config fingerprint, seed policy, pairing і статус усіх реплікацій наведено в [маніфесті](../../output/missing_experiments/gateway_distance/full/manifest.json).
