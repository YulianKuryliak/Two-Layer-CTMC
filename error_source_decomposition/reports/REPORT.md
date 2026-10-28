# Experiment 07 — Цілісна декомпозиція джерел похибки Two-Layer CTMC

Дата аналізу: 2026-08-14.

## 1. Мета

Мета цього пакета — не просто показати, що MicroMacro відрізняється від Full Micro, а розділити:

1. **похибку програмної реалізації Production**;
2. **структурну похибку двомасштабної апроксимації**;
3. **похибку глобальної калібрації масштабу macro-rate**;
4. механізми всередині структурної похибки: стан межі джерела, стан межі отримувача, точні пари міжспільнотних ребер та розміщення імпорту.

Важливе правило: відсотки з різних рівнів декомпозиції **не додаються між собою**. Наприклад, калібрація може поглинути той самий середній bias, який механістично породжують boundary/placement effects.

## 2. Головний результат — Production implementation

100 парних мереж, L=10, n=100, d≈6, k=5, beta_in=beta_out=0.003, SI, T_end=4000.

| Варіант | RMST |
|---|---:|
| Full Micro | 2936.73 |
| Event-driven A/U | 2715.94 |
| Current Production (tau=1) | 1677.55 |
| Production + independent RNG streams | 2729.43 |

Звідси загальна поправка від current Production до Full Micro дорівнює **1259.18 RMST**.

- **Implementation discrepancy:** 1038.39 RMST = **82.47%** загального signed Production bias (95% CI для частки: 74.92–89.80%).
- **Event-driven structural gap:** 220.79 RMST = **17.53%** (95% CI: 10.20–25.08%).

### 2.1. Shared RNG coupling

Заміна одного глобального Python RNG на незалежні RNG streams для micro-communities дала **+1051.88 RMST** (95% CI 890.92–1211.34) відносно current Production.

Після цього різниця з event-driven A/U становить лише **+13.49 RMST**, 95% CI **[-120.63; 94.91]** — статистично сумісна з нулем.

У signed decomposition це еквівалентно **101.30%** net implementation gap, а залишкові implementation semantics — **−1.30%**. Значення >100% допустиме: воно означає невеликий переліт та компенсуючий залишковий ефект, а не “більше ніж вся фізична помилка”.

**Висновок:** current shared global RNG architecture практично повністю пояснює великий Production-vs-event-driven gap у цьому baseline.

### 2.2. Unit-test RNG

50 000 двокомпонентних повторів показали:

| RNG | corr(I1,I2) | P(target infected), community 1 | community 2 |
|---|---:|---:|---:|
| shared global | 0.531 | 0.394 | 0.155 |
| independent | -0.002 | 0.394 | 0.390 |

Тобто shared RNG + horizon-safe rollback не лише корелює формально незалежні communities, а й змінює маргінальний розподіл другої community. Це вже correctness issue, а не просто “інша seed policy”.

### 2.3. Cache-preserve окремо

Збереження cached event при shared RNG **саме по собі не дало стабільного виправлення**:

- L=5, n=50: +4.04 RMST, 95% CI [-28.37; 37.77];
- L=10, n=20: −99.73 RMST, 95% CI [-323.54; 119.98].

Отже проблема — не “скидання cache” саме по собі, а coupling між shared RNG, rollback/resampling та іншими random draws.

### 2.4. Повторні імпорти як медіатор

Single-entry не є новим первинним джерелом; це втручання, яке вимикає повторні macro-imports.

- L=5, n=50: RNG correction у normal режимі +262.56 RMST, а interaction з re-imports = **+326.04**, 95% CI [209.74; 446.42].
- L=10, n=20: RNG correction +1119.41 RMST; interaction з re-imports = **+1143.54**, 95% CI [721.26; 1525.78].

Коли повторні імпорти заборонені, ефект RNG майже зникає/змінює знак. Отже дані підтримують ланцюг:

**shared RNG coupling → excess repeated macro-imports → прискорене downstream propagation.**

Водночас старий event-driven re-import ablation показує, що re-imports **не пояснюють structural aggregate-vs-exact hazard gap**: зміна gap = −19.51 RMST, 95% CI [-151.20; 105.00].

### 2.5. Чи це просто tau / midpoint integration?

Зменшення tau від 1 до 0.25 змінює terminal RMST лише на **31.15** одиниць, тобто близько **3.0%** величини implementation gap; 95% CI цього contrast включає нуль у попередньому audit. Це не підтримує пояснення, що ~1000 RMST виникають просто через завеликий synchronization step.

## 3. Структурна похибка при номінальній калібрації

Для чистої структурної декомпозиції використано event-driven Experiment 06 на 100 тих самих мережах. Деномінатор — directional gap **E/E − A/U = 228.33 RMST**. E/E відрізняється від Full Micro лише на +7.54 RMST, CI містить нуль.

Метод декомпозиції:

- source × destination boundary marginals утворюють повний 2×2 factorial; їх interaction ділиться методом Шеплі;
- exact edge-pair information додається після SD;
- placement оцінюється як E/U → E/E.

Симетричний `A_E` навмисно **не використовується**: exact edge-conditioned placement не визначене, якщо aggregate hazard спрацював при нульовому exact active-edge hazard. Штучна домовленість про null-event змішала б два механізми.

| Компонент | RMST-внесок | Частка directional structural gap | 95% CI RMST |
|---|---:|---:|---:|
| source_boundary | 102.32 | 44.81% | [35.63; 170.41] |
| destination_boundary | -39.38 | -17.25% | [-99.73; 20.70] |
| edge_pair_joint | -3.38 | -1.48% | [-56.04; 48.76] |
| placement | 168.77 | 73.92% | [60.26; 276.74] |


Ключова інтерпретація:

- **source boundary** має позитивний і статистично підтриманий внесок;
- **placement** має найбільший позитивний внесок;
- destination-only та exact edge-pair refinement у baseline не мають переконливого окремого ефекту;
- від'ємні signed contributions означають компенсацію інших механізмів.

## 4. Калібрація: окремий рівень, який не можна додавати до таблиці вище

Старий CE-07 не використано для цього висновку, бо він запускав Production MicroMacro і був змішаний із shared-RNG implementation error.

Новий тест виконано на **event-driven A/U**: 100 мереж, candidate c={0.7,0.8,0.85,0.9,1.0,1.1}, 5-fold cross-validation. c обирається тільки на train fold та застосовується до held-out fold.

Результат:

- nominal A/U vs Full: **−220.79 RMST**, 95% CI приблизно [-318.29; -124.59];
- held-out calibrated: **+4.41 RMST**, 95% CI [-106.82; 112.27];
- calibration correction: **+225.20 RMST**, 95% CI [152.40; 300.84];
- c*=0.85 у 4/5 folds, c*=0.80 у 1/5.
- restricted W1: **7.52% → 1.00%** від RMST Full Micro.

Це означає: **у одному фіксованому baseline більшість signed RMST bias можна поглинути scalar calibration**. Тому механістичні source/placement percentages не є унікальним “пирогом причин” — вони описують nominally calibrated model.

## 5. Чому калібрація не усуває structural information loss

Gateway validation: macro-visible A/U prediction однакова для near/random/far, але E/E reference має terminal RMST:

- near = 881.87;
- random = 1261.42;
- far = 1411.67.

Різниця між крайніми reference = **529.79 RMST**.

Навіть якщо дозволити calibration вибрати **будь-яке одне спільне RMST**, а не реалістичний scalar c, мінімально можливий minimax residual становить щонайменше **264.90 RMST**. Тому жодна architecture-blind scalar calibration принципово не може одночасно відтворити near і far.

Це і є сильніший доказ **структурної втрати інформації**: однакові macro observables не ідентифікують точний first-passage process.

## 6. Як правильно говорити про “відсотки похибки” у статті

### Можна складати до 100% лише всередині одного визначеного рівня

**Total Production → Full Micro:**
- implementation discrepancy: 82.47%;
- event-driven structural gap: 17.53%.

**Implementation gap current Production → event-driven A/U:**
- shared-RNG stream correction: 101.30%;
- residual implementation correction: −1.30% (CI включає 0).

**Nominal structural gap A/U → E/E:**
- source-boundary: +44.81%;
- destination-boundary: −17.25%;
- exact edge-pair refinement: −1.48%;
- placement: +73.92%;
- разом = 100%.

### Не можна додавати між рівнями

Не можна написати: “82% RNG + 74% placement + 48% boundary + 102% calibration”. Це різні деномінатори й альтернативні втручання.

Калібрація, boundary correction та placement correction можуть закривати перекривні частини одного observed bias. Тому в статті їх треба подавати як **ієрархію**:

1. усунути implementation defect;
2. визначити/навчити calibration на training data;
3. на held-out topology/regime вимірювати residual structural error;
4. mechanistic ablations пояснюють цей residual та межі застосовності.

## 7. Рекомендована класифікація джерел похибки

1. **Похибка програмної реалізації: shared RNG stream coupling** — підтверджено як домінантний source current Production discrepancy.
2. **Похибка глобальної калібрації macro-rate** — значуща в одному режимі, але не універсальний structural fix.
3. **Structural information loss на межі community:**
   - стан межі джерела — підтверджено;
   - розміщення імпорту — підтверджено;
   - destination marginal / exact edge-pair correction — слабкі в baseline.
4. **Modifiers, а не окремі sources:** gateway geometry, export/mixing ratio, довжина chain.
5. **Mediators, а не первинні sources:** repeated imports для Production RNG defect.
6. **Evaluation artifacts:** censoring/finite horizon — контролювати RMST, не включати в модельний “пиріг”.
7. **Нерівність Єнсена** — математичний механізм, який пояснює bias від усереднення випадкової cumulative hazard; це не окрема четверта похибка.

## 8. Файли

- `MASTER_RESULTS.csv` — одна зведена таблиця.
- `structural_factorial/` — 100-run hierarchical attribution та calibration CV.
- `production_factorial/L10_independent_n100/` — 100-run Production RNG correction.
- `production_factorial/L5_n50/` і `L10_n20/` — re-import mediation / cache diagnostics.
- `production_factorial/rng_unit/` — 50k unit diagnostic.
- `runners/` — усі нові ранери та аналізатори.

## 9. Головний висновок

Після усунення shared-RNG coupling Production практично повертається до event-driven A/U. Після цього лишається вже **справжня апроксимаційна задача**: nominal bias можна підкрутити scalar calibration у фіксованому regime, але hidden gateway architecture доводить, що однієї калібрації недостатньо. Для переносимої точності потрібна boundary-aware інформація та коректне local placement.
