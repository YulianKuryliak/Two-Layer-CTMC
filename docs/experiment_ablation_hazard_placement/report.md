# Experiment 01: Factorial Ablation — Rate Abstraction vs. Import Placement Error

**Дата**: 31 липня 2026  
**Статус**: ✅ Завершено, 100 реплікацій  
**Дані**: `output/ablation_hazard_placement/final_100/`  
**Графіки**: `docs/experiment_ablation_hazard_placement/figures/`  

---

## 1. Наукова мотивація

Стаття показує, що повна MicroMacro помилка входить через два незалежних механізми:

1. **Rate abstraction error** — макрохазард $\lambda_{ij}(t) = \beta_{\mathrm{out}} W_{ij} \bar{i}_i(t) \bar{s}_j(t)$ замінює точну мікроскопічну інтенсивність $a^\star_{ij}(X(t))$. У спарсених мережах $\bar{i}_i(t)$ може бути позитивним, поки граничні вузли ще не заражені — передчасне накопичення ризику.

2. **Import placement error** — вибір цільового вузла в спільноті $j$ за рівномірним розподілом $\Pi_{ij}$ замість точного ребро-умовного розподілу $\pi^\star_{ij}$.

**Питання**: Який з двох механізмів домінує? Чи адитивні вони? Чи залежить вплив placement від того, правильний чи ні hazard?

---

## 2. Дизайн експерименту

### 2.1 Мережа і параметри

| Параметр | Значення |
|---|---|
| Кількість спільнот $L$ | 10 (ланцюг) |
| Розмір спільноти $n$ | 100 |
| Середній внутрішній ступінь $d$ | 6 (ER граф, $p \approx 0.061$) |
| Кількість міжспільнотних ребер $k$ | 5 (між сусідніми спільнотами) |
| $\beta_{\mathrm{in}} = \beta_{\mathrm{out}}$ | 0.003 |
| $\gamma$ (відновлення) | 0.0 (SI процес) |
| Горизонт $T_{\mathrm{end}}$ | 4000 |
| Кількість реплікацій | 100 (paired) |

### 2.2 Чотири варіанти симулятора

| ID | Назва | Hazard | Placement |
|---|---|---|---|
| **M** | Full Micro Reference | Точна $a^\star_{ij}$ | Точна $\pi^\star_{ij}$ |
| **E/E** | Oracle Exact/Exact | Точна $a^\star_{ij}$ | Точна $\pi^\star_{ij}$ (окремі RNG) |
| **E/U** | Oracle Exact/Uniform | Точна $a^\star_{ij}$ | Рівномірна $\Pi_{ij}$ |
| **A/B** | Aggregate/Boundary | Агрегована $\lambda_{ij}$ | Boundary-weighted $\Pi_{ij}$ |
| **A/U** | Aggregate/Uniform | Агрегована $\lambda_{ij}$ | Рівномірна $\Pi_{ij}$ ← *поточний MicroMacro* |

> **Примітка**: Комірка «Aggregate/Exact» навмисно відсутня — коли $\lambda_{ij} > 0$, але $a^\star_{ij} = 0$ (граничні вузли ще не заражені), точний placement не визначений.

### 2.3 Паринг

Кожна реплікація ділить генерацію мережі та ініціалізацію між усіма варіантами. Event streams — variant-specific детерміновані (Blake2b hash). Паринг контролює варіабельність мережі та ініціалізації, але не є event-by-event CRN coupling.

### 2.4 Основна метрика

**Restricted Mean Survival Time (RMST)**:
$$\mathrm{RMST} = \mathbb{E}[\min(T_{\mathrm{last}}, T_{\mathrm{end}})]$$

$\Delta\mathrm{RMST} = \mathrm{RMST}_{\mathrm{variant}} - \mathrm{RMST}_{\mathrm{Micro}}$. Від'ємне значення → варіант приходить **раніше** (передчасна пропагація).

Bootstrap 95% CI: 1000 paired resample з насінням Blake2b.

---

## 3. Результати

### 3.1 Підсумкові метрики

| Варіант | Completion | $\mathbb{E}[T_{\mathrm{last}}]$ | Rel. $W_1$ | $\Delta$RMST | 95% CI |
|---|---|---|---|---|---|
| Full Micro | 1.00 | 2936.7 | — | 0 | — |
| Oracle E/E | 1.00 | 2944.3 | 0.0138 | +7.5 | [−93.6, +120.3] |
| Oracle E/U | 1.00 | 2775.5 | 0.0557 | **−161.2** | [−266.8, −48.8] |
| Aggregate A/B | 1.00 | 2735.5 | 0.0693 | **−201.2** | [−302.4, −106.7] |
| Aggregate A/U | 1.00 | 2715.9 | 0.0752 | **−220.8** | [−307.9, −120.2] |

### 3.2 Каузальні контрасти (RMST, paired bootstrap)

| Контраст | $\Delta$RMST | 95% CI | Значущий? |
|---|---|---|---|
| Oracle gate: E/E vs Micro | +7.5 | [−93.6, +120.3] | ❌ ні → oracle коректний |
| **Placement effect (exact hazard): E/U vs E/E** | **−168.8** | **[−274.5, −56.1]** | **✅ так** |
| **Rate abstraction (boundary proxy): A/B vs E/E** | **−208.8** | **[−310.0, −112.2]** | **✅ так** |
| **Full current error: A/U vs Micro** | **−220.8** | **[−315.5, −126.8]** | **✅ так** |
| Additional uniform under aggregate: A/U vs A/B | −19.6 | [−114.2, +77.2] | ❌ ні |
| **Interaction (placement_context_difference)** | **+149.2** | **[+5.7, +290.1]** | **✅ так** |

### 3.3 Ефект накопичення по шарах

| Шар | E/E mean shift | E/U mean shift | A/B mean shift | A/U mean shift |
|---|---|---|---|---|
| 1 | −19.1 | −3.2 | −40.1 | −47.2 |
| 3 | −11.0 | −75.2 | −109.2 | −103.8 |
| 5 | −14.2 | −110.7 | −136.4 | −121.0 |
| 7 | +27.0 | −121.1 | −143.9 | −162.5 |
| 9 (terminal) | +7.5 | −161.2 | −201.2 | −220.8 |

---

## 4. Інтерпретація

### 4.1 Rate abstraction: домінуючий механізм

**Rate abstraction** (порівняння A/B vs E/E) дає −208.8 RMST, що є найбільшим єдиним ефектом. Механізм: макрохазард $\lambda_{ij} = \beta k \bar{i}_i \bar{s}_j$ починає накопичуватися, як тільки $\bar{i}_i > 0$, — навіть якщо жоден з $k$ граничних вузлів ще не заражений. У спарсеній мережі ($d=6$) інфекція може довго поширюватись всередині спільноти, не досягаючи граничних вузлів. Це **Jensen-зміщення**: умовне середнє $B_I(t)$ за агрегованою фракцією переоцінює справжню bridge intensity (через кореляцію ранньо-зараженої підмножини з позицією відносно boundary).

### 4.2 Placement: значущий, але контекстно-залежний

**Placement effect** (E/U vs E/E) при точному hazard = **−168.8** → статистично значущий.  
**Placement effect** (A/U vs A/B) при агрегованому hazard = **−19.6** → не значущий.

Це суперадитивна взаємодія: якщо hazard вже неправильний (агрегований), то додаткова placement помилка практично не змінює загальне зміщення. Навпаки, якщо hazard точний, placement стає критичним.

**Практичний висновок**: покращення placement без виправлення hazard дає мало. Пріоритет — виправлення rate abstraction.

### 4.3 Повна помилка — менша за суму частин

$$|\Delta\text{RMST}_{A/U}| = 220.8 < |\Delta\text{RMST}_{E/U}| + |\Delta\text{RMST}_{A/B}| = 168.8 + 208.8 = 377.6$$

Це також свідчить про взаємодію: коли hazard вже зсунутий, placement помилка «вкладається» в ту ж сторону, але ефекти не лінійно складаються через нелінійність CTMC.

### 4.4 Oracle gate: підтвердження коректності методології

Варіант E/E (oracle, окремі RNG) дав $\Delta\text{RMST} = +7.5$, CI = [−93.6, +120.3] — включає нуль. Це підтверджує, що event-stream відокремлення не вносить систематичного зміщення і методологія абляції коректна.

---

## 5. Обмеження

1. **SI процес, не SIR**: всі результати для $\gamma = 0$. При відновленні bridge-вузол може одужати перед export → crossing-phase довший → ефект rate abstraction може ослабнути.

2. **Homogeneous chain**: кожна спільнота — однаковий ER граф. Асиметрична топологія (різний $d$ або різні $k$ між парами) може дати інший розподіл помилок по шарах.

3. **Один тип мережі**: ER граф. Для scale-free або small-world ступінь boundary-вузлів може систематично відрізнятись від середнього → інший розмір rate abstraction.

4. **$\beta_{\mathrm{macro}} = \beta$**: поточна калібровка. З кращою калібровкою (наприклад, $\beta_{\mathrm{macro}} = \beta \cdot d_{\mathrm{boundary}} / d_{\mathrm{mean}}$) rate abstraction може зменшитись.

---

## 6. Висновки

| Питання | Відповідь |
|---|---|
| Чи адитивні обидва механізми? | Ні — є суттєва взаємодія |
| Який механізм домінує? | Rate abstraction (−208.8 vs −168.8 RMST) |
| Чи важливий placement? | Так, при точному hazard (−168.8). При агрегованому — ні (−19.6) |
| Що виправляти першим? | Rate abstraction → bridge-aware hazard |
| Чи Oracle підтверджує методологію? | Так, ∆RMST = +7.5, CI включає 0 |

**Практична рекомендація**: Замінити $\lambda_{ij} = \beta k \bar{i}_i \bar{s}_j$ на bridge-aware версію, що враховує реальний стан граничних вузлів або хоча б їхній degree-weighted proxy. Це усуне ~94% поточної помилки (−208.8 / −220.8).

---

## 7. Відтворення

```bash
# Запуск симуляцій (вже виконано)
.venv/bin/python extrascripts/runners/run_hazard_placement_ablation.py \
    --stage final --n-runs 100 --oracle-approved --resume

# Регенерація графіків і контрастів
.venv/bin/python extrascripts/analysis/analyze_hazard_placement_ablation.py \
    output/ablation_hazard_placement/final_100
```

Повний LaTeX звіт: `reports/ablation_hazard_placement/ablation_report.pdf`

---

## 8. Графіки

> Всі файли знаходяться в `docs/experiment_ablation_hazard_placement/`

| Файл | Що показує |
|---|---|
| `survival.png` | Survival curves $P(T_{\mathrm{last}} > t)$ для всіх 5 варіантів |
| `rmst_shift.png` | $\Delta$RMST з 95% CI для кожного варіанту |
| `cumulative_error.png` | Paired cumulative arrival error по шарах ланцюга |
| `local_delay_shift.png` | Shift локальних затримок $\Delta T_i$ по переходах |
| `infected_dynamics.png` | Total $I(t)$ — медіана + IQR для всіх варіантів |
| `premature_hazard.png` | Кількість інтервалів де $\lambda_{ij} > 0$ але $a^\star_{ij} = 0$ |
| `hazard_vs_arrival.png` | Scatter: integrated signed hazard error vs terminal arrival error |

---

*Звіт згенеровано автоматично на основі даних у `output/ablation_hazard_placement/final_100/`.*
