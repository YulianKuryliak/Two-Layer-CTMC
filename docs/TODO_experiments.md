# TODO: Experiments for Two-Scale CTMC Validation

**Статус**: живий документ, оновлюється після кожного виконаного кроку.

---

## EXPERIMENT-01 — Factorial Ablation: Rate vs. Placement Error Decomposition
**Статус: ✅ ВИКОНАНО** | Дані: `output/ablation_hazard_placement/final_100/` | Звіт: `docs/experiment_ablation_hazard_placement/`

Розкласти повну помилку MicroMacro на два незалежних компоненти:
- **Rate abstraction error**: агрегований $\lambda_{ij}$ vs точна $a^\star_{ij}$
- **Import placement error**: рівномірний $\Pi_{ij}$ vs точний $\pi^\star_{ij}$

Варіанти: Full Micro | Oracle (exact/exact) | Exact hazard + uniform placement | Aggregate + boundary placement | Aggregate + uniform placement.

100 paired replications, $L=10$, $n=100$, $d=6$, $k=5$, $\beta=0.003$, SI.

**Результат**: Обидва механізми статистично значущі. Rate abstraction домінує (−208.8 RMST, 95% CI [−310.0, −112.2]). Placement значимий при точному hazard (−168.8), але нікчемний при агрегованому (−19.6). Сильний ефект взаємодії [149.2, CI=(5.7, 290.1)].

---

## EXPERIMENT-02 — Gateway Distance Sweep
**Статус: 📋 ЗАПЛАНОВАНО**

**Ціль**: показати, що внутрішня відстань між import і export граничними вузлами ($D_{\min}$, $D_{\mathrm{mean}}$) є предиктором помилки.

**Дизайн**:
- Зафіксувати: ER-граф, $n=100$, $|E|=\mathrm{const}$, $k=5$, один ланцюг $L=5$
- Варіювати: конфігурація граничних вузлів — **near** / **random** / **far**
- Метрики: $W_1(T_{\mathrm{last}})$, $\Delta$RMST, $\mathbb{E}[T_{\mathrm{find}}]$
- Очікування: $D_{\min}\uparrow \Rightarrow \Delta\text{RMST} \downarrow$ (більш негативний)

**Параметри**: $L=5$, $n=100$, $d=6$, $k=5$, $\beta=0.003$, SI, 100 replications, $T_{\mathrm{end}}=2000$.

---

## EXPERIMENT-03 — Beta_out / Beta_in Ratio Sweep
**Статус: � ВИКОНУЄТЬСЯ** | Скрипт: `extrascripts/runners/run_ce03_ce04.py --experiment ce03`

**Ціль**: визначити, як відношення $r = \beta_{\mathrm{out}} / \beta_{\mathrm{in}}$ впливає на величину помилки.

**Дизайн**:
- Зафіксувати: sparse chain, $n=100$, $d=6$, $k=5$, $L=10$
- Варіювати: $r \in \{0.1, 0.3, 1.0, 2.0, 5.0\}$, $\beta_{\mathrm{in}} = 0.003$
- Очікування: помилка зростає з $r$, мінімум біля $r=1$

**Результати**: *(з'являться після завершення)*

---

## EXPERIMENT-04 — SIR vs SI Validation
**Статус: ✅ ВИКОНАНО** | Дані: `output/claude_experiments/ce04_sir_validation/` | Звіт: `docs/ce04_sir_validation/`

**Ціль**: перевірити, чи змінюється знак і величина зміщення при введенні відновлення $\gamma > 0$.

**Ключові результати**:

| Сценарій | γ | R₀ | ΔRMST | Завершення Micro | Завершення MM |
|---|---|---|---|---|---|
| SI | 0 | ∞ | −1068 | 100% | 100% |
| SIR слабке | 0.0005 | 6 | −1244 | 89% | 96% |
| SIR середнє | 0.001 | 3 | −1587 | 70% | 93% |
| SIR сильне | 0.003 | 1 | −1726 | **28%** | **72%** |

**Критичний висновок**: При R₀=1 MicroMacro генерує 44 в.п. хибно-позитивних завершень. Похибка зростає з γ (+62% від SI до SIR-сильне).

---

## EXPERIMENT-05 — Bridge-Aware Hazard Correction
**Статус: 📋 ЗАПЛАНОВАНО** (після Exp-01 та Exp-02)

**Ціль**: показати «стелю» виправлення — oracle bridge hazard vs degree-weighted proxy.

**Дизайн**:
- Full Micro | MM-uniform (baseline) | MM-oracle-bridge (знаємо $B_I(t)$ точно) | MM-degree-weighted
- Метрика: наскільки oracle-bridge закриває gap до Full Micro?

---
