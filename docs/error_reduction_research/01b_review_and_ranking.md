# 01b — Review та ranking кандидатів корекції MicroMacro approximation error

**Роль:** Reviewer (read-only). Ranking candidate corrections під критеріями
BRIEF.md: **parameters-only**, **minimax generality** по topology-matrix,
**overshoot-safe** (заборона late-arrival bias), **ROI**.
**Вхід:** `BRIEF.md`, `01_analysis_strategies.md`, реально імплементований код у
`two_layer_ctmc/engines/macro_engine.py` (`correction_mode` = `boundary_phi`=C2,
`renorm_phi`=C12; `alpha_mode="spectral"`=C1).

---

## 0. Ключове переформулювання критерію overshoot-safety

BRIEF стверджує: «bounded multiplicative factors `phi in (0,1]` (C2/C12) cannot
overshoot rate». Це треба **уточнити**, бо саме тут ховається головна помилка.

- Boundedness `phi ∈ (0,1]` гарантує лише, що скоригована ставка **не перевищує**
  baseline. Вона **НЕ** гарантує overshoot-safety у сенсі *timing bias*.
- Похибка вимірюється як `dRMST_norm` (timing). Якщо на певній топології baseline
  вже точний (`dRMST_norm ≈ 0`, dense/well-mixed) або громада вже насичена
  (`I_i/N_i → 1`), будь-який множник `phi < 1` **сповільнює export** → downstream
  arrival зсувається праворуч → **late-arrival overshoot** (позитивний `dRMST`).
- Отже справжній гарант overshoot-safety — **не** boundedness, а **коректний
  self-disable до identity `phi → 1`** там, де baseline точний: (i) на dense/
  well-mixed micro, і (ii) при `I_i/N_i → 1` (насичення).

**Наслідок для аудиту:** множник, що не повертається до 1 на dense **або** на
saturation, класифікується як overshoot-safe = **NO**, навіть якщо `phi ∈ (0,1]`.

---

## 1. Аудит-таблиця C1..C12

| Cand | Target error source | Parameters-only (descriptors) | Overshoot-safe | Expected minimax generality | Impl cost | Key risk |
|---|---|---|---|---|---|---|
| **C1** spectral exponent `alpha=clamp(1/mu2,1,6)` | rate abstraction | YES (`mu2`) | **NO** (exponent, cap→overshoot) | Low | Low (coded) | Nonlinear power гасить hazard при малому `I/N`; overshoot на chain micro |
| **C2** `boundary_phi` (coded) | rate abstraction | YES (`mu2`,`boundary_count`,`N`) | **NO** (див. §3.1) | Low–Med | Low (coded) | Self-disable зламано на dense (keys off `boundary_fraction`); permanent undershoot при `I/N→1` |
| **C3** onset delay `tau_reach_i` | rate + sync | YES (`tau_find`,`mu2`,`boundary_fraction`) | **YES** (capped; no permanent damping) | **High** | Med (orchestrator) | Over-delay якщо `tau_reach` завищено; tau-step granularity |
| **C4** static `beta_macro` floor | rate abstraction | YES (`boundary_fraction`,`mu2`) | **NO** (static, no `I/N` → permanent damping) | Low–Med | Low | Не time-adaptive; late overshoot при saturation |
| **C5** degree-biased placement | import placement | PARTIAL (degree-class OK; boundary-membership NO) | YES (placement, не rate) | Low (лише під exact hazard) | Low–Med | Межа parameters-only; low ROI |
| **C6** learned `phi=model(descr)` | rate abstraction | YES (descriptors) *якщо train multi-topo* | PARTIAL (потрібен hard clamp) | Med (extrapolation risk) | Med–High | Overfit до train-топологій; непередбачувана extrapolation |
| **C7** QSD availability `q_i` | rate abstraction | YES (`mu2`,degree,`boundary_fraction`) | YES-ish (state-dep, bounded) | Med | Med | Timescale separation ламається при high `r` (де похибка найгірша) |
| **C8** gamma-delay non-Markovian | rate + shape | YES (`tau_find`,`mu2`) | **YES** (scale→0 self-disable) | Med–High (найкраще для `rel_W1`/shape) | High (tau-step) | Складність у midpoint-Gillespie; interplay з `thresh_int` |
| **C9** `k2_over_k` refinement | rate (refiner) | YES (треба **додати** `k2_over_k`) | N/A (no-op сам по собі) | Refiner (High у парі) | Low | Нічого не виправляє соло; цінність лише з C2/C3/C8 |
| **C10** pair approx / moment closure | rate + placement | PARTIAL (approximate closure) | Unclear (closure error топо-залежний) | Med | High | Найвища вартість; крихкий closure; сумнівний ROI vs C3/C12 |
| **C11** edge-based / DMP | rate + placement | **NO** (повна форма = node/edge-level) | N/A | Oracle-only | High | Порушує HARD CONSTRAINT; лише upper-bound діагностика |
| **C12** `renorm_phi = mu2/(mu2+r)` (coded) | rate abstraction | YES (`mu2`,`N`,`r`) | **PARTIAL** (self-disable OK лише low-`r`; over-damps dense high-`r`; no ramp) | Med | Low (coded) | Dense high-`r` overshoot; state-independent (без `I/N`) |

---

## 2. Ranked shortlist (top-3 to implement/validate first)

Критерій переможця — **best worst-case gap_closure (minimax)** при жорсткій
overshoot-safety та parameters-only. Порядок нижче — за **очікуваним minimax
merit**, а не за ROI (ROI враховано окремо в execution order, §3.4).

### Rank 1 — C3 reach-time onset delay `tau_reach_i`
**Обґрунтування (minimax winner):**
- **Єдиний rate-механізм з коректним self-disable в обох критичних точках:**
  на dense `tau_find → 0` ⇒ gate зникає ⇒ identity; після відкриття gate громада
  експортує на **повній mean-field ставці** ⇒ **немає permanent undershoot** при
  насиченні (на відміну від C2/C4/C12). Це прямо усуває late-arrival overshoot.
- Б'є домінантний premature-arrival механізм найпряміше: це **зсув часу першого
  можливого export**, а не гасіння ставки; отже не спотворює довгостроковий rate.
- Overshoot можливий лише «у часі» при завищеному `tau_reach`; обмежується жорстко
  `tau_reach ≤ κ·T_end/L`.
- Parameters-only чисто: `tau_find` (⇒ `lambda1`,`beta_in`,`N`), `mu2`,
  `boundary_fraction`; `t_seed_i` — це **count** (`I_i>0`), не node identity.

### Rank 2 — C12 `renorm_phi` (вже в коді) **з обов'язковим dense self-disable fix**
**Обґрунтування (найкращий ROI + правильна `r`-залежність):**
- Вже імплементовано (`correction_mode="renorm_phi"`) ⇒ zero implementation cost
  для першого діагностичного проходу.
- Єдина формула, що **явно** несе залежність від `r` — саме те, що пояснює CE-03
  sign-flip; на low-`r` (де baseline пізній/точний) коректно `phi → 1`.
- **АЛЕ deploy лише після фіксу:** при dense micro `mu2 ≈ 1` (не `≫ r`), тож при
  `r=2` `phi = 1/3` — сильне over-damping точного baseline ⇒ гарантований
  late-overshoot на `complete` micro (див. §3.2). Потрібен gate, що self-disable-ить
  на dense band `mu2` (напр. нормувати на dense-eigenvalue `N/(N-1)` або
  gate-ити через `boundary_fraction`). До фіксу C12 — **діагностика, не рішення**.

### Rank 3 — C9 `k2_over_k` refinement
**Обґрунтування (cheap prerequisite для генералізації 1 та 2):**
- Соло нічого не виправляє, але **необхідний**, щоб C3/C12 генералізували на
  heterogeneous micro (`random_connected`, `clique_leaves`), де швидкість
  внутрішнього поширення масштабується `⟨k²⟩/⟨k⟩`, а не плоским `d`. Без нього
  `tau_reach`/`phi` систематично помиляться саме на heterogeneous клітинах —
  найімовірніших «ламачах» minimax.
- Тривіальна вартість: додати `k2_over_k` у `compute_descriptors` (parameters-only
  з degree sequence); no-op на regular графах (`complete`/`chain`).

### Явна відповідь на питання завдання
- **Чи C2/C12 — правильні перші піки?** — **Частково.**
  - **C12** — так *для ROI/діагностики* (вже в коді, несе `r`-залежність), але
    **не як deploy-рішення без dense-gate фіксу**.
  - **C2 (`boundary_phi`) — НІ у поточній імплементації.** Її форма
    `phi = min(1, boundary_frac·(1 + mu2·I/N))` (i) keys off `boundary_fraction`,
    тому **не** self-disable-иться на dense, і (ii) насичується на рівні
    `≤ 2·boundary_fraction < 1`, тобто **ніколи не повертається до 1** навіть при
    `I/N → 1` ⇒ permanent undershoot ⇒ late overshoot. Треба **реформувати**
    перед довірою; наразі — нижче C3/C12.
- **Чи піднімати C3 / C9 / C7 / C8?** — Так: **C3 підняти на #1** (єдиний
  структурно overshoot-safe rate-механізм), **C9 — обов'язковий refiner (#3)**.
  C7 — принципова версія C3/C2, але ламається на high-`r`; C8 — цінний **у другій
  хвилі** для `rel_W1`/shape (стохастична версія C3), не для першого мінімума
  `dRMST`.
- **Combine?** Так, але за правилом неперекорекції: **рівно один rate-механізм на
  повну** (C3 **або** C12), і **ніколи** C2+C12+C4 разом.

---

## 3. Recommended execution order + критичний тест на кожен top-3

### 3.1 Чому C2 (`boundary_phi`) провалює overshoot-safety (доказ з коду)
`macro_engine.py`, `correction_mode="boundary_phi"`:
```
boundary_frac = boundary_counts / max(N,1)
phi = min(1, boundary_frac * (1 + mu2 * (I/N)))
```
- `mu2 ∈ (0, ~1]`, `I/N ∈ [0,1]` ⇒ множник `(1 + mu2·I/N) ∈ [1,2]` ⇒
  `phi ≤ min(1, 2·boundary_frac)`.
- Якщо `boundary_fraction < 0.5` (типово для sparse/chain-of-cliques), то навіть
  при **повному насиченні** `I/N=1` маємо `phi ≤ 2·boundary_frac < 1` ⇒ громада
  **вічно** експортує на зниженій ставці ⇒ downstream late bias.
- На dense `complete` micro `mu2 ≈ 1`, але `boundary_fraction` все одно **малий**
  (лише вузли з inter-links) ⇒ `phi` малий ⇒ **damping на точному baseline** ⇒
  late overshoot. **Self-disable по `mu2` зламано, бо форма keys off
  `boundary_fraction`.**

### 3.2 Чому C12 потребує dense-gate (доказ з коду)
```
r = beta_out / beta_in
phi = mu2 / (mu2 + r)          # state-INDEPENDENT
```
- Немає `I/N` ⇒ **немає ramp**: однакове damping на старті й при насиченні.
- Dense `complete`: `mu2 ≈ 1`, `r=2` ⇒ `phi ≈ 0.33` на точному baseline ⇒
  late overshoot. Self-disable вимагає `mu2 ≫ r`, чого на dense band `mu2≈1`
  не буває при `r ≥ 1`.

### 3.3 Critical validation cell на кожен top-3
| Cand | Клітина, що найімовірніше зламає | Що перевіряти |
|---|---|---|
| **C3** | `complete` micro (dense) — має бути identity | `dRMST_norm(off) ≈ dRMST_norm(C3)`, `gap_closure ≈ 0`, жодного late overshoot (`tau_reach→0`). Друга клітина: `chain` micro high-`r` — чи прибирає C1-overshoot без over-delay |
| **C12** | `complete` micro при `r ∈ {1,2}` — dense high-`r` overshoot | Чи `dRMST_norm` **не** стає позитивним (late). Якщо стає — dense-gate обов'язковий |
| **C9** | `clique_leaves` / `random_connected` (heterogeneous) | C(base)-with-`d` vs C-with-`k2_over_k`: `gap_closure` має зрости саме тут; no-op на `complete`/`chain` |

### 3.4 Execution order (з урахуванням ROI/залежностей)
1. **Fast diagnostic pass C12 (як є)** — запустити `renorm_phi` на topology-matrix,
   особливо `complete` micro × `r∈{0.5,0.8,2.0}` (CE-03 sign-change). Мета —
   **емпірично зафіксувати dense high-`r` overshoot** і скалібрувати dense-gate.
   (Нуль коду, максимум сигналу.)
2. **C9** — додати `k2_over_k` у `compute_descriptors` (дешево, розблоковує 1&3).
3. **C12-fixed** — додати dense self-disable gate (нормування `mu2` на `N/(N-1)`
   або `phi := 1 - (1-mu2/(mu2+r))·bounded(boundary_fraction, mixing)`); підставити
   `k2_over_k`; ретест повної матриці на minimax.
4. **C3** — реалізувати onset gate в orchestrator (`export_enabled_time[i]`,
   `tau_reach_i` capped `≤ κ·T_end/L`, калібр через `tau_find`/`k2_over_k`).
   Порівняти minimax C3 vs C12-fixed; обрати **один** rate-engine.
5. (Друга хвиля) **C8** для `rel_W1`/KM-shape; **C6** тільки з multi-topo CE-06
   датасетом і hard monotonic+clamp; **C4** як lower-bound floor baseline.
6. **Reformulate C2** перед будь-якою довірою: `phi` має → 1 при `I/N→1` і
   self-disable по `mu2` (не по `boundary_fraction`), інакше не використовувати.

**Правило неперекорекції:** один rate-механізм на повну (C3 **або** C12-fixed),
опційно + C3-gate для shape; **ніколи** C2+C12+C4 одночасно.

---

## 4. Parameters-only violations та математичні concerns

1. **C11 (DMP), повна форма — VIOLATION.** Message passing по конкретних ребрах =
   node/edge-level знання. Легальна лише annealed/aggregated версія, що вироджується
   в C2/C9. Тримати **oracle-only** (upper-bound), як і зазначив Analyst.
2. **C5 (degree-biased placement) — borderline.** «Посадити у boundary-вузол» ≈
   node-level. Легально лише bias за **degree-class** з відомого degree distribution;
   вибір вузла в межах класу лишається uniform. Тримати цю межу жорстко; low ROI.
3. **C10 (moment closure) — PARTIAL.** Boundary S–I пари не можна рахувати точно;
   лише approximate closure через `boundary_count`+degree+counts. Це approximation,
   не облік — closure error сам топо-залежний. OK як parameters-only, але ROI слабкий.
4. **Розбіжність специфікації vs імплементації C2.** У `01_analysis` C2 запропоновано
   `phi = 1 − exp(−(I/N)·mu2·N/c)` (коректно → 1 при `I/N→1`). У коді ж
   `min(1, boundary_frac·(1+mu2·I/N))` — **інша форма**, що (a) насичується нижче 1
   і (b) self-disable-иться по `boundary_fraction`, а не по `mu2`. Це не lint-violation
   parameters-only, але **математично не відповідає заявленому self-disable** і робить
   імплементований C2 overshoot-unsafe. Реформувати ближче до exp-форми (з поверненням
   до 1) перед deploy.
5. **Dimensional heuristic у C12.** `mu2` — безрозмірний spectral gap normalized
   Laplacian (`mu2 ∈ (0, N/(N-1)]`), а `r=beta_out/beta_in` — відношення ставок.
   Форма `mu2/(mu2+r)` складає різнорідні величини без масштабної константи; Analyst-ова
   альтернатива `1/(1+c/(mu2·N))` має явні `c`,`N` і краще обґрунтована. Поточний код
   **опускає** `c` та `N`-масштаб ⇒ dense-band `mu2≈1` не домінує над `r` ⇒ саме звідси
   dense-overshoot. Фікс dense-gate = і фікс цього concern.
6. **C1 exponent — нелінійність.** `alpha` на `I/N<1`: при малому `I/N` навіть помірний
   `alpha` майже занулює hazard (не bounded у термінах *часу*), і cap=6 дає overshoot.
   Тримати лише як **порівняльний baseline**, не deploy.
7. **C3 `t_seed_i` — parameters-only OK.** Час першого infection у громаді визначається
   з count (`I_i>0`), без node identity. Але дискретність tau-step може огрубити gate —
   валідувати на KM лівому хвості.

---

## 5. Вердикт

- **Deploy engine (minimax merit):** **C3 onset delay** — єдиний rate-механізм з
  коректним self-disable на dense **і** при насиченні ⇒ найкраща очікувана worst-case
  overshoot-safety.
- **Fast-ROI engine (з фіксом):** **C12 `renorm_phi`** — вже в коді, несе `r`-залежність;
  deploy тільки після dense self-disable gate.
- **Обов'язковий refiner:** **C9 `k2_over_k`** — щоб 1&2 генералізували на heterogeneous.
- **Не використовувати як є:** **C2 `boundary_phi`** (реформувати: return-to-1 +
  self-disable по `mu2`), **C1** (overshoot; лише baseline).
- **Oracle-only / low-ROI:** C11 (violation), C5 (borderline), C10 (cost), C7 (high-`r`),
  C4 (floor), C8/C6 (друга хвиля).
