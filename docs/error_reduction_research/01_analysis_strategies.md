# 01 — Аналіз стратегій зменшення MicroMacro approximation error (multi-topology)

**Роль:** Analyst (read-only, без коду і без симуляцій).
**Мета:** широкий, від першопринципів, огляд усіх правдоподібних способів зменшити
похибку MicroMacro відносно full Gillespie CTMC (ground truth), з жорсткою вимогою
**parameters-only feasibility** та **multi-topology robustness** (minimax по топологіях).

---

## 0. Постановка, обмеження і словник дескрипторів

### 0.1 Що саме апроксимує macro-рівень

Macro import hazard (див. `two_layer_ctmc/engines/macro_engine.py`):

$$
\lambda_{i\to j}(t) = \beta_{\text{macro}}\, T\, W_{ij}\,
\big(\underbrace{I_i/N_i}_{\text{infectivity}_i}\big)^{\alpha_i}\,
\big(\underbrace{S_j/N_j}_{\text{susceptibility}_j}\big).
$$

На macro-подію ціль обирається **uniformly at random** серед susceptibles громади `j`
(`import_infection()` в micro-моделі).

### 0.2 Три джерела похибки (що ми долаємо)

1. **Rate abstraction error (домінантне).** `I_i/N_i` припускає, що кожен infected
   миттєво доступний для експорту. Насправді зараження має спершу дійти до
   **boundary/gateway nodes**; sparse/погано перемішані громади це затримують →
   MicroMacro експортує **зарано** → premature arrival → negative `dRMST`.
2. **Import placement error.** Uniform placement проти реального boundary-локалізованого
   входу. Значуще переважно під exact/near-exact hazard.
3. **Synchronization / numerical error.** Macro tau-step проти неперервного micro-часу.

### 0.3 HARD CONSTRAINT — що дозволено бачити macro-рівню

Тільки **параметри мережі**, ніколи не точні node identities / positions / infected-set:

| Дескриптор (English) | Що це | Джерело в коді |
|---|---|---|
| `W` | inter-community weight matrix | `network.py` |
| `N_i` | community sizes | `network.py`, `graph["community_sizes"]` |
| `mean_degree` (`d`) | середній внутрішній степінь | `compute_descriptors` |
| `k2_over_k` = ⟨k²⟩/⟨k⟩ | degree heterogeneity (треба додати; зараз немає) | — |
| `lambda2_norm` = `mu2` | Fiedler value (2-ге найменше власне значення `L_norm`) | `compute_descriptors` |
| `spectral_radius` | adjacency spectral radius `lambda1` | `compute_descriptors` |
| `tau_find` | log(n)/(beta_in·(lambda1−1)) — internal find-time | `compute_descriptors` |
| `boundary_count_mean` | к-сть boundary nodes на громаду (з `W`, count only) | `compute_descriptors` |
| `boundary_within_deg_mean` | середній within-degree boundary nodes | `compute_descriptors` |

**Похідні parameters-only величини** (легально обчислити з таблиці вище):
`boundary_fraction` = `boundary_count_mean / N`, `mixing_time` ≈ `1/mu2`,
`b_deg_ratio` = `boundary_within_deg_mean / mean_degree`.

### 0.4 Критерії оцінки кандидата

- **Metric:** первинно `dRMST_norm` (horizon-commensurate, порівнюваний між
  топологіями) та `rel_W1`; вторинно `delta_completion`, KM/CDF shape.
- **gap_closure** = (|off| − |corrected|) / |off|, per-topology.
- **Winner = best worst-case gap_closure (minimax)** по всій topology-matrix, не best
  average. Кандидат, що покращує одну топологію, але **погіршує** іншу, — відхиляється
  або має бути gated parameters-only дескриптором із **self-disable до identity** на
  well-mixed/dense.

### 0.5 Топологічна матриця (обовʼязкова для валідації)

- `macro_graph_type` ∈ {`chain`, `complete`, `star`}
- `micro_graph_type` ∈ {`complete`, `random_connected`, `clique_leaves`} + `chain` (hard sparse stress)
- варіювати `d` (sparse..dense) та `r = beta_out/beta_in`.

Примітка щодо `star`: у поточному builder-і leaf-громади **single-node** → будь-яка
корекція, що залежить від micro-sparsity, автоматично `→ identity` на leaves (N=1 ⇒
`mu2` undefined ⇒ factor 1). Це «безкоштовний» self-disable, але означає, що `star`
переважно тестує hub-громаду + placement, не rate abstraction.

---

## 1. Каталог кандидатів

Для кожного: (1) name; (2) mechanism + цільове джерело похибки; (3) parameters-only
feasibility + потрібні дескриптори; (4) generality по топологіях + прапорці ризику;
(5) implementation sketch (hook у коді); (6) validation plan; (7) risks / failure modes.

---

### C1. Spectral damping exponent (baseline, вже в коді) — `alpha_i = clamp(1/mu2, 1, 6)`

- **Mechanism:** підносить `infectivity_i` до степеня `alpha_i ≥ 1`; оскільки `I/N < 1`,
  це **damps** ранній експорт у sparse громадах. Ціль: **rate abstraction**.
- **Parameters-only:** так. Потребує `mu2` (`lambda2_norm`).
- **Generality:** self-disable коректний (`mu2 → 1` dense ⇒ `alpha → 1`). Але **overshoot**
  на extreme sparsity (chain micro, `alpha` впирається у cap=6): smoke показав
  `dRMST_norm −0.41 → +0.18` — перекорекція у бік late arrival. Nonlinear exponent
  небезпечно взаємодіє з `I/N` (при малому `I/N` навіть помірний `alpha` майже гасить hazard).
- **Hook:** `MacroEngine._compute_hazards_matrix`, `alpha_mode="spectral"` (існує).
- **Validation:** уся topology-matrix; ключова клітина — `chain` micro (overshoot test).
- **Risks:** overshoot при cap; чутливість до вибору cap; exponent-форма не bounded по
  ефекту в термінах затримки часу. **Це лише один механізм — треба ширше.**

---

### C2. Boundary-availability effective-rate multiplier — `phi_i(t)` (MFPT / mixing-time)

- **Mechanism:** замінити exponent-на-infectivity **bounded multiplicative** фактором
  наявності gateway-у: `lambda_{i→j} → phi_i · lambda_{i→j}`, де
  `phi_i ∈ (0,1]` — оцінка частки infected, що вже досягла boundary. Мотивація —
  **mean first-passage time (MFPT)** зараження до boundary масштабується як
  `~ mixing_time ≈ 1/mu2`, а стаціонарна доступність boundary масштабується як
  `boundary_fraction`. Природна форма:
  `phi_i = min(1, boundary_fraction_i · g(mu2, I_i/N_i))`,
  напр. `phi_i = 1 − exp(−(I_i/N_i) · mu2 · N_i / c)` (soft ramp: мало infected або
  малий `mu2` ⇒ малий `phi`; багато infected або великий `mu2` ⇒ `phi → 1`).
  Ціль: **rate abstraction** (домінантне) + частково **synchronization**.
- **Parameters-only:** так. Дескриптори: `mu2`, `boundary_count_mean`, `N_i`; опційно `d`.
- **Generality (сильна сторона):** фактор **bounded в (0,1]** ⇒ **не може overshoot** у
  late-arrival бік (на відміну від exponent). Self-disable: `mu2 → 1` **або**
  `boundary_fraction → 1` ⇒ `phi → 1` ⇒ identity. Очікується найкращий minimax:
  на dense (complete micro, complete macro) — identity; на chain/clique_leaves —
  сильне, але **насичене** damping без перекорекції.
- **Hook:** `MacroEngine._compute_hazards_matrix` — новий `alpha_mode="availability"`;
  помножити `self._hazards_buffer` на `phi` (вектор по `i`, broadcast по рядках). Потрібно
  прокинути `boundary_count`/`N` у `MacroEngine` (зараз він має лише `N`; boundary count
  обчислюється в runner-і — додати як аргумент, аналог `alphas`).
- **Validation:** `dRMST_norm`, `rel_W1`, per-topology gap_closure; спец. увага —
  chain micro (чи прибирає overshoot C1) та complete micro (чи справді identity).
- **Risks:** форма `g(·)` потребує калібрування однієї-двох констант — робити це
  **structurally** (з `tau_find`/`mixing_time`), не тюнити під chain-of-cliques.
  Ризик undershoot (замало damping) якщо `phi` насичується надто швидко.

---

### C3. Reach-time onset delay — non-Markovian latency gate `tau_reach_i`

- **Mechanism:** громада не може експортувати, поки зараження ймовірно не досягло
  boundary. Ввести **deterministic onset offset** `tau_reach_i`: macro hazard від `i`
  = 0, поки `t < t_seed_i + tau_reach_i`, потім вмикається. `tau_reach_i` — parameters-only
  оцінка MFPT до boundary: `tau_reach_i ≈ tau_find_i · f(boundary_fraction_i)` або
  `≈ log(N_i)/(beta_in·(lambda1−1)) + (1/mu2)·const`. Це не rate-scaling, а **зсув часу
  першого можливого експорту** — прямо б'є premature-arrival механізм. Ціль:
  **rate abstraction** + **synchronization**.
- **Parameters-only:** так. Дескриптори: `tau_find` (⇒ `lambda1`, `beta_in`, `N`), `mu2`,
  `boundary_fraction`.
- **Generality:** природно self-disable: dense ⇒ `tau_find` малий, `tau_reach → 0` ⇒
  no gate. Не має overshoot у rate (це затримка, не гасіння) — але **може overshoot у
  часі**, якщо `tau_reach` завищено (експорт занадто пізно). Bounded зверху часткою
  горизонту (напр. `tau_reach ≤ κ·T_end/L`) обмежує ризик.
- **Hook:** orchestrator (`orchestrators/micromacro.py`) — тримати per-community
  `export_enabled_time[i]`; у `total_hazard_given`/`update_hazards` занулювати рядки `i`
  з `t < export_enabled_time[i]`. Потрібен `t_seed_i` (час першого infection у `i`),
  який orchestrator знає без node identities (лише count `I_i>0`).
- **Validation:** KM/CDF shape (зсув лівого хвоста), `dRMST_norm`, `rel_W1`; chain micro
  (велика затримка) vs complete (нульова).
- **Risks:** дискретність tau-step може огрубити gate; over-delay при завищеному
  `tau_reach`; взаємодія з C2 (не застосовувати обидва на повну — див. §2).

---

### C4. Structural `beta_macro` calibration floor — глобальний / per-community rescale

- **Mechanism:** помножити `beta_macro` на structural factor `s_i ≤ 1`, що відображає
  «ефективну провідність» boundary: `s_i = h(boundary_fraction_i, mu2_i)`. На відміну від
  C2 (стан-залежний `phi_i(t)`), тут **статичний** масштаб ставки. Ціль: **rate abstraction**.
- **Parameters-only:** так. Дескриптори: `boundary_count_mean`, `N_i`, `mu2`.
- **Generality:** простий, але грубий: статичний множник не ловить часову динаміку
  доступності boundary (рано мало infected — треба сильніший damping, пізно — слабший).
  Self-disable ок (`s → 1` dense). Ризик: недо/перекорекція на різних `r`, бо не залежить
  від `I/N`. Слабший minimax, ніж C2/C3, але хороший **fallback / floor**.
- **Hook:** тривіально — per-`i` множник у `MacroEngine` (масштабувати рядки `W` або
  `beta_macro` вектором).
- **Validation:** швидкий smoke; використати як baseline-floor для порівняння з C2.
- **Risks:** одновимірність; не адаптивний у часі; легко перетюнити під один `r`.

---

### C5. Parameters-only degree-biased placement — bias import target by degree distribution

- **Mechanism:** замість uniform placement обирати ціль з імовірністю ∝ (функція
  степеня вузла). Node identities **не** використовуються, але **degree of the candidate
  target** — це локальна властивість, яку treba трактувати обережно: дозволено лише якщо
  ми біасуємо за **degree-класом** із відомого degree distribution, не за тим, «який саме
  вузол на межі». Практично: sample degree-class d* ∝ p(d*)·d* (size-biased), потім вузол
  того класу — але вибір конкретного вузла все ще uniform у межах класу. Ціль:
  **import placement error**.
- **Parameters-only:** **межовий випадок.** Degree distribution — parameters-only;
  але «посадити у високостепеневий/boundary вузол» наближається до node-level знання.
  Безпечний варіант: біас лише за **degree class** (агрегована статистика), не за
  boundary-membership конкретного вузла. Потрібен `degree distribution` (додати до
  дескрипторів; зараз є лише `mean_degree`).
- **Generality:** ефект другого порядку; значущий лише під near-exact hazard (CE-01
  показав: placement значущий тільки під exact hazard). На well-mixed — нульовий (усі
  степені рівні). Ризик: на heterogeneous (clique_leaves) може допомогти, на regular
  (complete/chain) — no-op. Малий downside, малий upside.
- **Hook:** `MicroModel.import_infection()` — приймати degree-bias розподіл.
- **Validation:** переважно `rel_W1`/CDF на clique_leaves та random_connected під
  сильним hazard; очікувати no-op на complete.
- **Risks:** межа parameters-only (легко скотитися у node-level); малий очікуваний виграш
  ⇒ низький пріоритет попри легальність.

---

### C6. Learned parameters-only correction — регресія `phi = model(descriptors)` з CE-06

- **Mechanism:** з precomputed датасету CE-06 (`output/claude_experiments/.../summary.csv`,
  descriptors → `dRMST_norm`/`rel_W1`) навчити **parameters-only** функцію, що видає
  корекційний множник/exponent (напр. цільовий `phi_i` або `alpha_i`) як функцію
  (`mu2`, `boundary_fraction`, `d`, `r`, `k2_over_k`). Це «мета»-кандидат: обгортка над
  C2/C1, де форму `g(·)` не задають вручну, а фітують. Ціль: **rate abstraction** (та ін.,
  залежно від target).
- **Parameters-only:** так — вхід лише дескриптори. Ключова умова: **train на кількох
  топологіях**, інакше вивчить chain-of-cliques bias.
- **Generality:** потенційно найкращий average, але **ризик minimax**: extrapolation за
  межі train-топологій непередбачувана. Треба **monotonic / shape constraints** (напр.
  `phi` зростає з `mu2` і з `boundary_fraction`; `phi → 1` на краю dense) + self-disable
  clamp, щоб гарантувати identity на well-mixed навіть поза train-розподілом.
- **Hook:** pre-pass, що з дескрипторів рахує per-community множник і передає в
  `MacroEngine` (як C2/C4). Модель — проста (isotonic / GLM / shallow), не black-box.
- **Validation:** cross-topology hold-out (train на 2 micro-типах, test на 3-му +
  chain-stress); перевірити identity на complete.
- **Risks:** overfit до train-топологій; extrapolation; прихована залежність від `r`
  (censoring). Використовувати **тільки** з hard structural clamps.

---

### C7. Quasi-stationary boundary-availability (time-scale separation)

- **Mechanism:** явна two-timescale декомпозиція: внутрішня динаміка швидка → на macro-
  масштабі громада у quasi-stationary distribution (QSD) з **очікуваною часткою infected
  на boundary** `q_i`. Замінити `I_i/N_i` на `q_i · I_i/N_i` або на «reachable infectivity».
  `q_i` оцінюється з QSD SI-хвилі на графі з відомим `mu2`/degree — parameters-only.
  Ціль: **rate abstraction**.
- **Parameters-only:** так (`mu2`, degree stats, `boundary_fraction`).
- **Generality:** теоретично найпринциповіший; але time-scale separation **ламається**,
  коли `r` великий (macro не повільніший за micro) — саме там premature arrival найгірший.
  Тобто ризик: працює де похибка мала, слабшає де велика. Self-disable ок.
- **Hook:** аналог C2 (стан-залежний множник), але з QSD-derived `q_i`.
- **Validation:** перевірити валідність припущення по `r` (low-r ок, high-r під питанням).
- **Risks:** припущення розділення масштабів невірне у high-`r`; складніше калібрувати,
  ніж C2, з подібним ефектом ⇒ C2 як прагматична апроксимація C7.

---

### C8. Delay-distributed (gamma) non-Markovian macro hazard

- **Mechanism:** замість миттєвого hazard-у ввести **розподілений lag**: час перетину =
  internal-reach + crossing, де internal-reach ~ Gamma(shape, scale) з параметрами з
  `tau_find`/`mu2`. Реалізується як convolution ядра доступності або як stochastic
  `tau_reach_i ~ Gamma(...)` (стохастична версія C3). Ціль: **rate abstraction** +
  **shape** (краще відтворює хвіст CDF, не лише середнє).
- **Parameters-only:** так (`tau_find`, `mu2` → shape/scale).
- **Generality:** покращує **розподіл** (`rel_W1`, KM shape), не лише `dRMST`. Self-disable:
  scale → 0 dense. Ризик: дисперсія lag може розмити ефект; складніша реалізація у
  midpoint-Gillespie циклі.
- **Hook:** orchestrator — семплити per-export lag; або stochastic onset (розширення C3).
- **Validation:** `rel_W1` та CDF/KM — основна ціль; chain micro.
- **Risks:** імплементаційна складність у tau-step midpoint схемі; interplay з macro
  integral threshold (`thresh_int`).

---

### C9. Degree-based / annealed mean-field internal-speed correction (⟨k²⟩/⟨k⟩)

- **Mechanism:** швидкість внутрішнього поширення (а отже й досягнення boundary) для
  heterogeneous graph масштабується не `mean_degree`, а `⟨k²⟩/⟨k⟩` (annealed network
  mean-field). Використати `k2_over_k` для калібрування `tau_reach`/`phi` (C2/C3/C8)
  замість плоского `d`. Ціль: **rate abstraction** (уточнює *інші* кандидати).
- **Parameters-only:** так, але треба **додати `k2_over_k`** до `compute_descriptors`
  (зараз лише `mean_degree`). Тривіально з degree sequence (parameters-only).
- **Generality:** важливо для random_connected / clique_leaves (heterogeneous); no-op для
  regular (complete/chain, де ⟨k²⟩/⟨k⟩ = k). Це **множник-уточнювач**, не самостійна корекція.
- **Hook:** дескриптор-рівень (`compute_descriptors`), потім у формулу C2/C3.
- **Validation:** порівняти C2-with-`d` vs C2-with-`k2_over_k` на heterogeneous топологіях.
- **Risks:** сам по собі нічого не виправляє; цінність лише в парі з C2/C3/C8.

---

### C10. Pair approximation / moment closure для cross-community coupling

- **Mechanism:** трекати не лише `I_i`, а й **очікувану к-сть [S–I] boundary-пар** між
  громадами (перший момент кореляції). Macro hazard ∝ очікуваній boundary S–I концентрації,
  не `I_i/N_i·S_j/N_j`. Ціль: **rate abstraction** + **placement** (природно вбудовує
  boundary-локалізацію).
- **Parameters-only:** **частково.** Pair-closure потребує оцінки boundary S–I пар; її
  можна замкнути через `boundary_count`, degree stats і поточні counts `I_i, S_j`
  (parameters-only), **не** через реальні пари ⇒ це approximate closure, а не точний облік.
- **Generality:** принципово сильний на sparse/heterogeneous; на dense зводиться до
  mean-field (self-disable). Але closure error сам може бути топологічно-залежним.
- **Hook:** глибша зміна `MacroEngine` + orchestrator (треба evolve додаткову moment-змінну).
  Найвища імплементаційна вартість у списку.
- **Validation:** уся матриця; порівняти з C2 (простіший) — чи виправдовує складність.
- **Risks:** висока складність; closure може бути крихким; ROI під питанням проти C2.

---

### C11. Edge-based compartmental / DMP (message passing) для coupling

- **Mechanism:** edge-based compartmental model / dynamic message passing для оцінки
  ймовірності перетину по кожному inter-community каналу з урахуванням внутрішньої
  топології. Ціль: **rate abstraction** + **placement**.
- **Parameters-only:** **проблемно.** Повний DMP по конкретних ребрах = node/edge-level
  знання (порушує constraint). Легальна лише **агрегована/annealed** версія, що фактично
  вироджується у C2/C9. Тобто повний DMP — oracle-only (діагностика верхньої межі), не
  deploy.
- **Generality:** високий потенціал як **upper-bound oracle**, не як deployed рішення.
- **Hook:** окремий oracle-runner (діагностика), не у production hazard.
- **Validation:** використати як стелю (скільки взагалі можна закрити) для C2/C3.
- **Risks:** порушує HARD CONSTRAINT у повній формі; лише діагностична цінність.

---

### C12. Effective renormalized macro rate з mixing-time / spectral-gap (analytic phi)

- **Mechanism:** аналітичний окремий випадок C2/C7: `phi_i = mu2 / (mu2 + beta_out/beta_in)`
  або `phi_i = 1/(1 + c/(mu2·N_i))` — renormalization ставки за співвідношенням
  внутрішнього mixing-rate (`~ mu2·beta_in`) до export-rate. Ціль: **rate abstraction**.
- **Parameters-only:** так (`mu2`, `N_i`, `r`).
- **Generality:** захоплює критичну залежність від `r` (чому CE-03 має sign-flip): при
  малому `r` mixing встигає (`phi → 1`), при великому `r` — ні (`phi < 1`, сильніший
  damping). Self-disable по `mu2` **і** по `r`. Хороший, компактний, одно-формульний.
- **Hook:** як C2 (множник у `MacroEngine`).
- **Validation:** особливо навколо CE-03 sign-change (`r ≈ 0.5–0.8`).
- **Risks:** одна константа `c` на калібрування; форма може недооцінювати extreme sparse.

---

## 2. Ranked shortlist (top-6), оптимізовано під parameters-only + minimax generality

Пріоритет: (a) bounded ефект (не overshoot), (b) чіткий self-disable до identity на
dense/well-mixed, (c) прямий удар по домінантному rate-abstraction error, (d) низька
імплементаційна вартість / малий ризик регресу на «легких» топологіях.

| Rank | Candidate | Error source | Key descriptor(s) | Self-disable → identity коли | Тип hook | Ризик overshoot |
|---|---|---|---|---|---|---|
| **1** | **C2 boundary-availability multiplier `phi_i(t)`** | rate abstraction | `mu2`, `boundary_fraction`, `N_i` | `mu2 → 1` **або** `boundary_fraction → 1` | macro hazard multiplier | **низький (bounded (0,1])** |
| **2** | **C3 reach-time onset delay `tau_reach_i`** | rate + sync | `tau_find`, `mu2`, `boundary_fraction` | `tau_find → 0` (dense) | orchestrator onset gate | середній (bounded `κ·T_end/L`) |
| **3** | **C12 renormalized rate `phi = mu2/(mu2+r)`** | rate abstraction | `mu2`, `N_i`, `r` | `mu2 ≫ r` | macro hazard multiplier | низький |
| **4** | **C9 `k2_over_k` refinement** (уточнювач для 1–3) | rate abstraction | `k2_over_k` (додати) | regular graph (⟨k²⟩/⟨k⟩=k) | descriptor → формула | — (no-op сам по собі) |
| **5** | **C4 structural `beta_macro` floor** (fallback) | rate abstraction | `boundary_fraction`, `mu2` | dense ⇒ `s→1` | static rate multiplier | середній |
| **6** | **C6 learned parameters-only `phi`** (meta) | rate abstraction | descriptors + `r` | hard clamp `phi=1` на краю | pre-pass | середній (extrapolation) |

**Відхилено / oracle-only / низький пріоритет:** C1 (baseline exponent — залишити як
порівняння, але overshoot-схильний), C5 (placement — легальний, але межовий і low-ROI),
C7 (принциповий, але C2 — прагматична апроксимація), C8 (цінний для shape/`rel_W1`, але
складніший — друга хвиля), C10 (moment closure — висока вартість, сумнівний ROI vs C2),
C11 (DMP — порушує constraint у повній формі, лише oracle-стеля).

### 2.1 Рекомендований execution order для Executor

1. **C2** — реалізувати bounded multiplier `phi_i(t)` (новий `alpha_mode="availability"`),
   прокинути `boundary_count` у `MacroEngine`. Валідувати всю topology-matrix; ключові
   клітини: `complete` micro (має бути identity, `gap_closure ≈ 0`, без регресу) і
   `chain`/`clique_leaves` micro (максимальне закриття без overshoot). **Гейт далі:** якщо
   на complete micro `dRMST_norm` погіршується — форму `phi` перекалібрувати (self-disable
   зламано).
2. **C12** — як альтернативна аналітична форма `phi` (одна формула з явним `r`); порівняти
   minimax gap_closure C2 vs C12 навколо CE-03 sign-change (`r ∈ {0.5, 0.8, 2.0}`).
   Обрати переможця як «rate-multiplier engine».
3. **C9** — додати `k2_over_k` у `compute_descriptors`; підставити у переможця (1/2) замість
   плоского `d`; ретест лише heterogeneous топологій (random_connected, clique_leaves).
4. **C3** — onset delay як **комплементарний** до rate-multiplier (не сумувати ефекти
   наосліп: спершу multiplier-only, потім delay-only, потім обидва з ослабленими
   коефіцієнтами). Ціль — покращити KM/CDF лівий хвіст і `rel_W1`, де multiplier зачепив
   лише середнє.
5. **C4** — тримати як простий structural floor/fallback і як lower-bound baseline.
6. **C6** — коли є мульти-топологічний CE-06 датасет: фітнути `phi` з monotonic + clamp
   обмеженнями; deploy лише якщо cross-topology hold-out б'є ручні форми (1–3) за minimax.

**Правило неперекорекції:** застосовувати рівно **один** rate-механізм на повну (C2 **або**
C12), опційно доповнений C3 для shape; ніколи C2+C12+C4 одночасно на повну — це гарантована
перекорекція.

---

## 3. Top recommendation

**C2 — boundary-availability effective-rate multiplier `phi_i(t)`** (bounded множник у
`MacroEngine`, керований `mu2` + `boundary_fraction`, зі стохастично-незалежним
self-disable `phi → 1` на dense/well-mixed).

**Одно-рядкове обґрунтування:** він прямо б'є домінантний rate-abstraction error, потребує
лише parameters-only дескрипторів (`mu2`, `boundary_count`, `N`), і — на відміну від
exponent-корекції C1 — **обмежений у (0,1]**, тож не може overshoot і **автоматично
вироджується в identity** на щільних/добре перемішаних топологіях, що дає найкращу
worst-case (minimax) генералізацію по всій topology-matrix.
