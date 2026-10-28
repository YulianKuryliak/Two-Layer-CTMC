# Звіт: джерела похибки двомасштабного симулятора

Аналіз джерел approximation error MicroMacro відносно повного Micro (Gillespie
CTMC). Документ (a) критикує наявну в статті декомпозицію на три джерела і (b)
додає джерела, які я вважаю значущими, але які в статті відсутні або злиті в одну
категорію та недооцінені.

Позначення як у статті: $a^\star_{ij}(x)$ — точна edge-level інтенсивність,
$\lambda_{ij}=\beta_{\text{out}}W_{ij}\bar i_i\bar s_j$ — макро-апроксимація,
$\pi^\star_{ij}$ — edge-conditioned target-розподіл, $\Pi_{ij}$ — placement rule.

---

## 1. Наявна в статті декомпозиція (recap)

Стаття виділяє три джерела:

1. **Rate abstraction error** — $\varepsilon_{ij}=a^\star_{ij}-\lambda_{ij}$.
2. **Import placement error** — $\Pi_{ij}\neq\pi^\star_{ij}$.
3. **Synchronization / numerical error** — quadrature $|\hat H-H|\le\eta$.

Каркас корисний, але він (i) змішує різні за природою механізми в одному пункті,
(ii) припускає **адитивність** трьох джерел, якої насправді немає, і (iii)
повністю ігнорує принаймні чотири значущі джерела (розділ 3).

---

## 2. Критика наявних трьох джерел

### 2.1. «Rate abstraction» зливає три різні механізми
Пункт 1 статті насправді містить **три** незалежні за причиною і за ліками
похибки:

- **(A) Structural (irreducible) error** — навіть за ідеального калібрування
  функція від агрегатів $(\bar i_i,\bar s_j,W_{ij})$ не може відтворити
  $a^\star_{ij}$, коли export контролюється конкретними bridge-вузлами. Це те, що
  доводить Proposition (pigeonhole). **Непереборне** за фіксованого макро-стану.
- **(B) Calibration (level) error** — вибір $\beta_{\text{macro}}=\beta$,
  $W_{ij}=k$ робить масштаб $\lambda_{ij}$ довільним. Це **систематичний зсув
  рівня**, повністю **виправний** калібруванням проти $a^\star_{ij}$. Стаття сама
  це визнає в Outlook, але не відокремлює від (A).
- **(C) Moment-closure (covariance) error** — макро бере **добуток середніх**
  $\bar i_i\cdot\bar s_j$, тоді як точна очікувана інтенсивність містить
  коваріацію:
  $$\mathbb{E}[a^\star_{ij}] \;\propto\; \mathbb{E}[I_i^{\partial}S_j^{\partial}]
  \;=\; \mathbb{E}[I_i^{\partial}]\mathbb{E}[S_j^{\partial}] + \mathrm{Cov},$$
  де $I_i^{\partial},S_j^{\partial}$ — boundary-стани. Mean-field відкидає $\mathrm{Cov}$.
  Ця похибка присутня **навіть за широкого/повного інтерфейсу**, де structural
  error (A) зникає. Стаття її не називає взагалі.

**Наслідок:** плутанина (A)+(B)+(C) не дає діагностувати, скільки з виміряного
bias є непереборним (A), скільки — артефактом калібрування (B), скільки — closure
(C). Це критично, бо лікуються вони по-різному.

### 2.2. «Import placement»: iff-твердження тавтологічне, магнітуда не виміряна
Proposition про placement («пост-подієвий law збігається ⇔ розподіли збігаються»)
— тавтологія. Реальні питання — **скільки** та **куди поширюється** — не
розкриті:
- магнітуда $\|\Pi_{ij}-\pi^\star_{ij}\|$ ніде не оцінена;
- placement **не є термінальним**: uniform-вкидання змінює майбутню траєкторію
  $i_j(t)$ призначення, а отже його майбутню export-інтенсивність до **наступної**
  громади. Тобто placement error **перетікає** в rate error нижче по ланцюгу.

### 2.3. «Synchronization»: доведений bound тривіальний і не про те
$|\hat t-t^\star|\le\eta/\theta_{\min}$ — наслідок монотонності, а $\eta$ ніколи не
вимірюється. Справжня похибка синхронізації — **discretization bias** від
$\tau_{\text{micro}}=1$ (розділ 3.3), який має інший характер (систематичний, не
випадковий) і в статті не аналізується.

### 2.4. Хибне припущення адитивності
Стаття подає три джерела як окремі доданки. У ланцюгу вони **взаємодіють
нелінійно**: (rate → коли стається import) × (placement → який стан призначення)
× (synchronization → о котрій його зафіксовано) спільно визначають траєкторію
наступної громади. Тому спостережуваний $\Delta\mu$ — це **не** сума трьох
незалежних внесків, і зміна знаку bias уздовж ланцюга (§таблиці статті) —
найпряміший доказ саме цієї взаємодії.

---

## 3. Додаткові значущі джерела (відсутні або недооцінені в статті)

### 3.1. Susceptible-depletion / re-import error ⚑ (значуще)
Макро імпортує в $j$, доки $\bar s_j>0$, і uniform-placement може «витрачати»
представницький susceptible-пул інакше, ніж повна мережа, де конкретні
bridge→target ребра насичуються. Для **вже активованої** громади макро допускає
**повторні upstream-імпорти**, які пришвидшують її та все нижче за течією.
Closed-form first-passage у статті це **явно ігнорує** («no longer exact closed
forms»), але як **джерело похибки** воно не каталогізоване. Ймовірний внесок у
sign-flip між $\Delta T_0$ (+) та пізніми $\Delta T_i$ (−).

### 3.2. Moment-closure / covariance error ⚑ (значуще, див. 2.1-C)
Добуток $\bar i_i\bar s_j$ замість $\mathbb{E}[I_i S_j]$. Присутнє завжди, зникає
лише в детермінованій межі. Тест: повний інтерфейс ($k=N$), де structural error =0,
але залишковий bias ≠0 ⇒ це саме closure.

### 3.3. Sync-interval discretization bias ⚑ (значуще, не тестоване)
$\Theta(s)$ між точками синхронізації **застаріле** (piecewise-constant snapshot
на сітці $\tau_{\text{micro}}=1$). Це quadrature-bias порядку $O(\tau_{\text{micro}})$,
**систематичний**: коли $\Theta$ зростає — події запізнюються, коли спадає —
випереджають. $\tau_{\text{micro}}$ **ніколи не свіпається**, тож цей внесок
повністю **сконфаунджений** у кожне число bias. Мінімальний тест: повторити baseline
з $\tau_{\text{micro}}\in\{0.1,0.5,1\}$ і подивитись, чи рухається $\Delta\mu$.

### 3.4. Finite-horizon / censoring estimator bias ⚑ (значуще)
Не похибка симулятора, а **похибка вимірювання**: порівняння
conditional-on-completion середніх за різних completion (90/100 vs 64/100). Цензуровані
пізні прогони MicroMacro викидаються, тягнучи його умовне середнє вниз ⇒ штучний
негативний $\Delta\mu$. Найімовірніше пояснює частину «premature arrival». Лік:
Kaplan–Meier / вирівняна completion / явний censoring-aware естіматор.

### 3.5. Boundary-degree variance (а не лише $W_{ij}$)
Скаляр $W_{ij}$ фіксує **кількість** cross-ребер, але не **концентрацію**. Два
інтерфейси з однаковим $W$, але різною дисперсією boundary-degree дають різні
$a^\star$. Реальний драйвер structural error — **variance** розподілу
boundary-degree, а не $W$. Стаття згадує концентрацію якісно, але не як окремий
параметр.

### 3.6. Seeding / initial-condition mismatch
Аналітика припускає uniform seed у кліці. Якщо стартовий інфікований у Micro та
спосіб ініціалізації в MicroMacro дають різні розподіли позиції першого
export-релевантного вузла, це окреме джерело ранньої розбіжності, незалежне від
трьох названих.

---

## 4. Переглянута таксономія (пропозиція)

| # | Джерело | Природа | Переборне? | Тестоване в статті |
|---|---|---|---|---|
| A | Structural aggregation (bridge) | irreducible за агрегатів | ні (лише більше boundary-info) | так (якісно) |
| B | Calibration / level ($\beta_{\text{macro}},W$) | систематичний зсув | так (калібрування) | ні |
| C | Moment-closure (covariance) | закриття моментів | частково | ні |
| D | Import placement + downstream feedback | coupling | так (bridge-aware $\Pi$) | частково |
| E | Susceptible-depletion / re-import | booking межі | частково | ні |
| F | Sync discretization ($\tau_{\text{micro}}$) | quadrature bias | так (менший крок) | ні (bound тривіальний) |
| G | Censoring / finite-horizon | **estimator** bias | так (KM) | визнано, не виправлено |
| H | Boundary-degree variance / seeding | параметризація | так | ні |

**Ключове перегрупування:** статтєві «3 джерела» = {A+B+C у пункті 1} ∪ {D у
пункті 2} ∪ {F у пункті 3}. Пропущені повністю: **C, E, G, H**. Серед них
**C, E, G** я вважаю кандидатами на **домінантні** пояснення спостережуваних
sign-flips і «premature arrival», бо вони не потребують structural bridge-ефекту,
який мінімальний бенчмарк статті показав як **слабкий** ($n=100,k=2$).

---

## 5. Пріоритети для перевірки

1. **G (censoring)** — переміряти всі $\Delta\mu/\Delta q$ censoring-aware; це може
   прибрати або перевернути «premature arrival» без жодної зміни симулятора.
2. **B (calibration)** — відкалібрувати $\beta_{\text{macro}}$ проти $a^\star$;
   відокремити level-error від structural.
3. **F ($\tau_{\text{micro}}$-свіп)** — показати, чи bias рухається з кроком сітки.
4. **C (closure)** — тест на повному інтерфейсі ($k=N$): будь-який залишковий bias =
   covariance-error.
5. **E (re-import)** — увімкнути/вимкнути повторні upstream-імпорти й виміряти
   внесок у downstream-прискорення.

**Підсумок:** декомпозиція статті концептуально правильна, але **занадто груба**:
пункт «rate abstraction» ховає три різні механізми (A/B/C), пункт «synchronization»
доводить не ту похибку (F), а чотири значущі джерела (C, E, G, H) відсутні. Перш
ніж інтерпретувати будь-який виміряний $\Delta\mu$ як structural bias, треба зняти
**G** і **B** — інакше кількісні висновки не ізолюють заявлений bridge-механізм.

---

## 6. Фікси по джерелах

Нижче — конкретні виправлення. Позначки: **[eval]** — фікс у методі вимірювання
(симулятор не чіпаємо); **[impl]** — зміна коду симулятора; **[design]** — зміна
самої макро-моделі.

### A — Structural aggregation (bridge) → **bridge-aware macro state** [design]
Чистими агрегатами $(\bar i_i,\bar s_j)$ це непереборне. Фікс — додати **мінімальну
boundary-інформацію** до макро-стану: тримати лічильник інфікованих boundary-вузлів
$B_{I,i}(t)$ (одне число на громаду) і рахувати
$$\lambda_{ij}=\beta_{\text{out}}\,W_{ij}\,\frac{B_{I,i}(t)}{|\partial_i|}\,\bar s_j(t)$$
замість $\bar i_i$. Вартість $O(1)$ на громаду, зберігає обчислювальну перевагу,
прибирає premature leakage (макро не накопичує hazard, доки $B_{I,i}=0$). Це саме
той «hybrid bridge-aware layer», який Outlook статті обіцяє, але не реалізує.
**Залишок:** covariance (C) всередині boundary-набору.

### B — Calibration / level → **fit $W_{ij}$ проти $a^\star$** [impl, одноразово]
Прогнати Micro-reference один раз, записати кумулятивні інтеграли й покласти
$$W_{ij}=\frac{\int_0^{T}\mathbb{E}[a^\star_{ij}(s)]\,ds}
{\beta_{\text{out}}\int_0^{T}\bar i_i(s)\bar s_j(s)\,ds},$$
тобто узгодити time-integrated hazard (а не миттєвий). Альтернатива — moment-match
$\mathbb{E}[T_A]$. Одноразово на топологію. Після цього level-error = 0 і будь-який
залишковий bias — це вже A/C, а не масштаб.

### C — Moment-closure (covariance) → **conditional/pair замість добутку** [design]
Замінити добуток середніх на умовну величину:
$$\lambda_{ij}\propto \mathbb{E}[I_i^{\partial}S_j^{\partial}]
=\bar i_i^{\partial}\bar s_j^{\partial}+\widehat{\mathrm{Cov}},$$
де $\widehat{\mathrm{Cov}}$ оцінюється з boundary-лічильників (дешевий pair-closure),
або абсорбувати середню коваріацію в ефективний $W_{ij}$ (частковий, безкоштовний
фікс — фактично збігається з B). Повний фікс потребує трекінгу boundary-joint.
**Найдешевше:** якщо застосовано A (boundary-lічильник), більша частина C зникає
автоматично.

### D — Placement + downstream feedback → **degree-weighted placement** [impl]
Замінити uniform $\Pi_{ij}=1/s_j$ на наближення $\pi^\star_{ij}$: садити import у
susceptible boundary-вузол з імовірністю $\propto$ boundary-degree:
$$\Pi_{ij}(v)\propto \deg_{\partial}(v)\,\mathbf 1\{x_v=S\}.$$
Виправляє і immediate post-event law, і downstream feedback (правильний стан
призначення → правильна його майбутня export-інтенсивність). Вартість — доступ до
boundary-degree, який уже є в графі.

### E — Susceptible-depletion / re-import → **boundary-capacity booking** [impl]
(i) використовувати **залишкову** кількість susceptible boundary-вузлів замість
$\bar s_j$; (ii) для вже активованої громади вводити saturation-фактор або
позначати «використані» boundary-ребра, щоб макро не сідало повторно швидше, ніж
дозволяє межа. Прибирає штучне downstream-прискорення (ймовірний драйвер
sign-flip $\Delta T_0{+}\!\to\!\Delta T_i{-}$).

### F — Sync discretization → **event-driven hazard, без сітки** [impl] ⭐
Найчистіший фікс і він **безкоштовний за точністю**: $\Theta(t)$ між micro-подіями
**кусково-стала** (стаття сама це стверджує). Отже накопичувати $H$ треба **точно**
на кожній micro-події, а не на сітці $\tau_{\text{micro}}$:
$$H \mathrel{+}= \Theta(t_k)\,(t_{k+1}-t_k)\quad\text{на кожному micro-інтервалі},$$
і розв'язувати $H=E$ аналітично всередині інтервалу. Тоді quadrature-bias $=0$,
$\tau_{\text{micro}}$ як параметр зникає. Якщо event-driven важко — тимчасово
зменшити $\tau_{\text{micro}}$ (збіжність $O(\tau)$) і показати плато.

### G — Censoring / finite-horizon → **RMST на спільному горизонті** [eval] ⭐
Не порівнювати conditional-on-completion середні. Замість цього:
$$\mathrm{RMST}_{\text{model}}(T^\ast)=\int_0^{T^\ast}S_{\text{model}}(t)\,dt,\qquad
\Delta\mathrm{RMST}=\mathrm{RMST}_{\text{MM}}-\mathrm{RMST}_{\text{Micro}},$$
з $S$ за Kaplan–Meier і **однаковим** $T^\ast\le T_{\text{end}}$ для обох моделей;
completion-різницю звітувати окремо (log-rank). Альтернатива — підняти
$T_{\text{end}}$, доки completion $\to 1$. Це знімає найімовірніший артефакт
«premature arrival» **без зміни симулятора**.

### H — Boundary-degree variance / seeding → **параметризація + спільний seed** [impl/eval]
Параметризувати інтерфейс розподілом boundary-degree, а не скаляром $W$; свіпати
**дисперсію** при фіксованому середньому. Для seeding — використовувати **ідентичне**
правило ініціалізації в обох симуляторах (той самий вузол/розподіл), щоб зняти
ранню розбіжність.

### Зведення

| Джерело | Фікс | Тип | Вартість | Залишок |
|---|---|---|---|---|
| A | boundary-лічильник у $\lambda_{ij}$ | design | $O(1)$/громаду | C |
| B | fit $W_{ij}$ за integrated hazard | impl | одноразово | — |
| C | pair-closure / eff. $W$ | design | дешево | малий |
| D | degree-weighted placement | impl | низька | — |
| E | boundary-capacity booking | impl | низька | — |
| F | event-driven hazard (без сітки) | impl | нульова за точністю | — |
| G | KM + $\Delta$RMST на спільному $T^\ast$ | eval | нульова (лише аналіз) | — |
| H | degree-розподіл + спільний seed | impl/eval | низька | — |

**Порядок впровадження (рекомендований):** спершу **G** і **B** (обидва майже
безкоштовні й знімають найбільші confounders вимірювання/масштабу), потім **F**
(робить timing точним), і лише тоді оцінювати залишковий structural bias, який
адресують **A + D** (bridge-aware rate + placement). **C, E, H** — уточнення після
того, як головні confounders зняті.
