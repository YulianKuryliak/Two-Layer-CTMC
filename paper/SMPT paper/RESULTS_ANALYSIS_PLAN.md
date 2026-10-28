# План секції Results та рисунків

## 1. Мета першого етапу

На першому етапі числова частина статті має відповісти на три питання:

1. Чи може MicroMacro добре відтворювати маргінальний розподіл часу експорту, але порушувати microscopic-механізм передачі?
2. Як signed timing bias, беззнакова distributional error і незавершеність прогонів змінюються на наявній сітці параметрів?
3. Чи накопичується похибка вздовж ланцюга і чи зберігається різниця після вирівнювання локальної динаміки за arrival time?

Основний текст не повинен містити всі наявні рисунки. Для цих питань достатньо чотирьох змістовних figures. Повний grid, точні значення та додаткові sensitivity plots слід винести в Appendix або Supplement.

Цей документ є планом аналізу. До завершення поточного продовження симуляцій до (T_{\max}=30000) фінальні grid-метрики та висновки не фіксуються.

## 2. Джерело даних і обов'язковий data-freeze

### 2.1. Не змішувати різні покоління результатів

У `output/data/` одночасно лежать:

- історична 72-точкова сітка з (\beta\in\{0.001,0.003\}) і початковим горизонтом (4000);
- новий long-horizon запуск для (\beta=0.001), описаний `output/data/grid_manifest_latest.json`, із (T_{\max}=30000);
- додаткові пілотні та старі сценарії з іншими (L,n,d,k,\beta);
- старі consolidated CSV, які можуть не відповідати поточним checkpoint-ам і manifest-файлам.

Тому джерелом істини для нового аналізу мають бути лише `checkpoints/rep_*.json.gz`. Старі `runs.csv`, `community_arrivals.csv` і `trajectories.csv` не можна використовувати автоматично.

Primary grid у головному тексті: 36 long-horizon cells для (\beta=0.001):

\[
n\in\{50,100,250,1000\},\qquad
k\in\{1,5,10\},
\]

для clique communities та connected random communities з (d\in\{6,10\}).

Старі 36 cells для (\beta=0.003) аналізуються окремо й можуть увійти в Appendix лише після перевірки checkpoint-ів. Їх не слід непомітно об'єднувати з long-horizon (\beta=0.001) grid.

### 2.2. Перевірка після завершення поточного runner-а

Перед обчисленням метрик потрібно створити frozen analysis manifest і для кожної cell перевірити:

1. Є рівно 100 replication checkpoint-ів із унікальними `replication_id`.
2. Кожний checkpoint містить обидві моделі:
   `Full Micro` і `Production MicroMacro + independent RNG`.
3. У парі узгоджені scenario configuration, network seed, initial node та epidemiological parameters.
4. Є arrival records для communities (0,\ldots,9), а missing arrival не підмінено значенням (T_{\max}).
5. Перевірено фактичні `T_end`, `stop_reason` і `stop_time`, а не лише зовнішній `manifest.json`.
6. Для trajectories перевірено фактично збережений часовий інтервал і termination metadata.
7. Старі completed pairs, повторно використані з коротшого горизонту, допустимі для arrival-time analysis, якщо event уже відбувся. Для trajectory analysis їх можна продовжувати сталою траєкторією лише після документованого absorbing termination; horizon-censored trajectory так продовжувати не можна.

Frozen manifest має містити список scenario folders, replication IDs, фактичні horizons, seeds, completion/termination status і checksum або розмір/mtime checkpoint-файлів. Це захистить аналіз від зміни даних під час повторної побудови рисунків.

## 3. Політика метрик при censoring

### 3.1. Чи доцільно використовувати (W_1(\min(T,T_{\max})))?

Нехай

\[
Y=\min(T,\tau),\qquad
\delta=\mathbf 1\{T\leq\tau\},
\]

де (\tau=30000) є спільним administrative horizon.

Відстань

\[
W_1(Y^{\mathrm{Full}},Y^{\mathrm{MM}})
\]

математично коректна, але це **restricted/capped Wasserstein distance**, а не Wasserstein distance між повними розподілами (T). В одновимірному випадку вона дорівнює

\[
rW_{1,\tau}
=
\int_0^\tau
\left|
S_{\mathrm{MM}}(t)-S_{\mathrm{Full}}(t)
\right|\,dt.
\]

Вона не бачить жодної різниці в порядку подій після (\tau). Якщо більшість прогонів обох моделей не завершена, великі маси спостережень потрапляють у точку (\tau), і restricted (W_1) може виглядати штучно малим. Це не numerical perturbation у звичайному сенсі, а зміна estimand: замість повного arrival-time distribution порівнюється лише його частина до горизонту.

Отже:

- не називати (T_{\max}) фактичним arrival time;
- не називати (rW_{1,\tau}) звичайним (W_1(T));
- завжди показувати completion разом із restricted distance;
- не робити висновок про точність необмеженого хвоста без додаткових assumptions.

### 3.2. Primary metrics для кожної grid cell

За наявності хоча б одного censored run використовувати три співпервинні характеристики:

1. **Completion до (\tau)**

   \[
   p_{\mathrm F}(\tau)=P(T_9^{\mathrm F}\leq\tau),\qquad
   p_{\mathrm{MM}}(\tau)=P(T_9^{\mathrm{MM}}\leq\tau),
   \]

   а також парну різницю (\Delta p=p_{\mathrm{MM}}-p_{\mathrm F}). У machine-readable table зберегти чотири paired categories: both complete, Full-only, MM-only, neither.

2. **Signed timing bias**

   \[
   \Delta\mathrm{RMST}_\tau
   =
   E[\min(T_9^{\mathrm{MM}},\tau)]
   -
   E[\min(T_9^{\mathrm F},\tau)].
   \]

   Негативне значення означає раніше поширення MicroMacro в обмеженому вікні. Показувати також нормовану величину

   \[
   \Delta\mathrm{RMST}_\tau/
   \mathrm{RMST}^{\mathrm F}_\tau.
   \]

3. **Unsigned distributional error**

   \[
   rW_{1,\tau}
   =W_1(\min(T_9^{\mathrm F},\tau),
         \min(T_9^{\mathrm{MM}},\tau)).
   \]

   Показувати absolute value та, як secondary scale-free statistic,

   \[
   rW_{1,\tau}/\mathrm{RMST}^{\mathrm F}_\tau.
   \]

Restricted KS,

\[
rKS_\tau=
\sup_{t<\tau}|S_{\mathrm{MM}}(t)-S_{\mathrm F}(t)|,
\]

залишити secondary metric для таблиць і Appendix. Не використовувати стандартний `scipy` KS p-value: observations є paired, а capped values створюють ties.

### 3.3. Якщо всі прогони завершилися до 30000

Якщо в певній cell обидві моделі мають 100/100 arrivals, restricted quantities автоматично збігаються зі звичайними:

- (\Delta\mathrm{RMST}=E[T_9^{\mathrm{MM}}]-E[T_9^{\mathrm F}]);
- (rW_1=W_1(T_9^{\mathrm F},T_9^{\mathrm{MM}}));
- (rKS=KS(T_9^{\mathrm F},T_9^{\mathrm{MM}})).

У такій cell в тексті можна використовувати звичайні назви mean shift, Wasserstein-1 і KS. Якщо censored хоча б один run, restricted nomenclature зберігається для всієї цієї cell.

Completed-only (W_1), mean або local delay не використовувати як primary metrics: conditioning on completion по-різному відбирає швидкі реалізації двох моделей. Їх можна дати лише як sensitivity analysis з явним попередженням.

Quantile (q) звітувати лише тоді, коли він identifiable в обох моделях: median потребує completion понад (50\%\), (q_{0.9}) — понад (90\%\).

### 3.4. Uncertainty

Для всіх contrasts використовувати paired nonparametric bootstrap на рівні replication/network block, зберігаючи Full/MM pair. Рекомендовано 5000 bootstrap resamples з окремим deterministic analysis seed.

Bootstrap intervals потрібні для:

- (\Delta\mathrm{RMST}_\tau) або uncensored mean shift;
- (rW_{1,\tau});
- (rKS_\tau) у supplementary table;
- (\Delta p);
- layer-wise contrasts.

Не називати IQR окремих paired differences «похибкою методу»: він залежить від stochastic variability та від способу pairing RNG. За 100 pairs/cell grid залишається exploratory; не робити 36 або 72 окремих significance claims без multiplicity correction.

### 3.5. Horizon-sensitivity як перевірка збурення

У Appendix варто повторно обчислити restricted metrics із уже зібраних long-horizon arrivals для

\[
\tau\in\{4000,10000,20000,30000\}.
\]

Для кількох раніше сильно censored cells показати, як разом змінюються completion, (\Delta\mathrm{RMST}_\tau) і (rW_{1,\tau}). Це безпосередньо продемонструє, чому малий capped (W_1) при короткому горизонті не є доказом точності повного розподілу. Нових симуляцій для цього не потрібно.

## 4. Figures для головного тексту

## Figure 1. Clique--bridge--leaf: близький endpoint, порушений механізм

Один compact two-panel figure:

### Panel A — total export-time ECDF

- ECDF Full Micro і Production MicroMacro;
- (N=10000) для кожної моделі;
- у caption або невеликому annotation: (W_1=3.249), relative (W_1=2.95\%\), (KS=0.0142);
- mean export times (110.18) і (112.62);
- paired shift (+2.43), bootstrap 95% CI ([-0.46,5.15]).

ECDF тут читабельніший за дві накладені KDE/density curves і не створює залежності від bandwidth.

### Panel B — bridge-ordering diagnostic

- signed quantity
  \[
  D=T_{\mathrm{export}}-T_{\mathrm{bridge}};
  \]
- чітка vertical line при (D=0) і заштрихована область (D<0);
- direct annotation: Full (0/10000), MicroMacro (98/10000=0.98\%\);
- пояснення, що для Full `D` є causal waiting time, а для MM — signed diagnostic difference;
- якщо повний long right tail робить negative region невидимим, основна панель має бути zoom біля нуля, а повний distribution — inset або Appendix.

Центральний текстовий висновок:

> Close agreement of the marginal export-time distribution does not imply preservation of the microscopic transmission mechanism.

Bridge-activation distribution і повні density panels можна залишити в Appendix; у головному тексті вони не потрібні окремими великими рисунками.

## Figure 2. Accuracy atlas на long-horizon grid

Основний atlas побудувати для (\beta=0.001), (T_{\max}=30000). Рекомендований layout — одна full-width figure з heatmap small multiples:

- rows: `complete`, random connected (d=10), random connected (d=6);
- columns:
  1. signed relative timing bias;
  2. relative distributional error;
  3. completion information.

У кожному panel:

- horizontal axis: (n=50,100,250,1000);
- vertical axis: (k=1,5,10), обов'язково в такому порядку зверху вниз;
- однакові limits і color scales для порівнюваних panels;
- signed bias — diverging scale, симетрична навколо нуля;
- distributional error — sequential nonnegative scale;
- completion — color за
  \[
  p_{\min}=\min(p_{\mathrm F},p_{\mathrm{MM}}),
  \]
  а в cell annotation або supplementary table — (\Delta p).

Якщо після (T_{\max}=30000) completion дорівнює 100% у всіх 36 cells, третю column не слід залишати як неінформативну суцільну карту. Тоді:

- у тексті дати одну перевірену фразу про 100% completion;
- completion atlas перенести в Appendix як validation;
- main Figure 2 скоротити до (3\times2) panels: signed bias і relative (W_1).

Якщо censored cells залишаться:

- перші дві columns повинні містити (\Delta\mathrm{RMST}_\tau) і (rW_{1,\tau});
- censored cells позначити hatch/outline;
- не змішувати в одному color scale restricted і unrestricted values без позначення;
- exact estimates та confidence intervals подати в Appendix table.

У main text описувати не кожну cell, а лише patterns:

- чи повторюється раннє/пізнє зміщення в sparse regimes;
- чи змінюється знак між regimes;
- чи є monotonic dependence on (n,k,d);
- де різні визначення accuracy не погоджуються;
- чому малий unsigned error не компенсує poor completion.

Не створювати єдиний composite «accuracy score» і не ранжувати cells одним числом.

## Figure 3. Накопичення arrival-time error по communities

Вибрати наперед структурно мотивовані representative cases, а не лише cells із найбільшим observed effect:

1. clique control: (n=100,k=5,\beta=0.001);
2. topology-matched sparse case: random connected (n=100,d=6,k=5,\beta=0.001);
3. narrow-interface large-community stress case: random connected (n=1000,d=10,k=1,\beta=0.001), якщо після data-freeze є достатня completion.

Figure має два panels із community index (j=0,\ldots,9):

### Panel A — signed cumulative timing bias

\[
\Delta_j
=
E[T_j^{\mathrm{MM}}]-E[T_j^{\mathrm F}]
\]

для fully observed layers або (\Delta\mathrm{RMST}_{j,\tau}) при censoring. Показати estimate і paired-bootstrap 95% interval/band. Нульова horizontal reference line обов'язкова.

### Panel B — layer-wise distributional discrepancy

\[
W_1(T_j^{\mathrm F},T_j^{\mathrm{MM}})
\]

або (rW_{1,j,\tau}) при censoring. Для зіставлення regimes додати relative version, нормовану на Full RMST/mean arrival time відповідного layer.

Якщо completion різниться за layer, додати тонкий третій strip або Appendix panel із (p_j^{\mathrm F}) та (p_j^{\mathrm{MM}}). Не рахувати mean local delays лише серед completed runs.

У тексті розрізняти:

- cumulative error (T_j^{\mathrm{MM}}-T_j^{\mathrm F});
- local hop delay
  \[
  (T_j-T_{j-1})^{\mathrm{MM}}-(T_j-T_{j-1})^{\mathrm F}.
  \]

Local hop-delay profiles для selected cases подати в Appendix. Вони потрібні, щоб відрізнити одноразовий initial offset від повторюваної transition error.

## Figure 4. Локальна динаміка після arrival-time alignment

Цей figure перевіряє, чи різниця між моделями переважно є зсувом часу приходу, чи зберігається і після активації спільноти.

Для кожної replication, model і community визначити

\[
s=t-T_{r,j}^{(m)},
\qquad
\widetilde I_{r,j}^{(m)}(s)
=
\frac{I_{r,j}^{(m)}(T_{r,j}^{(m)}+s)}{n}.
\]

Вирівнювання виконується за arrival time кожної моделі у кожній replication, а не за середнім або median arrival time.

Основний figure: (2\times3) small multiples.

- rows: clique control (n=100,k=5) та matched sparse case (n=100,d=6,k=5);
- columns: early, middle і terminal communities, наприклад (j=1,5,9);
- Full та MicroMacro — дві стабільні line styles/colors;
- y-axis — infected fraction `I_j/n`, спільна шкала `[0,1]`;
- x-axis — time since first local infection (s), однаковий window усередині row;
- pointwise mean curves із paired-bootstrap uncertainty bands або median з IQR, але caption повинен точно називати тип band.

Правила coverage:

- вибрати (s_{\max}), для якого в кожній plotted curve залишається наперед визначений мінімум paired contributors, наприклад 90 із 100;
- показати або зберегти (N(s));
- не переносити останнє значення за horizon-censored trajectory;
- після documented absorbing termination plateau можна продовжити точно, бо стан уже не змінюється;
- не порівнювати panels із різним (s_{\max}) так, ніби windows однакові.

Інтерпретація:

- якщо Full/MM curves майже збігаються після alignment, головна різниця пов'язана з arrival timing;
- якщо різниця зберігається, це сумісно з впливом import placement, repeated imports або їхньої взаємодії;
- цей figure є діагностикою, а не окремим causal decomposition rate/placement/synchronization error.

## 5. Що описувати в Results

Рекомендована структура тексту:

### 5.1. Mechanistic fidelity can fail despite distributional agreement

- коротко нагадати clique--bridge--leaf mechanism;
- навести (W_1), relative (W_1), KS та mean-shift CI;
- центральним результатом зробити 98 bridge-bypass histories;
- пояснити signed MM bridge-to-export diagnostic;
- не заявляти встановленого late bias і не подавати point estimate як спростування Jensen result.

### 5.2. Accuracy is regime- and observable-dependent

- представити common grid protocol і data-freeze;
- окремо визначити signed bias, distributional error і completion;
- описати patterns atlas, не каталог усіх 36 cells;
- наголосити, якщо sign changes між regimes;
- не стверджувати monotonic improvement із (n,k,d), якщо його немає;
- при censoring використовувати restricted terminology.

### 5.3. Arrival-time discrepancies accumulate along the chain

- показати layer-wise cumulative shift та (W_1/rW_1);
- перевірити, чи зростання є послідовним або виникає на окремих transitions;
- відділити cumulative arrival error від local hop-delay error;
- не інтерпретувати широкий IQR paired paths як deterministic method error.

### 5.4. Arrival alignment separates timing from post-arrival dynamics

- описати matched clique/sparse comparison;
- вказати coverage window і summary statistic;
- зробити висновок лише про те, чи зникає видима різниця після time alignment;
- не приписувати залишок конкретно placement error без ablation experiment.

### 5.5. Limits of the present evidence

- SI, (\gamma=0);
- chain macro topology;
- clique та connected-random micro graph ensembles;
- 100 paired replications per grid cell;
- fixed (\beta_{\mathrm{out}}/\beta_{\mathrm{in}}=1);
- exploratory grid without a universal accuracy threshold;
- production comparison combines rate, placement і synchronization effects.

## 6. Appendix / Supplement

До Appendix винести:

1. Повну table всіх primary grid cells із configuration, completion, four paired completion categories, signed shift, (W_1/rW_1), KS/rKS і bootstrap intervals.
2. Повні heatmaps для historical (\beta=0.003) grid після checkpoint audit.
3. Horizon-sensitivity plots для (\tau=4000,10000,20000,30000).
4. Survival curves для кількох cells із залишковим censoring.
5. Full layer-wise arrival-error profiles, згруповані на окремих сторінках за topology/density та (n), із послідовним порядком (k=1,5,10).
6. Local hop-delay profiles для representative cells.
7. Додаткові arrival-aligned trajectories; не вставляти один гігантський unreadable grid.
8. Exact metric definitions, bootstrap seed, number of resamples і analysis manifest identifier.

Повна сітка має бути доступною як numerical table і набір узгоджених atlas pages. Це корисніше й читабельніше, ніж десятки майже однакових великих trajectory figures.

## 7. Planned analysis artifacts

Після завершення симуляцій доцільно створити окремі reproducible analysis scripts:

- `extrascripts/analysis/analyze_smpt_results_grid.py` — checkpoint validation, censoring-aware metrics, bootstrap і frozen tables;
- `extrascripts/visualization/plot_smpt_results.py` — лише побудова figures із frozen summary data.

Рекомендований output:

```text
output/analysis/smpt_results_v1/
    analysis_manifest.json
    checkpoint_validation.csv
    cell_metrics.csv
    layer_metrics.csv
    aligned_trajectory_summary.csv
    horizon_sensitivity.csv
    figures/
        fig_clique_bridge_leaf.pdf
        fig_accuracy_atlas.pdf
        fig_arrival_error_accumulation.pdf
        fig_arrival_aligned_local_dynamics.pdf
    appendix/
        full_grid_metrics.csv
        full_grid_atlas_*.pdf
        survival_selected_cells.pdf
        horizon_sensitivity.pdf
```

Analysis scripts не повинні змінювати або консолідувати simulation checkpoints. Усі derived files пишуться в versioned analysis directory.

## 8. Decision gates

1. **Runner still active:** не фіксувати grid results і не будувати publication figures.
2. **Runner finished, all cells 100/100 complete:** використовувати ordinary mean shift, (W_1), KS; completion залишити validation result.
3. **Runner finished, residual censoring exists:** використовувати completion + (\Delta\mathrm{RMST}_\tau) + (rW_{1,\tau}); не екстраполювати tail.
4. **Trajectory coverage sufficient:** будувати Figure 4 на preselected cases.
5. **Trajectory coverage insufficient:** скоротити (s_{\max}), показати (N(s)) або замінити stress case на fully observed structurally matched cell; не заповнювати missing data штучно.
6. **Historical (\beta=0.003) checkpoint audit passed:** включити scaling/sensitivity analysis в Appendix; інакше не використовувати старі consolidated CSV як фінальний evidence.

## 9. Найважливіше методологічне рішення

Для цієї статті не потрібна одна універсальна «метрика точності». Completion, signed timing bias і unsigned distributional discrepancy відповідають на різні питання й мають залишатися окремими.

Головна narrative line першого етапу:

> The fidelity of the aggregate interface is observable- and regime-dependent: close endpoint distributions can coexist with mechanistically impossible transmission histories, and apparently small restricted distances can be produced by censoring rather than genuine agreement.
