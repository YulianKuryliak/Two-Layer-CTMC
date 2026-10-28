# Час експорту з однієї спільноти: шляхи, спектр і похибка MicroMacro

Дата: 2026-09-18. Статус: дослідницький план і технічне завдання для GPT-5.6 Luna; нові симуляції для цього плану ще не виконано.

Цільовий рукопис: `paper/SMPT paper/main-SMRT (1).tex`. Цей документ доповнює `RESULTS_ANALYSIS_PLAN.md`; не замінює чинні результати benchmark-ів.

## 1. Питання і рішення щодо дизайну

Дослідити окремо:

1. Від чого залежить **справжній час першого експорту** у Full SI CTMC?
2. Які властивості спільноти пояснюють **похибку часу експорту** Production MicroMacro з independent RNG?

Ці питання не тотожні. Спільнота може експортувати повільно, але MicroMacro правильно відтворюватиме цей час; або швидко, але з істотною відносною похибкою. Знак похибки також не заданий наперед: MM може експортувати як раніше, так і пізніше.

Основний об'єкт: **одна спільнота-джерело з n вузлів + один зовнішній вузол-приймач**. У коді це дві групи розмірів `[n, 1]`, а не ланцюг. Приймач служить тільки реєстратором першого експорту. Немає накладання затримок сусідніх спільнот або повторних імпортів до джерела перед першим експортом.

Початковий інфікований вузол `s` у джерелі однаковий для Full і MM. До приймача `z` проведено `k` ребер із різних вихідних вузлів `B`. У первинному дизайні `s` не належить `B`. Саме тут `k` — число вихідних ребер, а не середній внутрішній ступінь і не число спільнот.

Ключове рішення: **не намагатися оголосити D_mean або eigenvalue універсальною причиною затримки**. Спочатку виконати контрольовані порівняння, які можуть спростувати достатність кожного показника, і лише потім оцінювати його прогностичну користь.

## 2. Механістична основа та виправлення попередніх спрощень

### 2.1. Безпосередній механізм експорту

Нехай `X_v(t)` — індикатор інфікованості вузла джерела, а `I(t)=sum_v X_v(t)`. За одиничних вихідних ваг до першого зараження приймача:

\[
h_{\rm Full}(t)=\beta_{\rm out}\sum_{b\in B}X_b(t),
\qquad
h_{\rm agg}(t)=\beta_{\rm out}k\frac{I(t)}{n}.
\]

Друга формула — aggregate closure базового Production при `macro_T=1`, незараженому приймачі та вимкнених corrections. Вона замінює частку заражених **на виході** часткою заражених **у всій спільноті**.

Для ізольованої внутрішньої SI-траєкторії джерела, з незалежним годинником експорту:

\[
H(t)=\int_0^t h(u)\,du,
\qquad
P(T_{\rm export}>t\mid X)=e^{-H(t)},
\qquad
S(t)=\mathbb E[e^{-H(t)}].
\]

Це умовлення на контрфактичну внутрішню траєкторію без зворотного зараження від приймача. До першого експорту вона має потрібний закон. Не підставляти в цю формулу траєкторію джерела після зворотних передач від уже зараженого приймача.

Найближчий до механізму діагностичний показник:

\[
q_B(t)=\frac{1}{k}\sum_{b\in B}X_b(t)-\frac{I(t)}{n},
\qquad
H_{\rm Full}(t)-H_{\rm agg}(t)
=\beta_{\rm out}k\int_0^t q_B(u)\,du.
\]

На **тій самій внутрішній траєкторії** негативний q_B означає, що aggregate hazard завищує поточну інтенсивність експорту. Якщо інтегральний порядок зберігається для всіх t, спільний Exp(1)-поріг дає відповідний порядок export times. Один негативний фрагмент q_B або його усереднений знак сам по собі цього не гарантує.

Розкладати Full export time на `T_bridge_first` і залишок після нього. За k=1 залишок — незалежний Exp(beta_out), тому вся структурна залежність лежить у часі до bridge. За k>1 після першого bridge заражаються інші виходи, hazard зростає; залишок загалом не є Exp(beta_out*k) і не незалежний від внутрішньої історії. Такий розклад допоможе розрізнити затримку досягнення інтерфейсу та конкуренцію вихідних ребер.

Небажано пояснювати затримку лише середньою кривою I(t). Також не прирівнювати `E[exp(-H)]` до `exp(-E[H])`: неоднорідність траєкторій впливає на розподіл експорту. Наведені формули — прямий вивід для зазначеного дизайну, а не вже встановлений результат нового експерименту.

### 2.2. Що означає шлях у SI

Для SI без одужання з незалежними сталими per-edge rates час досягнення вузла має представлення через **first-passage percolation**: мінімум сум випадкових часів передачі вздовж шляхів. Для ребра e час має розподіл `Exp(beta_in * w_e)`. Отже, шлях із найменшою кількістю ребер не обов'язково є шляхом найшвидшого зараження. Це стандартний зв'язок SI/FPP; див. [Aldous, слайди 1–3](https://www.stat.berkeley.edu/~aldous/FMIE/warwick_4.pdf). Дослідження [Bhamidi, van der Hofstad, Hooghiemstra](https://arxiv.org/abs/0903.5136) також розрізняє вагову відстань і hopcount; їхні асимптотичні результати не є готовою формулою для нашого скінченного графа.

Для нашого фіксованого старту s і одного приймача це дає представлення **за розподілом**:

\[
T_{\rm export}\overset d=
\min_{b\in B}\{T^{\rm FPP}(s,b)+E_b\},
\qquad E_b\sim\operatorname{Exp}(\beta_{\rm out}),
\]

де всі внутрішні FPP-відстані обчислено на одному випадково зваженому графі, тому вони корельовані; E_b незалежні від внутрішніх ваг. Не генерувати незалежний внутрішній граф для кожного b і не називати це pathwise-відтворенням production за тим самим seed. Формула тут пояснює механізм; замінювати Full simulator на FPP у первинних результатах не потрібно.

За **єдиного** маршруту довжини ell до одного bridge-вузла і k=1:

\[
T_B\sim\operatorname{Gamma}(\ell,\mathrm{rate}=\beta_{\rm in}),
\quad
\mathbb E T_{\rm export}=\frac{\ell}{\beta_{\rm in}}+
\frac{1}{\beta_{\rm out}}.
\]

При рівних beta весь export time має Gamma(ell+1, rate=beta). Це точний контроль, а не наближення для довільної мережі. Для довільного невагового графа з k=1 довжина одного найкоротшого шляху дає верхню межу очікування через цей маршрут; альтернативні шляхи можуть прискорити зараження.

### 2.3. Який саме спектр

Не використовувати слово «eigenvalue» без матриці та індексу.

| Показник джерела | Зміст | Чого з нього не випливає |
|---|---|---|
| rho(A), найбільше власне значення adjacency | Містить інформацію про можливе швидке зростання; виникає в лінійних/mean-field оцінках | Не визначає час до конкретного виходу |
| lambda2(L), L=D-A | Algebraic connectivity комбінаторного лапласіана; масштаб залежить від ступенів/ваг | Не порівнювати чисельно з normalized lambda2 як з тією самою величиною |
| lambda2(L_norm), L_norm=I-D^(-1/2)AD^(-1/2) | Глобальні bottlenecks і спектральна релаксація відповідного random walk | `T_export = 1/(beta*lambda2)` не є формулою SI |

Для random walk зі стрибками сумарної частоти 1 спектральна релаксація пов'язана з normalized gap; для процесу зі стрибками сталої частоти на кожному ребрі — з combinatorial Laplacian. Mixing time і relaxation time також не тотожні: потрібні умови та додаткові множники. Ці поняття викладено у [Levin–Peres–Wilmer, розділи 12 і 20](https://darkwing.uoregon.edu/~dlevin/MARKOV/mcmt2e.pdf). SI — розгалужений процес зараження, а не один випадковий блукач. Попереднє формулювання «більша lambda2 означає швидше поширення SI» треба вважати гіпотезою з обмеженою застосовністю.

Наприклад, для точних SI marginal probabilities:

\[
\dot p_i=\beta\sum_j A_{ij}\mathbb E[(1-X_i)X_j]
\leq \beta\sum_j A_{ij}p_j.
\]

Звідси видно роль adjacency spectrum в лінійній верхній оцінці, але не точний export time. Не переносити SIS epidemic-threshold формули на наш SI без одужання; кореляції та обмеження спектральних висновків обговорено в [Van Mieghem](https://arxiv.org/abs/1402.1731).

Спектр обчислювати **лише для внутрішнього графа джерела**. Додавання приймача змінює повний спектр і зіпсує контроль «незмінний граф, різні виходи».

На d-регулярних зв'язних графах rho(A)=d точно. Це зручний контроль: можна змінювати геометрію та bottleneck при незмінному spectral radius. Але незмінність rho не означає незмінність решти спектра.

### 2.4. Чому один scalar не може бути універсально достатнім

1. На тому самому графі можна пересунути B. Внутрішній спектр не зміниться взагалі, але відстані від s та час до виходів можуть змінитися.
2. Можна залишити єдиний шлях s–b та змінити бічний підграф, приєднаний тільки в s. Глобальний спектр зміниться, а закон Full часу до b не зміниться при незалежних per-edge clocks.
3. За однакової hop distance можна додати паралельні маршрути. Перший успішний маршрут може скоротити час, хоча мінімальна кількість ребер та сама.

Ці конструкції дають сильніші висновки про недостатність показника, ніж кореляція по випадкових графах. Вони не означають, що D або спектр непридатні як предиктори у конкретному класі мереж.

## 3. Спільний протокол

Початковий конфіг на початку нового runner-а:

```python
MODEL = 2                       # SI у поточному коді; перевірити у smoke
GAMMA = 0.0
BETA_IN = 0.003
OUT_IN_RATIOS = [1.0]            # beta_out / beta_in; sensitivity окремо
MACRO_T = 1.0
TAU_MICRO = 1.0
T_MAX = 30_000.0                 # верхня межа, не обов'язкова довжина траєкторії
INITIAL_NODE = 0
BASE_SEED = 20260918
WORKERS = 6
HEAVY_WORKERS = 2
HEAVY_N = 1000
GRAPH_REPLICATES = 10            # первинна exploratory сітка
EVENT_REPLICATES = 10            # 100 пар на cell, 10 незалежних graphs
COMMUNITY_SIZES = [50, 100, 250, 1000]
PRIMARY_N = 100
INTERNAL_DEGREE = 6
K_VALUES = [1, 5]
BLOCKS = ["E0", "E1", "E2"]      # E3/E4 явними додатковими профілями
OUTPUT_ROOT = "output/data"
```

100 пар на cell — початкова точність, а не гарантія переконливого висновку. Для фінальних ключових контрастів рекомендовано 30 graphs × 100 event replicates, з повторним використанням уже виконаних пар. Для детермінованої топології без структурної рандомізації не створювати фіктивні незалежні graph replicates: один graph_id та 100 незалежних stochastic runs.

В одному graph block той самий внутрішній граф, s, rates, node order та event replicate зіставляються для near/middle/far виходів. Однакові числові seeds Full/MM не означають однакових випадкових подій. Pairing означає спільну задачу та мережу.

Зберігати окремі deterministic seeds для topology, gateway selection, Full events, MM macro events і кожного MM MicroModel. Генерувати з canonical ключа через hashlib/SeedSequence, не через Python `hash()`. Фактичні числові seeds обов'язково записувати.

Primary comparison: **Full Micro і Production MicroMacro + independent RNG**. Cache discard, clone/refresh semantics, midpoint, tau scheduler, uniform placement, normal imports залишаються production. Спектральні та boundary corrections вимкнені. Independent RNG виправляє ownership потоків, але не додає інформацію про bridge-вузли у macro hazard.

Оскільки приймач один і до експорту не інфікований, uniform destination placement тут тривіальне. Тому дизайн досліджує передусім source-rate abstraction та synchronization, а не помилку розміщення імпорту у великій приймальній спільноті.

## 4. E0 — аналітичні контроли: шлях, обхідні маршрути і бічна маса

### E0a: незмінний шлях, різний глобальний спектр

Для n=100, k=1, ell у `[2, 4, 8]` побудувати corridor `0--1--...--ell`; s=0, b=ell. Решта `n-ell-1` вузлів утворює бічний підграф, приєднаний **одним ребром тільки до s**. Варіанти бічного підграфа: path, clique, connected random з mean degree приблизно 6. Жодних ребер із нього до corridor, крім одного attachment до s.

Усі варіанти мають той самий n, ell, k, source degree=2, bridge degree=1, rates. Глобальний середній ступінь не контролюється; це навмисний аналітичний контрприклад, не degree-matched тест lambda2.

**Точне передбачення:** розподіл Full export time однаковий для трьох side topologies при фіксованому ell і дорівнює Gamma(ell+1, beta). Бічні зараження не «відбирають швидкість» у незалежних clocks ребер corridor. У Gillespie зміниться послідовність загальних подій, але не marginal law на corridor.

MM використовує I(t)/n, тому зміна бічного зростання може змінювати його час експорту. Це гіпотеза для перевірки; не фіксувати очікуваний знак без результатів. Якщо підтвердиться, приклад покаже реагування aggregate export на зараження, які не відкривають додаткового маршруту до справжнього виходу.

Зберегти eigenvalues і справжні ступені всіх варіантів; не припускати конкретний порядок lambda2 без розрахунку. Достатньо 100 пар на варіант для початку; Gamma validation зробити окремою reusable малою перевіркою, за потреби до 2000 дешевих Full replicates.

### E0b: та сама найкоротша відстань, більше альтернативних шляхів

n=100, k=1, ell=4; між s і b побудувати q внутрішньо vertex-disjoint шляхів довжини ell, q у `[1, 2, 4]`. Незайняті вузли приєднати як бічну path тільки до s, щоб зберегти n. До b додати одне export edge.

Час до b — мінімум q незалежних Gamma(ell, beta_in); після зараження b додається незалежний Exp(beta_out). Мінімальна hop distance завжди ell, але час досягнення виходу зменшується стохастично з q. Ступені s і b змінюються — явно записати; цей блок не є незалежним контролем degree effect.

**Навіщо E0:** два недубльовані аналітичні факти: спектр може змінюватися без зміни export time; distance може не змінюватися при зміні export time. E0 не слід включати у pooled regression випадкових графів.

## 5. E1 — той самий граф і весь спектр, різне розташування виходу

Це головний емпіричний блок. n=100, d=6, k у `[1,5]`.

Сімейства джерела:

1. `random_regular`: зв'язний 6-регулярний граф через NetworkX.
2. `modular_regular_cut4`: два 6-регулярні підграфи по 50 вузлів; двома degree-preserving cross swaps створити 4 cross edges. Це внутрішні модулі **однієї** MicroModel, а не macro communities.
3. `random_connected_gnm`: використати саме наявну семантику `network.py` (G(n,m), відбір за зв'язністю, m=round(n*d/2)). Не називати цей код Bernoulli G(n,p).

Для кожного незмінного graph_id та s підібрати `B_near`, `B_middle`, `B_far` з однаковим multiset внутрішніх ступенів. Вимірювати:

\[
D_{\rm mean}=k^{-1}\sum_{b\in B}d_G(s,b),
\qquad D_{\min}=\min_{b\in B}d_G(s,b).
\]

Це відстані від **одного фактичного seed**, не середня по двох множинах gateway-вузлів. Не використати непомітно старий k×k D_mean із chain runner-а.

Алгоритм вибору: на regular graphs всі не-seed вузли мають degree 6; ранжувати за distance від s, випадково розв'язувати ties окремим gateway seed, формувати низький/середній/високий attainable mean. На G(n,m) спочатку вибрати спільний degree multiset із достатнім просторовим діапазоном, потім шукати k-набори лише з цим multiset. Кандидатний бюджет `MAX_GATEWAY_CANDIDATES=10000`, фактичні цільові/досягнуті відстані записати. Множини B різних умов можуть перетинатися; s у B заборонено.

Якщо degree-matched distinct distance conditions недосяжні, зберегти `design_unavailable` і пояснення. Не підміняти near/far ідентичними наборами, не послаблювати matching мовчки та не відбирати graphs за вже отриманою похибкою симулятора. До симуляцій вивести feasibility report.

**Особливо сильне передбачення:** для basic MM за тих самих G, s, n, k, weights, seeds і вимкнених corrections переміщення B не повинно змінювати час першого експорту. W і внутрішня динаміка, видимі алгоритму, однакові. У Full час може змінитися. Перевірити equality MM export times/префіксів як regression invariant; не додавати різний event seed залежно від near/middle/far.

Вихідні різні B можуть мати однаковий D_mean, але інший D_min, розкид або gateway centrality; все зберігати. Висновок цього блока — про **geometry of the output interface при незмінному спектрі**, а не доведення, що лише D_mean спричиняє весь ефект.

Для reference додати один clique n=100, k=5, s не в B, 100 event replicates. Відстані до всіх виходів дорівнюють 1, а перестановки B симетричні за законом. Це reference, а не density-matched порівняння.

Початковий бюджет E1: 3 topologies × 2 k × 3 placements × 100 пар = 1800 пар плюс 100 clique pairs. MM може бути фізично обчислений один раз на graph/event/k і повторно використаний у трьох placement conditions **лише після проходження invariant test**; зберігати shared-result provenance, не рахувати копії незалежними реплікаціями.

## 6. E2 — внутрішній bottleneck при фіксованих n, d і rho(A)

Мета: перевірити додаткову роль bottleneck/normalized gap та розташування виходу відносно нього.

n=100, d=6, k=5, два внутрішні блоки по 50 вузлів. Взяти два connected 6-regular graphs. Degree-preserving swap: видалити internal edges (a,b) і (c,d) з різних блоків та додати (a,c), (b,d), якщо немає петель/дублікатів. Створити вкладені варіанти з кількістю cross edges `CUT_EDGES=[2,4,16]`. Перевіряти connectedness усього графа та degree=6 після кожної конструкції. Це окремий генератор контрольованої топології, не зміна production `network.py`.

В усіх варіантах rho(A)=6; normalized gap, local expansion, D і число альтернативних шляхів можуть змінюватися. Не називати втручання «змінили тільки lambda2».

Seed завжди у першому блоці. Для кожного cut level порівняти виходи:

- `same_side`: усі B у тому самому внутрішньому блоці, що s;
- `opposite_side`: усі B в іншому блоці.

Спочатку topology-only підбір: бажаний exact distance histogram `[3,3,3,3,3]` у всіх шести умовах graph block. Якщо неможливо, спробувати D=2, потім D=4, у фіксованому порядку; вибрана D однакова для всіх умов matched block. Якщо жодна спільна shell не має ≥5 вузлів, зберегти graph як unmatched. Не робити необмежений random search.

Первинний механістичний контраст — subset із exact common histogram та degree 6; unmatched graphs можуть увійти лише в окремий exploratory аналіз з фактичними D. До основного запуску повідомити кількість matched graph blocks. Якщо їх менше 10 при початковому пулі 30, не запускати великий автоматичний пошук: вивести feasibility limitation і залишити E2 як exploratory. Початковий topology pool та відбір не залежать від epidemic results.

Дві різні перевірки:

1. У тому самому graph variant same/opposite виходи мають однакову відстань, а спектр однаковий; відмінність export law покаже недостатність distance+global eigenvalues без локалізації виходу.
2. Між cut variants дослідити зміну часу разом із lambda2 при контрольованому n,d,rho та distance histogram. Це ефект внутрішньої структури/bottleneck, а не доведена причинна дія одного eigenvalue.

Початковий бюджет: до 3 cuts × 2 sides × 100 пар = 600 пар. Вхідні graphs та частина `cut4` можуть бути спільні з E1; reusable network_id і simulation_id визначати за точними даними, а не лише назвою блока.

## 7. E3 — однакова абсолютна відстань, різний розмір

Цей блок зберігає початкове питання користувача про n. Виконувати після E0/E1; він не замінює основне розділення geometry/spectrum.

`n=[50,100,250,1000]`, connected random regular d=6, `k=[1,5]`, усі виходи на **distance=2** від s. Exact histogram `[2]*k`, а не намагання влучити у D_mean=2.72 у кожній випадковій спільноті. За недостатньої shell записати infeasible; не запускати нескінченний пошук. Передбачити до 30 topology candidates на size, потрібні перші 10 feasible згідно з детермінованим порядком.

Тут ступені, rho(A), k і distance histogram фіксовані, але n, k/n, обсяг поширення, alternatives і normalized gap можуть змінюватися. Це ефект розміру **за таким matching**, не універсальний чистий ефект n. При однаковій абсолютній відстані відносне положення у графі може відрізнятися.

Додатково зберігати ASPL_est та `D_mean/ASPL_est`, але не оголошувати друге відношення фізичним законом чи primary predictor. ASPL_est: BFS із до 32 збережених випадково обраних source nodes; для n≤100 допустимо exact all-sources BFS. Метод і sampling seed записати.

Початковий бюджет: 4 sizes × 2 k × 100 пар = 800 пар. Малий n не замінювати більшим заради підбору shell. Дані старого L=5 chain експерименту не є прямими даними цього односпільнотного тесту.

## 8. E4 — sensitivity до export competition і tau, без великого beta-grid

За SI зі сталими вагами одночасне множення beta_in і beta_out на c масштабує точний CTMC час у 1/c. Тому варіювати спільне beta як незалежний «топологічний фактор» зайве. Для production зі сталим tau масштабна інваріантність може порушуватися чисельно; щоб зберегти dimensionless scheduler scale, tau також треба ділити на c.

Більш змістовний sensitivity: `beta_out/beta_in=[0.1,1,10]` при фіксованому beta_in=0.003, на 3 заздалегідь вибраних representative designs: E1 random regular near/far та E2 cut2 opposite. Для досяжності горизонту спочатку pilot, без підбору rates за ступенем згоди моделей.

Поточний `_run_experimental_micromacro` приймає один benchmark.beta. Без зміни алгоритму підтримати ratio через **вагу export edges** r=beta_out/beta_in, `W[0,1]=W[1,0]=k*r`, і спільне beta=beta_in для micro/macro. Перевірити, що Full враховує weight. Тоді k лишається числом ребер, total export weight дорівнює k*r. Зберігати обидві величини; в усіх формулах hazard врахувати weights. Це не оптимізація або виправлення Production.

Окремо `tau=[1,0.25,0.0625]` на тих самих представниках, ratio=1. Повторювати тільки MM; Full reuse. Порівнювати розподіли та невизначеність, не вимагати pathwise збігу при різному tau. Стабільність висновків із зменшенням tau є numerical sensitivity check, але не доказ точності tau=0 limit.

Очікувана роль ratio: при дуже повільному export внутрішня структура може стати менш помітною; при швидкому export особливо важливий час до першого bridge. Це гіпотеза; не зашивати відповідь у calibration.

## 9. Дескриптори і observables

### Структура, обчислюється один раз на graph/interface

Обов'язково: n, m, realized mean degree, degree variance, source degree, sorted bridge degree multiset, k, total export weight, distance histogram s→B, D_min/mean/median/max/std, rho(A), lambda2(L_norm), lambda2(L), graph family, cut count/partition для E2, source/bridge sides, gateway-selection status.

Додатково дешево: clustering, кількість найкоротших шляхів s→b (BFS dynamic programming, зберігати log1p кількості), ASPL_est. Кількість найкоротших шляхів — predictor redundancy, а не число незалежних маршрутів.

За потреби після першого аналізу: effective resistance s→B (вузли B заземлити та розв'язати sparse Laplacian system), local cut/edge connectivity. Це альтернативні дескриптори, не SI hitting-time identities. Betweenness усіх вузлів і повний spectrum не потрібні за замовчуванням.

Eigenvectors не primary. Fiedler-вектор має sign ambiguity, а при кратному lambda2 — basis ambiguity. Не регресувати сире значення v2(s). Якщо додавати, використати sign-invariant source-to-B separation і перевірити spectral multiplicity; за виродження зберегти unavailable або projector-based descriptor з окремо визначеною нормалізацією.

Sparse matrices: `scipy.sparse.linalg.eigsh`, adjacency `which="LA"`, Laplacian smallest eigenvalues; не `which="LM"` для найбільшого algebraic adjacency eigenvalue. Для маленьких n можна dense контроль. Перевірити residuals, близький до нуля перший Laplacian eigenvalue і connectedness. Failure не замінювати фіктивним lambda2=1. Для clique використати відомі rho=n−1, lambda2(L)=n, lambda2(L_norm)=n/(n−1), n>1.

### Події, окремо для кожної моделі

- `T_export`: перше зараження z; `export_observed`.
- `T_bridge_first`: перше зараження справжнього b∈B, якщо його спостерігали.
- `bridge_infected_count_at_export_minus`: скільки справжніх B інфіковано безпосередньо перед експортом.
- `bypass`: observed export та count=0; у Full завжди false.
- `I_source_at_export_minus`, infected fraction.
- Для спостережуваних обох подій `bridge_to_export_delay=T_export-T_bridge_first`; MM значення може бути від'ємним, якщо спостереження продовжувалося після експорту.
- Якщо зупинка відбулася на експорті раніше будь-якого bridge, `T_bridge_first=null`, delay=null, **bypass=true**. Не потрібно домодельовувати майбутнє лише заради негативного delay і не підставляти 0 замість missing.
- Фактичні horizon, stop_time, stop_reason; runtime тільки технічна metadata.

Не плутати `export_observed` з «усі n+1 вузлів інфіковані». Для цього дослідження завершена ціль — перший export.

### Окремий mechanistic audit на внутрішніх траєкторіях

Для заздалегідь вибраних designs E1/E2 згенерувати або повторно використати **source-only** SI trajectories без приймача; достатньо infection time кожного вузла. На одній такій траєкторії обчислити h_Full і h_agg та інтегрувати точно по change points. При потребі згенерувати export times для обох hazards з одним Exp(1)-порогом окремого RNG. Це paired rate-closure diagnostic, **не додатковий Production simulator** і не заміна primary Full/MM.

Збережені source infection times дають точну conditional survival `exp(-H(t))` без Monte Carlo noise export clocks. Усереднювати survival по source trajectories; cumulative hazard `-log(mean survival)` не тотожний mean(H). Якщо будувати hazard серед завершених/ще не завершених production runs, потрібен risk-set conditioning; краще використати цей чітко визначений source-only audit.

Audit дозволяє перевикористати одну source trajectory для багатьох B і export weights, якщо G,s,beta_in незмінні. Це зменшить обсяг додаткових симуляцій і прямо покаже, яку інформацію втрачає I/n closure.

## 10. Метрики, незавершеність і статистика

Primary per cell: export completion обох моделей; mean/median/IQR T_export; signed mean bias `mean(T_MM)-mean(T_Full)`; relative signed bias із denominator mean Full; W1 повних observed export distributions, relative W1 з тим самим denominator; KS; bypass probability. Відсотковий W1 — нормована беззнакова різниця розподілів, не «відсоток запізнення».

CI для mean bias — hierarchical/paired bootstrap: outer resampling graph blocks, inner resampling event pairs усередині graphs. Near/far та model contrasts мають зберігати matching при resampling. Для змішаних умов з shared MM result bootstrap повинен зберегти його спільність. Детерміновані graphs — тільки event bootstrap; не приписувати їм структурну невизначеність.

Якщо є censored export, не відкидати такі пари для primary повних метрик і не підставляти T_MAX як event. Позначити unrestricted mean/W1 як unavailable; продовжити лише censored runs із тими самими seed/config до більшого horizon, зберігаючи старі checkpoints. Порівняння префікса після rerun перевірити, не обіцяти його без regression test: midpoint і обрізаний останній крок можуть залежати від horizon.

Поки tails не спостережені: completion, survival до спільного horizon, restricted RMST/bias та restricted W1 лише з явним написом restricted. Побудова plots не повинна падати через менше 100 валідних пар: warning + фактичні N_graphs/N_events/completion на рисунку. Не приховувати censoring за smooth KDE; основні distribution plots — ECDF/survival.

Індивідуальні MM−Full paired differences мають сенс за оголошеним coupling, але їхній розкид залежить від нього. Primary mean difference і marginal W1 не потребують твердження, що одна пара — це та сама епідемія.

Перед статистичними висновками окремо показати between-graph і within-graph variability. 10000 event runs на одному графі не замінюють 30 незалежних graphs.

Empirical W1 додатний навіть між двома скінченними вибірками з одного закону. Для ключових claims про малу distributional error додати Full-vs-Full split-sample reference з однаковими розмірами вибірок і збереженим graph sampling. Не віднімати цей рівень як нібито точну «корекцію» W1; використати як орієнтир Monte Carlo resolution.

## 11. Перевірка «який фактор пояснює більше»

Первинна доказова база — E0/E1/E2 contrasts, не таблиця pooled correlations. Потім, якщо вистачає незалежних graph blocks, виконати просте secondary predictive comparison.

Два окремі targets на cell/network рівні: `log(mean Full T_export)` і `relative mean bias`. Використати linear/ridge regression без великого hyperparameter search:

- M0: log(n), d_realized, log(k), log(beta_out/beta_in), degree_source, mean_degree_B.
- M_path: M0 + D_min, D_mean, distance_std, log shortest-path count.
- M_spectrum: M0 + log(lambda2_norm), rho(A).
- M_both: об'єднання path і spectrum features.

Нульової variance features (наприклад rho=6 на regular subset) не оцінювати. Не дублювати k/n поряд із усіма залежними transforms без контролю collinearity. E0 аналітичні контроли, clique і випадкові graphs не зливати в одну модель без stratification. Хоча б показати результати по topology family.

Cross-validation split робити **за base graph_id**: всі placements, rewires, event repeats і похідні matches від одного базового graph pair залишаються в одному fold. Standardization/feature selection лише на training fold. Метрика: held-out MAE з graph-bootstrap CI; за достатніх topology families додатково leave-family-out validation. Не використовувати observed T_bridge чи q_B як «передсимуляційний predictor»; це mechanistic outcomes.

Якщо M_both не дає відтворюваного покращення над M_path, немає підстав вводити spectral correction. Якщо дає — це predictive evidence у дослідженому класі, а не доказ, що lambda2 спричиняє всю затримку. Якщо sample надто малий, залишити contrasts і не будувати нестабільну багатофакторну регресію.

## 12. Рисунки: три основні, решта в додаток

### Figure 1 — Interface location at fixed internal spectrum (E1)

2×2 panels, один topology family/набір параметрів, вибраний до перегляду epidemic outcomes (default random_regular n100,d6,k5; якщо near/far unattainable — заздалегідь визначений modular_cut4).

A. Невелика схема source + sink і near/far виходів; ніякого chain. B. Full export ECDF для near/middle/far та одна спільна MM ECDF, якщо invariant підтверджений. C. Graph-paired mean Full T_export проти фактичної D_mean з graph-bootstrap CI; MM як відповідна спільна reference. D. Relative signed bias та bypass rate по placement (окремі підпанелі/осі без подвійної y-axis).

Висновок, який можна перевірити: однаковий spectrum не гарантує однакового export law; MM може не бачити interface rearrangement. Не вважати near/far абсолютною універсальною монотонністю для кожного випадкового graph.

### Figure 2 — Distance is not sufficient; spectrum is not sufficient (E0 + E2)

Максимум 4 panels. A. E0a: Full ECDF на трьох side topologies при ell=4 + Gamma(5,beta) reference; окремо компактно показати фактичні normalized gaps. B. E0b: mean/ECDF export для q=1,2,4 при distance=4. C. E2: mean Full export vs lambda2_norm при exact matched distance histogram; кольори same_side/opposite_side, лінії лише всередині matched graph family. D. Відповідний MM signed bias.

Якщо E2 matching не забезпечує достатніх даних, Figure 2 обмежити E0 та чесно віднести E2 у exploratory appendix. Не заповнювати порожні cells інтерполяцією. Позначати local degree confound у E0b і global degree differences у E0a.

### Figure 3 — Why the aggregate closure misses the export (source-only audit)

Для двох preselected representative designs near/far показати: A/B. E[I(t)/n] та E[infected_bridge_fraction(t)] на спільній абсолютній осі часу. C/D. Conditional-survival averages для exact boundary hazard та aggregate hazard, обчислені на тих самих source trajectories; за наявності накласти окремо primary Full/MM ECDF для контексту.

Bands для середніх — graph-bootstrap CI, не IQR trajectories. Не трактувати difference of averaged fractions як прямий доказ stochastic ordering. Саме integrated-hazard/survival panels перевіряють наслідок closure.

### Appendix / Supplement

- E3: mean Full export і relative MM bias проти n при exact D=2, facets за k; окремо normalized gap, completion і досягнутий matching.
- E4: ratio sensitivity та tau convergence diagnostics, фактичні N на panels.
- Всі topologies/placements/k, без наддовгої unreadable матриці в основному тексті.
- Scatter predictor comparisons/CV MAE — тільки якщо достатня вибірка.
- Таблиця graph descriptors, matching feasibility та censoring.
- За потреби negative MM bridge-to-export delay — лише коли обидві події справді збережені.

Загальні правила: Full/MM сталі кольори; near→middle→far і numeric k/n у зростаючому порядку; спільна нормалізація relative metrics; без подвійного змісту bands; не згладжувати ECDF. Plot captions повинні пояснювати mechanism, sampling unit, uncertainty, horizon і межі висновку. Нові PDF+PNG для рукопису копіювати в `paper/SMPT paper/Images/` лише після review; оригінали аналізу тримати окремо.

## 13. Конкретна реалізація для Luna

### 13.1. Файли та перевикористання

Підготувати, не запускаючи автоматично великий grid:

```text
extrascripts/runners/run_single_community_export_independent_rng.py
extrascripts/runners/single_community_export_design.py
extrascripts/analysis/analyze_single_community_export.py
extrascripts/visualization/plot_single_community_export.py
extrascripts/validation/check_single_community_export.py
```

Editable config — на початку runner-а, поруч із ним можливий JSON override. `single_community_export_design.py` містить лише deterministic graph/interface construction і descriptors; не новий epidemic engine. За малої реалізації helper можна залишити в runner-і, не розносити по багатьох модулях без потреби.

Перевірені поточні опорні місця:

- `extrascripts/runners/run_original_benchmarks_independent_rng.py`: `_rng_ownership_patch`, `_patched_clone`, `_patched_refresh`, `_run_experimental_micromacro`, `_regression_gate`, `Benchmark`.
- `_run_experimental_micromacro` задає `model.rng=random.Random(event_seed+10000+1009*community)` і копіює/синхронізує стан clone. Scheduler не переписувати.
- `two_layer_ctmc/engines/macro_engine.py`: `_compute_hazards_matrix`; `alpha_mode="off"`, `correction_mode="off"` — defaults.
- `two_layer_ctmc/orchestrators/micromacro.py`: підтримує uneven sizes і `dense_logs=False`, `stop_at_absorption`; передавати явні `alphas=[1.0,1.0]`, інакше `_compute_alphas` може непотрібно запускати dense eigensolver.
- `two_layer_ctmc/simulators/micro.py`, `two_layer_ctmc/orchestrators/micro.py`: uneven `comm_nodes`, `community_timings`, events/transmissions; перевірити seed-at-time-zero в infection records.
- `extrascripts/runners/checkpoint_io.py`: atomic compact gzip checkpoints.

Перед використанням reread helper signatures; не припускати, що `stop_at_absorption=True` означає first export — зараз це різні умови. Найпростіша коректна v1 може виконати production до absorption/horizon і витягнути prefix до first export. Це зайва робота, але не зміна law першого експорту. Для важких side paths бажано runner-local observer/adapter, що перериває **після committed infection приймача**, а не після speculative midpoint probe. Observer не повинен робити RNG draws. Production `.py` файли не змінювати.

Перед увімкненням early-stop перевірити точний збіг event prefixes/export time з unmodified run на reusable fixtures. Якщо безпечного hook немає, не копіювати весь orchestrator: залишити v1 absorption fallback і повідомляти policy. На диску зберігати тільки потрібний prefix; fake padding до T_MAX заборонено.

Graph source nodes `0..n-1`, sink `n`; `micro_graphs=[G_source,G_sink]`, `comm_nodes=[list(range(n)),[n]]`, W offdiagonal k*r, diagonal zero. Одиничні internal weights. `full_graph.graph` metadata мають бути explicit та узгоджені; production graph construction не мутує immutable cached networks.

До запуску надрукувати кількість feasible cells, independent graphs, event pairs, reused/new і оцінку загальної роботи з pilot runtimes. У процесі — block/topology/n/k/placement/graph/event, progress, remaining. Pool — multiprocessing, не threads, бо legacy macro RNG і monkeypatch process-global. Обмеження workers/heavy_workers на початку коду; один worker виконує одну пару/блок послідовно.

### 13.2. Структура даних і resume

```text
output/data/
  export_single_<topology>_n100_d6_k5_bin0p003_bout0p003_tau1_<placement>_independent_rng_<designhash>/
    manifest.json
    networks/<network_id>.json.gz
    checkpoints/g0000_e0000.json.gz
    ...

output/analysis/single_community_export/
  inventory.csv
  graph_descriptors.csv
  design_feasibility.csv
  export_observables.csv
  cell_metrics.csv
  matched_contrasts.csv
  hazard_audit.csv.gz
  prediction_validation.csv       # тільки якщо виконано
  analysis_manifest.json
  figures/*.pdf, *.png
  README.md
```

Graph payload: sorted nodes/edges/weights, B, s, graph construction parameters, seeds, source-graph hash, full-task hash. Зберігати граф один раз, не кожної реплікації; спільний graph cache/reference дозволений у `output/data/_network_cache/`, якщо reference перевіряється checksum. Не створювати порожні trajectories.csv заглушки.

Checkpoint обох моделей: exact scientific config, seed map, graph/interface hash, node infection times до stop (або компактні infection events), export/bridge observables, horizon/termination, algorithm/RNG version, content checksum. Для SI час зараження кожного вузла достатній, щоб відновити I(t) та infected fraction B до фактичного stop; це значно компактніше за кожен регулярний timestep. Зберегти missing reason для незаражених/неспостережених вузлів.

Формат імен не містить T_MAX або requested N. Усі параметри, не вміщені в коротке ім'я, входять у manifest і designhash. Reuse визначає **manifest+hash**, а не збіг folder name. Новий exp не можна переплутати із chain data.

Scientific designhash виключає requested horizon, requested sample counts, workers і output/reporting settings; вони зберігаються як execution metadata. Окремий network hash включає фактичні ребра, ваги, s і interface, а simulation identity — цей network hash, algorithm settings і фактично використані event seeds. Це дозволяє додавати runs/подовжувати horizon без створення іншої scientific cell та водночас не плутати реалізації мережі.

Якщо export уже observed, такого checkpoint достатньо для export metrics навіть при старому меншому horizon. Для censored runs із недостатнім horizon rerun лише їх, зі збереженими seeds; перед заміною atomic checkpoint зберегти попередню версію. Requested run count не змінює scientific identity. При reuse старий фактичний seed зберігається. Спочатку сформувати read-only execution plan, потім запускати лише missing tasks; не виконувати smoke повторно на resume, якщо code/config validation signature чинний.

Повна CSV-консолідація trajectories не потрібна. Analysis/plots читають checkpoints; маленькі таблиці observables/metrics можна матеріалізувати з cache keyed by source checksums. Немає повторних симуляцій під час plotting.

### 13.3. Валідація перед handoff

1. Topology-only: sizes `[n,1]`, connected source, k distinct bridges, s∉B, W=sum export weights, degree/cut/distance invariants, exact adjacency hashes для E1.
2. Production regression gate: experimental shared mode pathwise збігається з original; independent clone/refresh RNG tests проходять. Не вимагати збігу independent і shared paths.
3. Full: bypass=0; export не раніше first bridge. MM: bypass дозволений, correctly labeled.
4. E0a: Gamma reference з правильною параметризацією rate/scale; analytical CDF/mean check з наперед заданим Monte Carlo tolerance, не «кожен p-value має бути >0.05».
5. E1: MM first-export invariant за near/middle/far при тих самих internal graph і seeds. Якщо порушений — зупинити цей блок, знайти metadata/correction/RNG залежність до science run.
6. E2: degree=6 і rho≈6; primary matching histogram справді однаковий. Solver residuals та tolerance записані.
7. k1 source b initially infected контроль: Full export Exp(beta_out), щоб перевірити weights; це окремий test case, не primary s∉B data.
8. stop observer/fallback: однаковий prefix при спільному horizon/seed; compact logging не змінює події.
9. Resume: переривання після кількох fixtures, повторний запуск не змінює готові bytes/seed/results; censored horizon extension не відкидає completed pairs.
10. Plot smoke на saved маленьких fixtures, включно з missing/censored runs, правильні labels, N і signed axes. Тести повторно використовують fixture files; великі networks не генеруються кожного разу.

Очікувані команди **після майбутньої реалізації**, наразі цих нових entry points ще немає:

```bash
.venv/bin/python3 extrascripts/runners/run_single_community_export_independent_rng.py --topology-only
.venv/bin/python3 extrascripts/validation/check_single_community_export.py
.venv/bin/python3 extrascripts/runners/run_single_community_export_independent_rng.py --smoke-only
.venv/bin/python3 extrascripts/runners/run_single_community_export_independent_rng.py --resume
.venv/bin/python3 extrascripts/analysis/analyze_single_community_export.py
.venv/bin/python3 extrascripts/visualization/plot_single_community_export.py
```

`--smoke-only`: 2 feasible graphs × 2 events на selected block, без прихованого повного grid. Default main profile — E0/E1/E2 exploratory config; E3/E4 через `--blocks`/явний config. Перший implementation handoff містить робочі scripts, validation report, feasibility table і одну команду запуску для користувача; великий запуск не починати як побічну дію написання коду.

## 14. Що можна використати з наявного і що не можна

- Наявний clique–bridge–leaf benchmark може лишитися ілюстрацією export distribution vs bridge bypass у статті; не треба перемальовувати старий рисунок лише через цей план. Reuse конкретних даних для нового контрольованого дизайну можливий тільки при exact config/initial/gateway identity.
- `extrascripts/runners/run_gateway_distance_sweep.py` викликає звичайний `MicroMacroSimulator`, без independent-RNG intervention. Дані `output/missing_experiments/gateway_distance/full` не є новою independent-RNG evidence; звідти можна запозичити лише прийоми graph construction після перевірки.
- `run_gateway_distance_by_size_independent_rng.py` — попередній L=5 chain дизайн із k×k target D_mean=2.72. Він не відповідає поточному one-source+sink дизайну; не «доробляти» його як нібито той самий експеримент.
- Старі partial Dmean2p72 дані перенесені до `output/backup/abandoned_gateway_distance_Dmean2p72_20260918/`. Не рухати й не перезаписувати їх у межах реалізації цього плану.
- Existing independent chain grid підходить для зовнішньої перевірки згодом, але не дозволяє автоматично відокремити internal export від repeated imports та upstream timing.

## 15. Критерій наукового результату

У статті має з'явитися відповідь такого типу, **якщо її підтримають дані**: точність aggregate export визначається тим, наскільки загальна зараженість репрезентує стан вихідного інтерфейсу. Шляхи, їхня множинність, bottlenecks і розмір впливають на цю відповідність. Глобальний eigenvalue може бути корисним descriptor, але не містить сам по собі інформації про розташування входу і виходу.

Не писати наперед, що distance «перемогла» spectrum чи навпаки. Повідомити окремо exact analytical controls, спостережувані contrasts із CI, predictive performance та невизначені/недоступні comparisons. Не пропонувати spectral correction до Production лише на підставі однієї кореляції.

Порядок реалізації: **E0 + topology feasibility → E1 → E2 matched subset → mechanistic hazard audit → перші 3 figures → E3 size control → E4 sensitivity за потреби**. Це дає змістовний результат до розгортання великої багатопараметричної сітки.
