# Стан врахування рекомендацій

## Враховано

- Додано формальний full-network CTMC reference model із генератором, відносно якого визначається похибка двошарового симулятора.
- Переписано базову нотацію: множини \(S_i,I_i,R_i\), лічильники \(s_i,i_i,r_i\), частки \(\bar{s}_i,\bar{i}_i,\bar{r}_i\) розділені явно.
- Розділено \(\beta_{\mathrm{in}}\) для внутрішньоспільнотної передачі та \(\beta_{\mathrm{out}}\) для міжспільнотної передачі в основних модельних формулах.
- Макро-події описано не як звичайний NHPP з детермінованою інтенсивністю, а як процес із стохастичною передбачуваною інтенсивністю; умовно на micro paths використано cumulative-hazard construction.
- Додано import placement rule \(\Pi_{ij}\) і пояснено, що вибір цільового вузла є частиною визначення симулятора.
- Додано декомпозицію похибки: rate abstraction error, import placement error, synchronization/numerical timing error.
- Додано твердження про conditional micro-layer exactness: локальний CTMC є точним за фіксованого процесу імпортів.
- Додано критерій rate abstraction: макро-інтенсивність не може бути точною, якщо однакові агрегати відповідають різним boundary/bridge конфігураціям.
- Додано критерій правильного import placement через edge-conditioned target distribution.
- Додано bound для synchronization timing error через похибку cumulative hazard.
- Значною мірою враховано clique--bridge--leaf benchmark: bridge-dependent hazard, two-phase structure, bridge-finding time, mean-field hazard, Jensen survival inequality, early-export bias.
- Додано chain-of-cliques benchmark із arrival-time decomposition, accumulation of local timing shifts і distributional metrics.
- Suggested abstract із рекомендацій фактично перенесено в abstract рукопису.

## Частково враховано

- Масштабування \(W_{ij}\) згадано, але ще не визначено достатньо однозначно для всіх випадків: треба чітко сказати, чи це кількість cross edges, mobility weight, \(N_iN_j\)-масштабування, або калібрований коефіцієнт.
- Bridge-finding formula наведена для uniform seeding, але ще не всюди явно відокремлено випадок fixed non-bridge seed; у chain-of-cliques місцями формула використовується без достатньої умови про початкове зараження.
- Mean-field conditional-mean argument додано, але припущення exchangeability треба сформулювати жорсткіше в самому твердженні.
- Numerical section має графіки та первинну інтерпретацію, але ще не має повної таблиці параметрів, Monte Carlo protocol, seed policy, stopping criterion і повної таблиці метрик.
- Distributional metrics \(W_1\), \(D_{KS}\), mean/quantile shifts описані, але ще не подані як фактичні результати.
- Runtime/speedup згадано як пункт для завершення, але вимірювання ще не додані.
- Абляційні експерименти запропоновані в рекомендаціях, але в тексті ще не реалізовані як окремий experimental design.

## Не враховано або ще треба виправити

- У рукописі залишилися collaboration/read-status boxes.
- Залишилася секція `--------------In Progress---------------`.
- Залишився великий блок `Other notes, ideas, generated sections`, який дублює вже оформлені частини й не має бути в journal submission.
- Потрібно прибрати дублікати clique-to-leaf / clique--bridge--leaf і chain-of-cliques секцій або перенести зайве в appendix.
- Потрібно додати pseudocode для повного двошарового симулятора, micro update, macro hazard accumulation, event selection та import placement.
- Потрібні редакційні правки: `data avaliability` -> `data availability`, `Coefficients` spelling, `There is inverse approach` -> `An alternative representation is`, уніфікація `within-community` / `between-community`.
- Треба уникати формулювання, що macro layer дає "exact timing" без уточнення: точність стосується лише approximate macro intensity, не повного microscopic reference model.

## Короткий висновок

Основна математична лінія рекомендацій уже перенесена в текст: reference CTMC, двошарова архітектура, джерела похибки, bridge benchmark, Jensen bias і chain benchmark. Найбільші залишкові задачі стосуються не нової теорії, а підготовки рукопису до журнального формату: прибрати службові/дубльовані секції, зафіксувати припущення, додати reproducible numerical protocol, метрики, runtime/speedup і pseudocode.
