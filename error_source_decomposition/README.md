# Error source decomposition — 2026-08-14

Цей пакет містить відтворюване дослідження джерел похибки двомасштабного CTMC-симулятора.

## Структура
- `reports/` — повний україномовний звіт (`REPORT.md`) і методика (`METHODS.md`).
- `data/MASTER_RESULTS.csv` — головна таблиця чисельних висновків.
- `runners/` — усі нові runner-и/аналізатори та `RUNNER_HASHES.json`.
- `figures/` — 4 ключові діагностичні графіки.
- `structural_factorial/` — структурна декомпозиція, held-out calibration, gateway calibration bound.
- `production_factorial/` — 100-run independent-RNG experiment, mediation/sensitivity runs, RNG unit test, tau sensitivity.
- `error_source_decomposition_2026-08-14.zip` — повна копія локального пакета з усіма per-rep/part файлами.

## Короткий результат
1. У current Production приблизно 82.5% signed bias відносно Full припадає на implementation gap, 17.5% — на event-driven structural gap (один baseline).
2. 100-run intervention з незалежними RNG streams практично повністю прибирає implementation gap: correction +1051.9 RMST; residual до event-driven −13.5 RMST, 95% ДІ включає 0.
3. Повторні імпорти є сильним медіатором RNG-дефекту, але не незалежним структурним джерелом.
4. При nominal calibration c=1 структурний directional gap A_U→E_E = 228.3 RMST; signed attribution: source boundary +44.8%, destination −17.2%, edge-pair −1.5%, placement +73.9%.
5. Held-out scalar calibration у fixed baseline майже прибирає signed RMST bias, але не може усунути приховану gateway-architecture dependence; optimistic minimax lower bound near↔far ≈264.9 RMST.

Деталі, довірчі інтервали, застереження щодо інтерпретації відсотків і причинна ієрархія наведені в `reports/REPORT.md`.
