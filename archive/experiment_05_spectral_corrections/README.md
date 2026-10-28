# Архів Experiment 05: spectral corrections

Дата архівування: 2026-08-07 UTC.

Дані переміщено з `output/experiment_05_spectral_corrections/`, щоб велика
похідна діагностика та реплікації поза фінальним статистичним вікном не
займали робочий каталог і не змішувалися з фінальною вибіркою.

Фінальний звіт Experiment 05 використовує рівно replication IDs 1–100 у
кожній із 25 scenario–variant комірок. Ці файли залишено в `output/`.

## Вміст

| Архівний шлях | Початковий шлях | Кількість / розмір | Причина |
|---|---|---:|---|
| `summaries/hazard_diagnostics.csv` | `output/experiment_05_spectral_corrections/summaries/hazard_diagnostics.csv` | 41,920,540,193 bytes | Нестиснена похідна CSV-копія hazard-рядків із raw Parquet; фінальний звіт її не читає |
| `raw/baseline/V0/rep_0101.parquet` … `rep_0646.parquet` | `output/experiment_05_spectral_corrections/raw/baseline/V0/` | 546 файлів; 27,853,315 bytes | Реплікації поза фінальним вікном 1–100 |
| `raw/baseline/V1/rep_0101.parquet` … `rep_0645.parquet` | `output/experiment_05_spectral_corrections/raw/baseline/V1/` | 545 файлів; 236,620,751 bytes | Реплікації поза фінальним вікном 1–100 |
| `raw/baseline/V2/rep_0101.parquet` … `rep_0645.parquet` | `output/experiment_05_spectral_corrections/raw/baseline/V2/` | 545 файлів; 183,097,047 bytes | Реплікації поза фінальним вікном 1–100 |
| `raw/baseline/V3/rep_0101.parquet` … `rep_0645.parquet` | `output/experiment_05_spectral_corrections/raw/baseline/V3/` | 545 файлів; 404,084,535 bytes | Реплікації поза фінальним вікном 1–100 |
| `raw/baseline/V4/rep_0101.parquet` … `rep_0645.parquet` | `output/experiment_05_spectral_corrections/raw/baseline/V4/` | 545 файлів; 268,619,839 bytes | Реплікації поза фінальним вікном 1–100 |

Разом: один CSV і 2726 Parquet-файлів, 43,040,815,680 bytes.

## Що залишено в робочому output

- усі 25 комірок із replication IDs 1–100;
- компактні `replication_level.csv`, `arrivals_long.csv`, terminal/layer/paired summaries;
- configs, seeds, checkpoints, manifests, figures і provenance;
- raw Parquet для фінальної вибірки 1–100.

## Як відновити

Запускати з кореня репозиторію:

```bash
mv "архів/experiment_05_spectral_corrections/summaries/hazard_diagnostics.csv" \
  output/experiment_05_spectral_corrections/summaries/

for variant in V0 V1 V2 V3 V4; do
  mv "архів/experiment_05_spectral_corrections/raw/baseline/$variant"/rep_*.parquet \
    "output/experiment_05_spectral_corrections/raw/baseline/$variant"/
done
```

Після відновлення не включайте IDs понад 100 у фінальний аналіз без нового
наперед визначеного плану. `hazard_diagnostics.csv` також можна повторно
згенерувати з raw Parquet функцією `aggregate_hazard_diagnostics` у
`two_layer_ctmc/experiments/experiment_05_spectral_corrections/runner.py`.
