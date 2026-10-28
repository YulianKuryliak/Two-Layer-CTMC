"""Small isolated experiment: residual hazard versus redraw after one import.

An independent import time is Exp(mu). A local counting CTMC has rate lambda0
before that import and lambda1 afterwards. Completion means m local events.
No project simulator or scientific baseline is changed.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, kstest


def clocks(budgets, imports, fresh, rate0, rate1):
    """Continue the interrupted total-local clock; local events reset its budget."""
    old_times = np.cumsum(budgets, axis=1) / rate0
    total = old_times[:, -1]
    completed = (old_times <= imports[:, None]).sum(axis=1)
    active = completed < budgets.shape[1]
    rows = np.flatnonzero(active)
    stage = completed[active]
    previous = np.where(stage > 0, old_times[rows, np.maximum(stage - 1, 0)], 0.0)
    remaining = budgets[rows, stage] - rate0 * (imports[active] - previous)
    if np.any(remaining < -1e-12):
        raise AssertionError("Negative residual hazard")
    tail = budgets[rows, stage] + rate0 * (total[active] - old_times[rows, stage])
    later_budgets = tail - budgets[rows, stage]
    residual = total.copy()
    redraw = total.copy()
    residual[active] = imports[active] + (remaining + later_budgets) / rate1
    redraw[active] = imports[active] + (fresh[active] + later_budgets) / rate1
    return residual, redraw, active, old_times


def exact_mean(events, rate0, rate1, mu):
    # Race between a local event and the one external import; after the import
    # all remaining events have constant rate rate1.
    mean = 0.0
    for remaining in range(1, events + 1):
        mean = (1 + rate0 * mean + mu * remaining / rate1) / (rate0 + mu)
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, default=Path("output/import_clock_policy_toy"))
    args = parser.parse_args()
    if args.runs < 100:
        parser.error("--runs must be at least 100")
    args.output.mkdir(parents=True, exist_ok=True)
    streams = iter(np.random.SeedSequence(args.seed).spawn(20))
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    rate0, mu = 1.0, 1.0
    for row, events in enumerate((1, 5)):
        for col, rate1 in enumerate((1.0, 2.0, 0.5)):
            samples = []
            for policy in ("residual", "redraw"):
                rng = np.random.default_rng(next(streams))
                budgets = rng.exponential(size=(args.runs, events))
                imports = rng.exponential(1 / mu, args.runs)
                fresh = rng.exponential(size=args.runs)
                residual, redraw, active, old = clocks(budgets, imports, fresh, rate0, rate1)
                if rate1 == rate0 and not np.allclose(residual, old[:, -1], rtol=1e-12, atol=1e-12):
                    raise AssertionError("Residual clock changed a path with unchanged rate")
                samples.append(residual if policy == "residual" else redraw)
            a, b = samples
            diff = b.mean() - a.mean()
            se = math.sqrt(a.var(ddof=1) / args.runs + b.var(ddof=1) / args.runs)
            ks = ks_2samp(a, b, method="asymp")
            exact = exact_mean(events, rate0, rate1, mu)
            rec = dict(events=events, rate_before=rate0, rate_after=rate1, import_rate=mu,
                       runs_per_policy=args.runs, exact_mean=exact,
                       mean_residual=float(a.mean()), mean_redraw=float(b.mean()),
                       diff_redraw_minus_residual=float(diff), ci95_low=float(diff - 1.96 * se),
                       ci95_high=float(diff + 1.96 * se), ks_D=float(ks.statistic), ks_p=float(ks.pvalue))
            for name, values in zip(("residual", "redraw"), samples):
                std_error = values.std(ddof=1) / math.sqrt(args.runs)
                rec[f"mean_z_vs_exact_{name}"] = float((values.mean() - exact) / std_error)
                rec[f"q50_{name}"] = float(np.quantile(values, 0.5))
                rec[f"q95_{name}"] = float(np.quantile(values, 0.95))
                rec[f"ks_vs_exact_p_{name}"] = (
                    float(kstest(values, "expon").pvalue)
                    if events == 1 and rate1 == rate0 else None
                )
            records.append(rec)
            ax = axes[row, col]
            for label, values, style in zip(("Residual budget", "Redraw"), samples, ("-", "--")):
                sorted_values = np.sort(values)
                idx = np.unique(np.linspace(0, args.runs - 1, 1200).astype(int))
                ax.plot(sorted_values[idx], (idx + 1) / args.runs, style, label=label, lw=1.8)
            ax.set(xlim=(0, max(np.quantile(a, .99), np.quantile(b, .99))), ylim=(0, 1),
                   title=f"{events} local event(s), rate {rate0:g} -> {rate1:g}",
                   xlabel="Completion time", ylabel="Empirical CDF")
            ax.grid(alpha=.25)
            if row == 0 and col == 0:
                ax.legend()
    fig.suptitle(f"One independent random import; {args.runs:,} independent runs per policy")
    fig.savefig(args.output / "completion_cdfs.png", dpi=180)
    fig.savefig(args.output / "completion_cdfs.pdf")
    plt.close(fig)

    # Coupled trajectories illustrate changes to individual outcomes, not an
    # independent-sample distribution test. Both policies share pre-import noise.
    rng = np.random.default_rng(next(streams))
    budgets = rng.exponential(size=(args.runs, 1))
    imports = rng.exponential(size=args.runs)
    fresh = rng.exponential(size=args.runs)
    residual, redraw, active, old = clocks(budgets, imports, fresh, 1.0, 1.0)
    paired_diff = redraw - residual
    paired_se = paired_diff.std(ddof=1) / math.sqrt(args.runs)
    paired = dict(runs=args.runs, import_before_local_fraction=float(active.mean()),
                  mean_diff=float(paired_diff.mean()),
                  ci95_low=float(paired_diff.mean() - 1.96 * paired_se),
                  ci95_high=float(paired_diff.mean() + 1.96 * paired_se),
                  mean_absolute_path_difference=float(np.abs(paired_diff).mean()),
                  redraw_later_fraction=float((paired_diff > 0).mean()),
                  redraw_earlier_fraction=float((paired_diff < 0).mean()))
    examples = np.flatnonzero(active)[:12]
    import csv
    with (args.output / "trajectory_examples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "old_local_time", "import_time", "residual_time", "redraw_time"))
        for i in examples:
            writer.writerow((int(i), old[i, 0], imports[i], residual[i], redraw[i]))

    # Deterministic mean-wait control: memorylessness does not apply to a wait 1.
    deterministic_residual = np.ones(args.runs)
    deterministic_redraw = np.where(imports < 1, imports + 1, 1)
    deterministic = dict(mean_residual=float(deterministic_residual.mean()),
                         mean_redraw=float(deterministic_redraw.mean()),
                         exact_mean_redraw=float(2 - 2 / math.e))

    # The initial run had an unexpected KS signal for the unchanged-rate case.
    # Keep that run and inspect 20 independent replication seeds, without
    # selecting or replacing the original result by a favorable seed.
    robustness = []
    for seed in range(args.seed + 1, args.seed + 21):
        children = np.random.SeedSequence(seed).spawn(2)
        independent_samples = []
        for policy, child in zip(("residual", "redraw"), children):
            rg = np.random.default_rng(child)
            count = 50_000
            r, d, _, _ = clocks(rg.exponential(size=(count, 1)),
                                rg.exponential(size=count), rg.exponential(size=count), 1.0, 1.0)
            independent_samples.append(r if policy == "residual" else d)
        a, b = independent_samples
        test = ks_2samp(a, b, method="asymp")
        robustness.append(dict(seed=seed, runs_per_policy=count, ks_D=float(test.statistic),
                               ks_p=float(test.pvalue), mean_diff=float(b.mean() - a.mean())))
    robustness_summary = dict(repetitions=20, runs_per_policy_per_repetition=50_000,
                              seed_range=[args.seed + 1, args.seed + 20],
                              ks_rejections_at_005=sum(r["ks_p"] < .05 for r in robustness),
                              mean_of_mean_differences=float(np.mean([r["mean_diff"] for r in robustness])),
                              repetitions_data=robustness)
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(seed=args.seed, runs_per_policy=args.runs, scenarios=records,
                    paired_same_rate=paired, deterministic_control=deterministic,
                    unchanged_rate_independent_replications=robustness_summary,
                    assumptions="One import Exp(1), independent of local random budgets; exact piecewise-constant rates. No midpoint probes, RNG rollback, placement, or network approximation.",
                    completion="finished", script=str(Path(__file__).resolve()))
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Експеримент: локальний годинник після випадкового імпорту", "",
             "Одна зовнішня подія має час Exp(1), незалежний від локального процесу. До імпорту локальна інтенсивність 1; після імпорту — 1, 2 або 0.5. Порівнюємо час першої локальної події та час завершення п'яти локальних подій.", "",
             f"На кожну комірку й кожен спосіб: {args.runs:,} незалежних запусків. Seed: {args.seed}. Довірчі інтервали нижче — 95% для різниці середніх; KS застосовано до незалежних вибірок.", "",
             "| Локальних подій | Інтенсивність після імпорту | Точне середнє | Залишковий бюджет | Новий відлік | Різниця та 95% ДІ | KS p |",
             "|---:|---:|---:|---:|---:|---|---:|"]
    for r in records:
        lines.append(f"| {r['events']} | {r['rate_after']:g} | {r['exact_mean']:.6f} | {r['mean_residual']:.6f} | {r['mean_redraw']:.6f} | {r['diff_redraw_minus_residual']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['ks_p']:.4f} |")
    lines += ["", "Парні траєкторії за незмінної інтенсивності:", "",
              f"- Імпорт випередив локальну подію у {100 * paired['import_before_local_fraction']:.2f}% запусків.",
              f"- Новий відлік дав пізніший результат у {100 * paired['redraw_later_fraction']:.2f}% усіх запусків, раніший — у {100 * paired['redraw_earlier_fraction']:.2f}%.",
              f"- Середня парна різниця: {paired['mean_diff']:+.6f}; 95% ДІ [{paired['ci95_low']:+.6f}, {paired['ci95_high']:+.6f}]. Середня абсолютна різниця окремих траєкторій: {paired['mean_absolute_path_difference']:.6f}.", "",
              f"Контроль із фіксованим очікуванням 1: залишковий відлік має середнє {deterministic['mean_residual']:.6f}; перезапуск — {deterministic['mean_redraw']:.6f}; точне значення для перезапуску {deterministic['exact_mean_redraw']:.6f}.", "",
              f"У початковому порівнянні незмінної інтенсивності KS p={records[0]['ks_p']:.6f}. Цей сигнал збережено. Одновибіркові KS проти точного Exp(1): p={records[0]['ks_vs_exact_p_residual']:.6f} для залишкового бюджету, p={records[0]['ks_vs_exact_p_redraw']:.6f} для нового відліку. Для шести початкових порівнянь найменше p після поправки Bonferroni становить {min(1, 6 * min(r['ks_p'] for r in records)):.6f}.", "",
              f"Після цього сигналу виконано 20 додаткових незалежних повторень лише випадку незмінної інтенсивності, по 50 000 запусків на спосіб. KS відхилив рівність на рівні 0.05 у {robustness_summary['ks_rejections_at_005']} із 20 повторень (очікувана кількість за рівності — 1). Середня різниця середніх у повтореннях: {robustness_summary['mean_of_mean_differences']:+.6f}. Усі seeds і p збережено в manifest.json; жоден результат не відкидався.", "",
              "Скінченний експеримент оцінює різницю з похибкою Monte Carlo; незаперечення KS не доводить рівність. Для цієї моделі рівність стохастичних розподілів випливає також із відсутності пам'яті Exp(1). Детермінований контроль має інший закон очікування.", "",
              "Це ізольований приклад. Він не перевіряє сітковий макропланувальник, прогнозні копії, rollback RNG або розміщення імпорту в основному симуляторі.", "",
              "Відтворення:", "", "```bash",
              f".venv/bin/python extrascripts/validation/compare_import_clock_policies.py --runs {args.runs} --seed {args.seed} --output {args.output}", "```", ""]
    (args.output / "report_uk.md").write_text("\n".join(lines))
    print(json.dumps(dict(scenarios=records, paired=paired, deterministic=deterministic,
                         robustness={k: v for k, v in robustness_summary.items() if k != "repetitions_data"}), indent=2))


if __name__ == "__main__":
    main()
