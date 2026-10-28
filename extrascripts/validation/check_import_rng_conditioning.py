"""Read-only diagnostic of the real MicroModel's RNG reuse at an import.

Path 0--1--2, beta=1, gamma=0, initial infected node 0. Condition on no local
event before a fixed import at log(4/3). Uniform placement must select each of
nodes 1 and 2 with probability 1/2. Subsequent rates are respectively 1 and 2,
so the correct mean waiting time is 3/4. This is a diagnostic, not a patch.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from two_layer_ctmc.engines.micro_model import MicroModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=710_000)
    parser.add_argument("--output", type=Path, default=ROOT / "output/import_rng_conditioning")
    args = parser.parse_args()
    time = math.log(4 / 3)
    results = {}
    for policy in ("current_rollback", "consume_pending_draw"):
        counts = {1: 0, 2: 0}
        waits = []
        for seed in range(args.seed, args.seed + args.runs):
            model = MicroModel(1.0, 0.0, rng=random.Random(seed))
            for node in range(3):
                model.add_node(node)
            model.add_edge(0, 1, 1.0)
            model.add_edge(1, 2, 1.0)
            model._infect_node(0)
            _, _, _, events = model.simulate_until(time)
            if events:
                continue
            if policy == "consume_pending_draw":
                # Safe for this isolated, privately owned RNG. This line alone
                # is not a production fix for shared streams or clone handling.
                model.rng.setstate(model._cached_rng_after)
            model.current_time = time
            target = model.import_infection()
            counts[target] += 1
            wait, _, _, _ = model._ensure_next_event()
            waits.append(wait)
        values = np.asarray(waits)
        n = len(values)
        fraction = counts[2] / n
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n))
        results[policy] = dict(conditioned_runs=n, import_node_counts=counts,
                               fraction_node2=fraction, mean_next_wait=mean,
                               mean_next_wait_ci95=[mean - 1.96 * se, mean + 1.96 * se])
    manifest = dict(runs=args.runs, seed_start=args.seed,
                    seed_end=args.seed + args.runs - 1, import_time=time,
                    expected_correct_node2=0.5, expected_correct_wait=0.75,
                    results=results, python=sys.version,
                    completion="finished", source_engine="two_layer_ctmc/engines/micro_model.py",
                    limitation="Isolated actual local engine; no macro midpoint scheduler. Control is not a production patch.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Перевірка RNG при імпорті у фактичному MicroModel", "",
             "Граф: шлях 0–1–2, beta=1, gamma=0, початково заражений вузол 0. Імпорт у фіксований момент ln(4/3). Аналізуємо запуски, в яких до імпорту не було локальної події: стан тоді однаковий у всіх запусках.", "",
             "Правильний рівномірний імпорт вибирає вузли 1 і 2 з імовірністю 1/2. Якщо заражено вузол 1, наступна інтенсивність 1; якщо вузол 2 — 2. Тому середнє наступне очікування має бути 0.5/1 + 0.5/2 = 0.75.", "",
             "| Варіант | Прийнятих запусків | Частка імпорту у вузол 2 | Середнє очікування | 95% ДІ середнього |",
             "|---|---:|---:|---:|---|",
             "| Точне значення | — | 0.500000 | 0.750000 | — |"]
    for name, r in results.items():
        lo, hi = r["mean_next_wait_ci95"]
        lines.append(f"| {name} | {r['conditioned_runs']} | {r['fraction_node2']:.6f} | {r['mean_next_wait']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "Причина: Exp(1) генерується як -ln(1-U). Умова відсутності події до ln(4/3) означає U>1/4. Поточний simulate_until відкочує RNG до цього U, а import_infection використовує той самий стан для choice. У CPython choice для двох елементів спершу бере два старші біти: при U між 1/4 і 1/2 одразу вибирається другий елемент; при U>1/2 відбувається rejection і беруться наступні біти. Це пояснює частку другого елемента приблизно 2/3 замість 1/2.", "",
              "Контроль використовує той самий початковий RNG та ті самі прийняті запуски, але перед placement переходить до збереженого RNG-стану після відкинутого proposal. Локальний час після імпорту далі генерується заново. Отже, контроль виправляє повторне використання випадковості, зберігаючи політику перезапуску.", "",
              "Цей приклад доводить конкретну помилку conditional placement у локальному рушії. Він не оцінює величину впливу на збережені повномережеві експерименти. Контрольна зміна не є готовим виправленням для спільних RNG, макрокопій або всього планувальника.", "",
              "Відтворення:", "", "```bash",
              f".venv/bin/python extrascripts/validation/check_import_rng_conditioning.py --runs {args.runs} --seed {args.seed}", "```", ""]
    (args.output / "report_uk.md").write_text("\n".join(lines))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
