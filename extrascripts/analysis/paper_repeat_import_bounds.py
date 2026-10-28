"""Identify unavoidable repeated imports in the paper's saved SI trajectories.

Compact checkpoints omit channel labels. An infected node without any infected
internal neighbour at its infection time must have been a macro import. Such
events after the community's first infection give lower bounds, never exact
counts. Reads existing graphs/checkpoints only; does not run simulations.
"""

import argparse
import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIZES = (50, 100, 250, 1000)


def read(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def analyze_cell(item):
    n, scenario = item
    paths = sorted(scenario.glob("checkpoints/*.json.gz"))
    tokens = scenario.name.split("__")
    aspl = int(next(x[4:] for x in tokens if x.startswith("aspl")))
    k = int(next(x[1:] for x in tokens if x.startswith("k") and x[1:].isdigit()))
    row = dict(n=n, aspl=aspl, k=k, runs=len(paths),
               identifiable_repeat_imports=0, identifiable_repeat_imports_to_T9=0,
               runs_with_identifiable_repeat=0, runs_with_identifiable_repeat_to_T9=0,
               community_runs_with_identifiable_repeat=0,
               community_runs_with_identifiable_repeat_to_T9=0,
               seeded_community_identifiable_repeats=0,
               checked_checkpoints=0, method="lower_bound_from_internal_neighbours")
    details = []
    if aspl == 1:
        # In a clique every infection after the first has an internal infected
        # neighbour. This criterion cannot identify ANY additional imports.
        row["method"] = "uninformative_bound_for_complete_graph"
    else:
        cache = {}
        network = None
        for path in paths:
            payload = read(path)
            task = payload["task"]
            if task["network_id"] != network:
                cache.clear()
                network = task["network_id"]
            mm = next(run for run in payload["runs"] if "MicroMacro" in run["simulator"])
            events = np.asarray(mm["infection_events"], dtype=float)
            ids = events[:, 2].astype(np.int64)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate infections in SI: {path}")
            if np.any(ids < 0) or np.any(ids >= 10 * n):
                raise ValueError(f"Unexpected node IDs: {path}")
            if not np.all(events[:, 1].astype(int) == ids // n):
                raise ValueError(f"Community/node mismatch: {path}")
            times = np.full(10 * n, np.inf)
            times[ids] = events[:, 0]
            times = times.reshape(10, n)
            first = times.min(axis=1)
            if not np.all(np.isfinite(first)):
                raise ValueError(f"Not all paper arrivals observed: {path}")
            counts, before = [], []
            for community, relative in enumerate(task["graph_paths"]):
                if relative not in cache:
                    graph = read(ROOT / relative)
                    edges = np.asarray(graph["edges"], dtype=float)
                    edges = edges[edges[:, 2] > 0, :2].astype(np.int64)
                    cache[relative] = edges
                edges = cache[relative]
                ts = times[community]
                u, v = edges[:, 0], edges[:, 1]
                internal_possible = np.zeros(n, dtype=bool)
                # Simultaneous neighbour infections are treated conservatively
                # as possible internal causes; no inference from tie ordering.
                internal_possible[v[ts[u] <= ts[v]]] = True
                internal_possible[u[ts[v] <= ts[u]]] = True
                repeated = (~internal_possible) & np.isfinite(ts) & (ts > first[community])
                counts.append(int(repeated.sum()))
                before.append(int(np.count_nonzero(repeated & (ts <= first[9]))))
            total, pre = sum(counts), sum(before)
            row["identifiable_repeat_imports"] += total
            row["identifiable_repeat_imports_to_T9"] += pre
            row["runs_with_identifiable_repeat"] += total > 0
            row["runs_with_identifiable_repeat_to_T9"] += pre > 0
            row["community_runs_with_identifiable_repeat"] += sum(x > 0 for x in counts)
            row["community_runs_with_identifiable_repeat_to_T9"] += sum(x > 0 for x in before)
            row["seeded_community_identifiable_repeats"] += counts[0]
            row["checked_checkpoints"] += 1
            details.append(dict(n=n, aspl=aspl, k=k, network_id=int(network),
                                replicate_id=int(task["replicate_id"]),
                                repeats_lower_bound=total, repeats_to_T9_lower_bound=pre))
    if row["runs"] != 400:
        raise ValueError(f"Expected 400 paper runs: {scenario}, got {row['runs']}")
    count = row["runs"]
    total = row["identifiable_repeat_imports"]
    pre = row["identifiable_repeat_imports_to_T9"]
    row.update(mean_repeats_lower_bound=total / count,
               mean_repeats_to_T9_lower_bound=pre / count,
               runs_with_repeat_lower_bound_percent=100 * row["runs_with_identifiable_repeat"] / count,
               runs_with_repeat_to_T9_lower_bound_percent=100 * row["runs_with_identifiable_repeat_to_T9"] / count,
               repeat_share_of_successful_imports_lower_bound_percent=100 * total / (9 * count + total),
               repeat_share_of_successful_imports_to_T9_lower_bound_percent=100 * pre / (9 * count + pre))
    return row, details


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "output/analysis/paper_repeat_import_bounds")
    args = parser.parse_args()
    tasks = []
    for n in SIZES:
        root = ROOT / "output/data" / f"l10_n{n}_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
        scenarios = sorted((root / "scenarios").iterdir())
        if len(scenarios) != 12:
            raise ValueError(f"Expected 12 paper scenarios: {root}")
        tasks.extend((n, s) for s in scenarios)
    rows, runs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(analyze_cell, task) for task in tasks]
        for future in as_completed(pending):
            row, details = future.result()
            rows.append(row)
            runs.extend(details)
            print(f"{len(rows)}/48 n={row['n']} ASPL={row['aspl']} k={row['k']}: lower mean={row['mean_repeats_to_T9_lower_bound']:.3f} to T9", flush=True)
    rows.sort(key=lambda x: (x["n"], x["aspl"], x["k"]))
    runs.sort(key=lambda x: (x["n"], x["aspl"], x["k"], x["network_id"], x["replicate_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cell_bounds.csv", rows)
    write_csv(args.output / "per_run_bounds.csv", runs)
    totals = {key: sum(r[key] for r in rows) for key in (
        "runs", "checked_checkpoints", "identifiable_repeat_imports",
        "identifiable_repeat_imports_to_T9", "runs_with_identifiable_repeat",
        "runs_with_identifiable_repeat_to_T9")}
    metadata = dict(completion="finished", article="paper/SMPT paper/claude-version_merged.tex",
                    exact_macro_counts_available=False, totals=totals,
                    assumptions="SI, positive internal edge weights, one initial seed in community 0, all first arrivals observed. Infections with no previously/simultaneously infected internal neighbour are forced imports. Only those strictly after the community's first infection count as repeats.",
                    limitation="Lower bounds only. ASPL=1 bound is uninformative zero; no absence of repeated imports inferred. Successful imports denominator: nine first imports plus repeats. Full horizon and up-to-T9 reported separately.")
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Повторні макроімпорти у даних статті: доказові нижні межі", "",
             "Точну кількість відновити неможливо: збережено лише час, спільноту та вузол, без типу події. Це прямо зазначено в claude-version_merged.tex, рядок 563.", "",
             "Якщо вузол заразився без зараженого внутрішнього сусіда, це однозначно макроімпорт. Якщо в спільноті вже був інший інфікований, цей імпорт є повторним. Імпорти у вузли з зараженими сусідами не розпізнаються, тому наведено лише нижні межі.", "",
             "У SI вже заражена спільнота завжди має I>0. Відсікання на T9 означає перше зараження останньої спільноти — основний endpoint статті. Повна траєкторія може містити пізні імпорти, які на T9 не вплинули.", "",
             "| n | ASPL | k | Запусків | Мін. повторних на запуск до T9 | Мін. % запусків із повтором до T9 | Мін. % повторів серед імпортів до T9 | Мін. повторних на повну траєкторію |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['aspl']} | {r['k']} | {r['runs']} | {r['mean_repeats_to_T9_lower_bound']:.3f} | {r['runs_with_repeat_to_T9_lower_bound_percent']:.2f} | {r['repeat_share_of_successful_imports_to_T9_lower_bound_percent']:.2f} | {r['mean_repeats_lower_bound']:.3f} |")
    lines += ["", "Для клік (ASPL=1) нуль у таблиці означає, що критерій нічого не розпізнає; це не доказ відсутності повторних імпортів.", "",
              "У clique–leaf і source–leaf до endpoint першого зараження листка повторний імпорт неможливий: листок ще не інфікований і не може передавати інфекцію назад, а перший імпорт у нього завершує вимірювання. Події після цього endpoint не входять у цей висновок.", "",
              "Частка повторних серед успішних макроімпортів обмежена знизу Rmin/(9N+Rmin), оскільки на кожну повністю досягнуту десятиспільнотну траєкторію припадає рівно дев'ять перших імпортів.", "",
              "Відтворення: `.venv/bin/python extrascripts/analysis/paper_repeat_import_bounds.py --workers 2`. Нові симуляції не запускались.", ""]
    (args.output / "report_uk.md").write_text("\n".join(lines))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
