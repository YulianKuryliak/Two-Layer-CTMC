#!/usr/bin/env python3
"""Audit completed corrected chain checkpoints and reproduce old metrics.

Saved data only. New immutable analysis directory; no simulator is imported.
"""
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extrascripts.runners.rerun_smpt_paper import log, object_hash, utc


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checked(path, sha=None):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if sha is not None and actual != sha:
        raise RuntimeError(f"Changed saved source: {path}")
    return json.loads(gzip.decompress(raw)), actual


def cell_key(task, metrics):
    match = metrics.SCENARIO_RE.search(task["scenario"])
    if match is None:
        raise RuntimeError("Unrecognized chain scenario")
    return int(match["n"]), int(match["aspl"]), int(match["k"])


def extract(task, full, mm, metrics):
    n, aspl, k = cell_key(task, metrics)
    original = task["original_task"]
    match = metrics.SCENARIO_RE.search(task["scenario"])
    row = dict(n=n, aspl=aspl, k=k, mean_degree=float(match["d"].replace("p", ".")),
               network_id=original["network_id"], replicate_id=original["replicate_id"],
               full_runtime_s=full["runtime_seconds"], mm_runtime_s=mm["runtime_seconds"],
               extended=task["full_extension"] is not None)
    bridges = {c: set() for c in range(9)}
    for u, v in original["interface_edges"]:
        assert v // n == u // n + 1
        bridges[u // n].add(u)
    for label, run in (("full", full), ("mm", mm)):
        assert len(run["arrivals"]) == 10
        for c, arrival in enumerate(run["arrivals"]):
            row[f"{label}_T{c}"] = np.nan if arrival is None else float(arrival)
        times = metrics.infection_times(run["infection_events"])
        for c in range(9):
            hits = [times[node] for node in bridges[c] if node in times]
            row[f"{label}_TB{c}"] = min(hits) if hits else np.nan
        if label == "full":
            nodes = np.fromiter(times.keys(), dtype=np.int64)
            values = np.fromiter(times.values(), dtype=float)
            spreads = [float(values[nodes // n == c].std()) for c in range(9)
                       if np.count_nonzero(nodes // n == c) == n]
            row["full_sd"] = float(np.mean(spreads)) if spreads else np.nan
    row.update(full_horizon=task["full_horizon"], mm_horizon=task["T_max"],
               repeat_imports=mm.get("repeat_imports", np.nan),
               repeat_imports_to_endpoint=mm.get("repeat_imports_to_endpoint", np.nan))
    return row


def trajectory_cache(time_groups, rows, target):
    # Preserve the old rendering policy and common window for both series.
    if any(len(series) != 10000 for group in time_groups for series in group):
        raise RuntimeError("Trajectory/IQR plot requires full n1000 SI completion")
    full_arrivals = rows.full_T9.to_numpy()
    mm_arrivals = rows.mm_T9.to_numpy()
    if not np.isfinite(np.r_[full_arrivals, mm_arrivals]).all():
        raise RuntimeError("CDF requires observed arrivals; do not drop censored runs")
    final_times = np.asarray([series[-1] for group in time_groups for series in group])
    horizon = 1.05 * max(full_arrivals.max(), mm_arrivals.max(), np.quantile(final_times, .99))
    grid = np.linspace(0, horizon, 501)
    quantiles = [np.quantile(np.asarray([np.searchsorted(series, grid, side="right")
                                        for series in group], dtype=np.int16), [.25, .5, .75], axis=0)
                 for group in time_groups]
    np.savez_compressed(target, checkpoint_count=len(rows), grid=grid,
                        full_quantiles=quantiles[0], mm_quantiles=quantiles[1],
                        full_arrivals=full_arrivals, mm_arrivals=mm_arrivals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/chain_twoscale_residual_midpoint_20261007")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), ROOT / "extrascripts/analysis/smpt_twoscale_paper_metrics.py",
               ROOT / "extrascripts/visualization/plot_l10_aspl_chain_random_initial.py"]
    code = {str(path.relative_to(ROOT)): digest(path) for path in scripts}
    fingerprint = object_hash(code)
    log(out, "saved_analysis_started", simulation=False, command=sys.argv, code_fingerprint=fingerprint,
        parameters={"source": str(source), "N_BOOT": 2000, "N_PERM": 2000},
        seeds=20260930, trajectory_policy="all matched old/new saved pairs; no resampling of events",
        resume="disabled; new analysis directory required")
    try:
        metrics = load("original_chain_metrics", "extrascripts/analysis/smpt_twoscale_paper_metrics.py")
        layers = load("original_chain_layers", "extrascripts/visualization/plot_l10_aspl_chain_random_initial.py")
        manifest = json.loads((source / "manifest.json").read_text())
        status = json.loads((source / "run_status.json").read_text())
        assert status["status"] == "complete" and status["full_design_complete"]
        tasks, _ = read_checked(source / "tasks.json.gz")
        tasks = tasks["tasks"]
        assert len(tasks) == 19200 and object_hash(tasks) == manifest["tasks_sha256"]
        assert len({task["id"] for task in tasks}) == 19200
        expected = {task["id"] + ".json.gz" for task in tasks}
        actual = {str(path.relative_to(source / "checkpoints")) for path in (source / "checkpoints").rglob("*.json.gz")}
        assert actual == expected, "Missing or extra checkpoints"
        for relative, sha in manifest["code_sha256"].items():
            assert digest(source / "sources" / relative) == sha, relative
        for reference in manifest["graphs"]:
            assert digest(source / reference["snapshot"]) == reference["sha256"]
        for path in scripts:
            target = out / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        cache = out / "trajectory_cdf_cache"
        cache.mkdir()
        new_rows, old_rows, audit, layer_rows, median_rows = [], [], [], [], []
        seen = set()
        for key, group in itertools.groupby(tasks, key=lambda task: cell_key(task, metrics)):
            n, aspl, k = key
            assert key not in seen
            seen.add(key)
            cell_tasks = list(group)
            assert len(cell_tasks) == 400
            local_rows = []
            time_groups = [[], []]
            for task in cell_tasks:
                checkpoint, checkpoint_sha = read_checked(source / "checkpoints" / (task["id"] + ".json.gz"))
                assert checkpoint["status"] == "complete"
                assert checkpoint["code_fingerprint"] == manifest["code_fingerprint"]
                assert checkpoint["version"] == manifest["version"]
                assert checkpoint["rerun_task"] == task and checkpoint["task"] == task["original_task"]
                assert checkpoint["event_seeds"] == task["event_seeds"]
                assert checkpoint["input_sha256"] == task["input_sha256"]
                assert checkpoint["comparison_horizons"] == {"Full": task["full_horizon"], "TwoScale": task["T_max"]}
                original, old_sha = read_checked(ROOT / task["source_checkpoint"]["source"], task["source_checkpoint"]["sha256"])
                assert original["task"] == task["original_task"]
                assert original["event_seeds"] == task["event_seeds"]
                old_full, old_mm = original["runs"]
                if task["full_extension"]:
                    extension, _ = read_checked(ROOT / task["full_extension"]["source"], task["full_extension"]["sha256"])
                    assert extension["task"] == original["task"] and extension["pathwise_prefix_verified"]
                    old_full = extension["full_run"]
                full, mm = checkpoint["runs"]
                assert full["simulator"] == "Full Micro" and mm["simulator"].startswith("Midpoint TwoScale")
                assert full == old_full, "Reused Full row changed"
                assert object_hash(full) == checkpoint["full_baseline_provenance"]["row_sha256"]
                new_row, old_row = extract(task, full, mm, metrics), extract(task, old_full, old_mm, metrics)
                new_rows.append(new_row)
                old_rows.append(old_row)
                local_rows.append(new_row)
                audit.append(dict(task_id=task["id"], n=n, aspl=aspl, k=k,
                    checkpoint_sha256=checkpoint_sha, historical_sha256=old_sha,
                    full_exact=True, mm_events_exact=mm["infection_events"] == old_mm["infection_events"],
                    mm_arrivals_exact=mm["arrivals"] == old_mm["arrivals"],
                    full_T9=full["arrivals"][9], old_mm_T9=old_mm["arrivals"][9], new_mm_T9=mm["arrivals"][9],
                    full_horizon=task["full_horizon"], mm_horizon=task["T_max"],
                    full_T9_observed_by_30000=full["arrivals"][9] is not None and full["arrivals"][9] <= 30000,
                    mm_T9_observed_by_30000=mm["arrivals"][9] is not None and mm["arrivals"][9] <= 30000))
                if n == 1000:
                    for index, run in enumerate((full, mm)):
                        time_groups[index].append(np.sort(np.fromiter((event[0] for event in run["infection_events"]), dtype=float)))
            frame = pd.DataFrame(local_rows)
            if n == 1000:
                trajectory_cache(time_groups, frame, cache / f"aspl{aspl}_k{k}.npz")
                pairs = {(aspl, k): [(list(row[[f"full_T{c}" for c in range(10)]]),
                                      list(row[[f"mm_T{c}" for c in range(10)]])) for _, row in frame.iterrows()]}
                # Original metric routine loops over all 12 cells; one-cell
                # globals are rebound only in this isolated imported module.
                layers.ASPL_VALUES, layers.K_VALUES = (aspl,), (k,)
                layer_rows.extend(layers.make_metrics(pairs))
                for c in range(1, 10):
                    f, m = frame[f"full_T{c}"].to_numpy(), frame[f"mm_T{c}"].to_numpy()
                    assert np.isfinite(np.r_[f, m]).all(), "Do not silently omit censoring"
                    median_rows.append(dict(target_ASPL=aspl, k=k, community=c, paired_runs=400,
                        median_full=np.median(f), median_mm=np.median(m),
                        signed_relative_median_shift_percent=metrics.median_shift(f, m),
                        relative_w1_percent=metrics.rel_w1(f, m)))
            print(f"[audit] n={n}, ASPL={aspl}, k={k}: 400 pairs; Full exact; {len(new_rows)}/19200", flush=True)
        new = pd.DataFrame(new_rows).sort_values(["n", "aspl", "k", "network_id", "replicate_id"]).reset_index(drop=True)
        old = pd.DataFrame(old_rows).sort_values(["n", "aspl", "k", "network_id", "replicate_id"]).reset_index(drop=True)
        new.to_csv(out / "per_run.csv.gz", index=False)
        old.to_csv(out / "old_per_run_recomputed.csv.gz", index=False)
        pd.DataFrame(audit).to_csv(out / "checkpoint_comparison.csv.gz", index=False)
        pd.DataFrame(layer_rows).to_csv(out / "layer_metrics.csv", index=False)
        pd.DataFrame(median_rows).to_csv(out / "median_layer_metrics.csv", index=False)
        summary_rows, terminal_rows = [], []
        rng = np.random.default_rng(metrics.SEED)
        for (n, aspl, k), cell in new.groupby(["n", "aspl", "k"], sort=True):
            assert len(cell) == 400
            summary = dict(n=n, aspl=aspl, k=k, mean_degree=cell.mean_degree.iloc[0], **metrics.summarise(cell, rng))
            summary_rows.append(summary)
            terminal_rows.append(dict(n=n, target_ASPL=aspl, k=k, checkpoint_count=400, paired_observed=400,
                extended_full_to40000=int(cell.extended.sum()), signed_median_shift_percent=summary["median_shift_pct"],
                relative_w1_percent=summary["rel_w1_pct"]))
            print(f"[metrics] n={n}, ASPL={aspl}, k={k}", flush=True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out / "cell_summary.csv", index=False)
        pd.DataFrame(terminal_rows).to_csv(out / "terminal_metrics.csv", index=False)
        old_summary_path = ROOT / "output/analysis/smpt_twoscale_paper/cell_summary.csv"
        old_summary = pd.read_csv(old_summary_path)
        # Establish that the cached baseline shown in old figures is precisely
        # this saved corpus, including its two historical Full extensions.
        cached = pd.read_csv(old_summary_path.with_name("per_run.csv.gz")).sort_values(
            ["n", "aspl", "k", "network_id", "replicate_id"]).reset_index(drop=True)
        columns = [c for c in cached if c in old and c not in ("mm_runtime_s", "full_runtime_s")]
        for column in columns:
            assert np.allclose(cached[column].to_numpy(dtype=float), old[column].to_numpy(dtype=float),
                               rtol=1e-13, atol=1e-10, equal_nan=True), column
        comparison = old_summary.merge(summary, on=["n", "aspl", "k"], suffixes=("_old", "_new"), validate="one_to_one")
        for field in ("rel_w1_pct", "median_shift_pct", "mean_shift_pct", "first_cross_shift_pct",
                      "later_cross_shift_pct", "mm_chain_bypass_pct"):
            comparison[field + "_change_pp"] = comparison[field + "_new"] - comparison[field + "_old"]
        comparison.to_csv(out / "cell_comparison.csv", index=False)
        for relative, sha in code.items():
            assert digest(ROOT / relative) == sha
        results = dict(status="complete", created_utc=utc(), source=str(source),
            source_manifest_sha256=digest(source / "manifest.json"), source_code_fingerprint=manifest["code_fingerprint"],
            analysis_code_fingerprint=fingerprint, code_sha256=code, tasks_sha256=manifest["tasks_sha256"],
            paired_tasks=19200, cells=48, pairs_per_cell=400, full_rows_exact=19200,
            mm_event_sequences_changed=sum(not row["mm_events_exact"] for row in audit),
            mm_arrival_sequences_changed=sum(not row["mm_arrivals_exact"] for row in audit),
            extended_full_runs=sum(task["full_extension"] is not None for task in tasks),
            common_30000_full_T9_observed=sum(row["full_T9_observed_by_30000"] for row in audit),
            common_30000_mm_T9_observed=sum(row["mm_T9_observed_by_30000"] for row in audit),
            old_summary_sha256=digest(old_summary_path), baseline_cache_matches_recomputed=True,
            summary_parameters=dict(N_BOOT=metrics.N_BOOT, N_PERM=metrics.N_PERM, seed=metrics.SEED),
            notes="Old figure metric definitions retained; two Full40000 extensions explicitly counted. Completion compared at common30000 separately. Trajectory IQR is observation spread, not CI. Runtime changes are not scientific accuracy.")
        (out / "analysis_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
        log(out, "saved_analysis_completed", simulation=False, code_fingerprint=fingerprint, status="complete",
            paired_tasks=19200, source_code_fingerprint=manifest["code_fingerprint"])
        print(json.dumps(results, indent=2), flush=True)
    except BaseException as error:
        (out / "failure.txt").write_text(traceback.format_exc())
        log(out, "saved_analysis_failed", simulation=False, code_fingerprint=fingerprint, error=repr(error))
        raise


if __name__ == "__main__":
    main()
