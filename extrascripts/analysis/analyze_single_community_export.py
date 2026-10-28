#!/usr/bin/env python3
"""Analyze independent-RNG checkpoints from the single-community export study.

This reader never launches simulations.  It accepts incomplete collections,
records their actual sample size and censoring, and never converts T_MAX into
an export time.  Results go only to output/analysis/single_community_export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extrascripts.runners.checkpoint_io import read_checkpoint
from extrascripts.runners.run_original_benchmarks_independent_rng import FULL, MICROMACRO

INPUT_ROOT = REPO_ROOT / "output" / "data"
OUTPUT_DIR = REPO_ROOT / "output" / "analysis" / "single_community_export"
EXPERIMENT = "single_community_export_independent_rng_v1"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summary(values: list[float], prefix: str) -> dict[str, float | int | None]:
    if not values:
        return {f"{prefix}_{name}": None for name in ("mean", "median", "q25", "q75", "iqr") } | {f"{prefix}_N": 0}
    array = np.asarray(values, dtype=float)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return {f"{prefix}_N": int(array.size), f"{prefix}_mean": float(array.mean()),
            f"{prefix}_median": float(np.median(array)), f"{prefix}_q25": float(q25),
            f"{prefix}_q75": float(q75), f"{prefix}_iqr": float(q75 - q25)}


def _all_checkpoints() -> list[Path]:
    return sorted(path for path in INPUT_ROOT.glob("export_single_*_independent_rng_*/checkpoints/*.json.gz") if path.is_file())


def _cell_key(payload: dict[str, Any]) -> str:
    return str(payload["scenario"])


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    descriptors: dict[tuple[str, int], dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in _all_checkpoints():
        try:
            payload = read_checkpoint(path)
            if payload.get("experiment") != EXPERIMENT:
                continue
            if payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is not True:
                failures.append({"path": str(path), "issue": "not_independent_rng"})
                continue
            cell = payload["cell"]
            scenario = _cell_key(payload)
            graph_id, event_id = int(payload["graph_id"]), int(payload["event_id"])
            network = payload["network"]
            graph_descriptor = dict(network["descriptors"])
            descriptor_key = (scenario, graph_id)
            if descriptor_key not in descriptors:
                descriptors[descriptor_key] = {"scenario": scenario, "graph_id": graph_id, **cell, **graph_descriptor}
            for run in payload["runs"]:
                observations.append({
                    "scenario": scenario, "graph_id": graph_id, "event_id": event_id,
                    "simulator": run["simulator"], **cell,
                    "T_export": _finite(run.get("T_export")),
                    "export_observed": bool(run.get("export_observed")),
                    "T_bridge_first": _finite(run.get("T_bridge_first")),
                    "bridge_infected_count_at_export_minus": run.get("bridge_infected_count_at_export_minus"),
                    "bypass": bool(run.get("bypass")),
                    "I_source_at_export_minus": run.get("I_source_at_export_minus"),
                    "bridge_to_export_delay": _finite(run.get("bridge_to_export_delay")),
                    "runtime_seconds": _finite(run.get("runtime_seconds")),
                    "stop_reason": (run.get("termination") or {}).get("stop_reason"),
                    "stop_time": _finite((run.get("termination") or {}).get("stop_time")),
                    "checkpoint": str(path.relative_to(REPO_ROOT)),
                })
            inventory.append({"scenario": scenario, "graph_id": graph_id, "event_id": event_id,
                              "checkpoint": str(path.relative_to(REPO_ROOT)), "design_hash": payload.get("design_hash")})
        except Exception as exc:  # malformed files are evidence, not silently ignored
            failures.append({"path": str(path), "issue": f"{type(exc).__name__}: {exc}"})
    return observations, list(descriptors.values()), inventory, failures


def metrics(observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[row["scenario"]].append(row)
    cells: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for scenario, rows in sorted(grouped.items()):
        template = next(iter(rows))
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_pair: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_model[row["simulator"]].append(row)
            by_pair[(int(row["graph_id"]), int(row["event_id"]))][row["simulator"]] = row
        full = by_model[FULL]
        mm = by_model[MICROMACRO]
        full_observed = [float(row["T_export"]) for row in full if row["export_observed"]]
        mm_observed = [float(row["T_export"]) for row in mm if row["export_observed"]]
        paired = []
        for (graph_id, event_id), models in sorted(by_pair.items()):
            left, right = models.get(FULL), models.get(MICROMACRO)
            if left is None or right is None:
                continue
            if left["export_observed"] and right["export_observed"]:
                delta = float(right["T_export"]) - float(left["T_export"])
                paired.append(delta)
                paired_rows.append({"scenario": scenario, "graph_id": graph_id, "event_id": event_id,
                                    "T_export_Full": left["T_export"], "T_export_MicroMacro": right["T_export"],
                                    "signed_export_error_MM_minus_Full": delta, **{k: template[k] for k in ("block", "kind", "topology", "placement", "k")}})
        completion_full = sum(row["export_observed"] for row in full) / len(full) if full else math.nan
        completion_mm = sum(row["export_observed"] for row in mm) / len(mm) if mm else math.nan
        record: dict[str, Any] = {"scenario": scenario, **{k: template.get(k) for k in ("block", "kind", "topology", "placement", "side", "ell", "q", "cut_edges", "k", "d", "beta_out")},
                                  "N_Full": len(full), "N_MicroMacro": len(mm),
                                  "completion_Full": completion_full, "completion_MicroMacro": completion_mm,
                                  "bypass_Full": sum(row["bypass"] for row in full) / len(full) if full else math.nan,
                                  "bypass_MicroMacro": sum(row["bypass"] for row in mm) / len(mm) if mm else math.nan,
                                  "paired_complete_N": len(paired),
                                  **_summary(full_observed, "Full_export"), **_summary(mm_observed, "MM_export"), **_summary(paired, "paired_signed_error")}
        if full_observed and mm_observed and completion_full == 1.0 and completion_mm == 1.0:
            record["W1_export"] = float(wasserstein_distance(full_observed, mm_observed))
            record["relative_W1_export"] = record["W1_export"] / record["Full_export_mean"] if record["Full_export_mean"] else None
            record["KS_export"] = float(ks_2samp(full_observed, mm_observed, method="auto").statistic)
        else:
            record["W1_export"] = record["relative_W1_export"] = record["KS_export"] = None
        cells.append(record)
    return cells, paired_rows


def quality(observations: list[dict[str, Any]], inventory: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    paired = defaultdict(dict)
    for row in observations:
        paired[(row["scenario"], row["graph_id"], row["event_id"])][row["simulator"]] = row
    full_bypass = [key for key, rows in paired.items() if rows.get(FULL, {}).get("bypass")]
    missing_models = [key for key, rows in paired.items() if set(rows) != {FULL, MICROMACRO}]
    return {"experiment": EXPERIMENT, "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_count": len(inventory), "observation_count": len(observations),
            "distinct_scenarios": len({row["scenario"] for row in observations}),
            "malformed_checkpoints": failures, "full_bypass_pairs": full_bypass,
            "missing_model_pairs": missing_models,
            "valid_for_analysis": not failures and not full_bypass and not missing_models,
            "censoring_policy": "unrestricted W1 and KS are emitted only when both model completion rates equal one; T_MAX is never used as an event time."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-invalid", action="store_true", help="write tables even when structural audit fails")
    args = parser.parse_args()
    observations, graph_descriptors, inventory, failures = collect()
    report = quality(observations, inventory, failures)
    if not observations:
        raise SystemExit("No single-community independent-RNG checkpoints found.")
    if not report["valid_for_analysis"] and not args.allow_invalid:
        _write_json(OUTPUT_DIR / "analysis_manifest.json", report)
        raise SystemExit("Data-quality gate failed; inspect analysis_manifest.json.")
    cell_metrics, paired = metrics(observations)
    _write_csv(OUTPUT_DIR / "export_observables.csv", observations)
    _write_csv(OUTPUT_DIR / "graph_descriptors.csv", graph_descriptors)
    _write_csv(OUTPUT_DIR / "inventory.csv", inventory)
    _write_csv(OUTPUT_DIR / "cell_metrics.csv", cell_metrics)
    _write_csv(OUTPUT_DIR / "paired_export_errors.csv", paired)
    report["files_sha256"] = {name: hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest() for name in
                              ("export_observables.csv", "graph_descriptors.csv", "inventory.csv", "cell_metrics.csv", "paired_export_errors.csv")}
    _write_json(OUTPUT_DIR / "analysis_manifest.json", report)
    print(f"[analysis] checkpoints={len(inventory)} observations={len(observations)} scenarios={report['distinct_scenarios']}")
    print(f"[analysis] saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
