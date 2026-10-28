#!/usr/bin/env python3
"""CE-12: power and precision calibration from saved paired arrivals only.

This runner performs no simulation.  It reuses the final CE-01 ablation data
and every per-cell arrival file from the CE-06 full grid.  Terminal arrival
times are restricted to each experiment's horizon before differencing, so a
censored member of a pair is retained as ``T_end`` rather than discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CE01_DIR = ROOT / "output" / "ablation_hazard_placement" / "final_100"
CE06_DIR = ROOT / "output" / "claude_experiments" / "ce06_regime_map_full"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "claude_experiments" / "ce12_power_calibration"
ALPHA = 0.05
POWERS = (0.80, 0.90)
MARGIN_FRACTIONS = (0.025, 0.05, 0.10)
MASTER_SEED = 812_120_026
Z_ALPHA = statistics.NormalDist().inv_cdf(1.0 - ALPHA / 2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_inventory() -> list[dict[str, Any]]:
    ce01_files = [CE01_DIR / "arrival_times.csv", CE01_DIR / "experiment_config.json"]
    ce06_files = [CE06_DIR / "config.json", CE06_DIR / "manifest.json"]
    ce06_files.extend(sorted((CE06_DIR / "cells").glob("arrivals_*.csv")))
    files = ce01_files + ce06_files
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing CE-12 source file(s): " + ", ".join(missing))
    if len(ce06_files) <= 2:
        raise FileNotFoundError(f"no CE-06 cell arrivals found under {CE06_DIR / 'cells'}")
    return [
        {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in files
    ]


def _same_sources(old: Any, new: list[dict[str, Any]]) -> bool:
    if not isinstance(old, list):
        return False
    return [(x.get("path"), x.get("sha256")) for x in old] == [
        (x["path"], x["sha256"]) for x in new
    ]


def _outputs_match(output_dir: Path, old: dict[str, Any], names: list[str]) -> bool:
    checksums = old.get("output_checksums")
    if not isinstance(checksums, dict):
        return False
    return all(
        (output_dir / name).is_file() and checksums.get(name) == _sha256(output_dir / name)
        for name in names
    )


def _restricted(values: pd.Series, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(raw) & (raw <= horizon)
    restricted = np.where(complete, np.minimum(raw, horizon), horizon)
    return restricted.astype(float), complete.astype(bool)


def _paired_terminal(
    frame: pd.DataFrame,
    *,
    id_column: str,
    model_column: str,
    reference: str,
    variant: str,
    terminal_layer: int,
    expected_ids: Iterable[Any],
    horizon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terminal = frame.loc[frame["layer"].eq(terminal_layer), [id_column, model_column, "T_i"]]
    if terminal.duplicated([id_column, model_column]).any():
        raise ValueError(f"duplicate terminal rows for {reference} versus {variant}")
    pivot = terminal.pivot(index=id_column, columns=model_column, values="T_i")
    ids = pd.Index(sorted(expected_ids), name=id_column)
    known_models = set(frame[model_column].dropna().astype(str))
    absent = [model for model in (reference, variant) if model not in known_models]
    if absent:
        raise ValueError(f"missing model(s) in source data: {absent}")
    expected_set = set(ids.tolist())
    for model in (reference, variant):
        present_ids = set(frame.loc[frame[model_column].eq(model), id_column])
        if not expected_set.issubset(present_ids):
            missing_ids = sorted(expected_set - present_ids)
            raise ValueError(f"missing paired source replication(s) for {model}: {missing_ids}")
    for model in (reference, variant):
        if model not in pivot.columns:
            pivot[model] = np.nan
    # Reindexing is intentional: absence of a terminal row means horizon censoring.
    pair = pivot.reindex(ids)[[reference, variant]]
    ref, ref_complete = _restricted(pair[reference], horizon)
    var, var_complete = _restricted(pair[variant], horizon)
    return ref, var, ref_complete, var_complete


def _ceil_n(value: float) -> int:
    if not math.isfinite(value):
        return 1
    return max(2, int(math.ceil(value)))


def _mcnemar_exact(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(n01, n10) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _analyse_comparison(
    *,
    experiment: str,
    scenario: str,
    reference_name: str,
    variant_name: str,
    horizon: float,
    reference: np.ndarray,
    variant: np.ndarray,
    ref_complete: np.ndarray,
    var_complete: np.ndarray,
    bootstrap_reps: int,
    comparison_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(reference) != len(variant) or len(reference) < 2:
        raise ValueError(f"invalid paired sample for {experiment}/{scenario}/{variant_name}")
    n = len(reference)
    diff = variant - reference
    sd_ref = float(np.std(reference, ddof=1))
    sd_var = float(np.std(variant, ddof=1))
    sd_diff = float(np.std(diff, ddof=1))
    covariance = float(np.cov(reference, variant, ddof=1)[0, 1])
    correlation = float(np.corrcoef(reference, variant)[0, 1]) if sd_ref > 0 and sd_var > 0 else float("nan")
    observed_mean = float(np.mean(diff))
    direction = -1.0 if observed_mean < 0 else 1.0
    normal_rows: list[dict[str, Any]] = []
    empirical_rows: list[dict[str, Any]] = []

    for margin_fraction in MARGIN_FRACTIONS:
        margin = margin_fraction * horizon
        z80 = statistics.NormalDist().inv_cdf(POWERS[0])
        z90 = statistics.NormalDist().inv_cdf(POWERS[1])
        paired_n80 = _ceil_n(((Z_ALPHA + z80) * sd_diff / margin) ** 2)
        paired_n90 = _ceil_n(((Z_ALPHA + z90) * sd_diff / margin) ** 2)
        variance_sum = sd_ref**2 + sd_var**2
        unpaired_n80 = _ceil_n((Z_ALPHA + z80) ** 2 * variance_sum / margin**2)
        unpaired_n90 = _ceil_n((Z_ALPHA + z90) ** 2 * variance_sum / margin**2)
        ci_precision_n = _ceil_n((Z_ALPHA * sd_diff / margin) ** 2)
        seed = comparison_seed + int(round(margin_fraction * 1000))
        grid = sorted({10, 20, 30, 50, n, 2 * n, paired_n80, paired_n90})
        shifted = diff - observed_mean + direction * margin
        rng = np.random.default_rng(seed)
        for sample_n in grid:
            counts = rng.multinomial(sample_n, np.full(n, 1.0 / n), size=bootstrap_reps)
            means = counts @ shifted / sample_n
            second = counts @ (shifted * shifted) / sample_n
            variances = np.maximum(0.0, (second - means * means) * sample_n / (sample_n - 1))
            standard_errors = np.sqrt(variances / sample_n)
            z_scores = np.zeros_like(means)
            np.divide(np.abs(means), standard_errors, out=z_scores, where=standard_errors > 0)
            reject = np.where(standard_errors > 0, z_scores >= Z_ALPHA, np.abs(means) > 0)
            empirical_rows.append({
                "experiment": experiment,
                "scenario": scenario,
                "reference": reference_name,
                "variant": variant_name,
                "margin_fraction_T_end": margin_fraction,
                "margin_absolute": margin,
                "assumed_signed_effect": direction * margin,
                "sample_size_pairs": sample_n,
                "bootstrap_reps": bootstrap_reps,
                "seed": seed,
                "empirical_power": float(np.mean(reject)),
                "extrapolation_gt_4x_source_n": bool(sample_n > 4 * n),
            })
        normal_rows.append({
            "experiment": experiment,
            "scenario": scenario,
            "reference": reference_name,
            "variant": variant_name,
            "T_end": horizon,
            "observed_N_pairs": n,
            "margin_fraction_T_end": margin_fraction,
            "margin_absolute": margin,
            "mean_restricted_difference_variant_minus_reference": observed_mean,
            "sd_reference_restricted": sd_ref,
            "sd_variant_restricted": sd_var,
            "sd_paired_difference": sd_diff,
            "covariance_reference_variant": covariance,
            "correlation_reference_variant": correlation,
            "paired_normal_N80": paired_n80,
            "paired_normal_N90": paired_n90,
            "unpaired_normal_N80_per_group": unpaired_n80,
            "unpaired_normal_N90_per_group": unpaired_n90,
            "ci95_halfwidth_N": ci_precision_n,
            "pairing_benefit_N80_unpaired_over_paired": unpaired_n80 / paired_n80,
            "pairing_benefit_N90_unpaired_over_paired": unpaired_n90 / paired_n90,
            "paired_N80_extrapolation_gt_4x_source_N": bool(paired_n80 > 4 * n),
            "paired_N90_extrapolation_gt_4x_source_N": bool(paired_n90 > 4 * n),
            "unpaired_N80_extrapolation_gt_4x_source_N": bool(unpaired_n80 > 4 * n),
            "unpaired_N90_extrapolation_gt_4x_source_N": bool(unpaired_n90 > 4 * n),
            "ci_N_extrapolation_gt_4x_source_N": bool(ci_precision_n > 4 * n),
            "reference_completion_probability": float(np.mean(ref_complete)),
            "variant_completion_probability": float(np.mean(var_complete)),
            "reference_censoring_probability": float(1.0 - np.mean(ref_complete)),
            "variant_censoring_probability": float(1.0 - np.mean(var_complete)),
        })

    n01 = int(np.sum(~ref_complete & var_complete))
    n10 = int(np.sum(ref_complete & ~var_complete))
    binary_cov = float(np.cov(ref_complete.astype(float), var_complete.astype(float), ddof=1)[0, 1])
    binary = {
        "experiment": experiment,
        "scenario": scenario,
        "reference": reference_name,
        "variant": variant_name,
        "observed_N_pairs": n,
        "both_complete": int(np.sum(ref_complete & var_complete)),
        "reference_censored_variant_complete_n01": n01,
        "reference_complete_variant_censored_n10": n10,
        "both_censored": int(np.sum(~ref_complete & ~var_complete)),
        "discordant_total": n01 + n10,
        "discordance_probability": (n01 + n10) / n,
        "completion_difference_variant_minus_reference": float(np.mean(var_complete) - np.mean(ref_complete)),
        "completion_indicator_covariance": binary_cov,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(n01, n10),
    }
    return normal_rows, empirical_rows, binary


def _load_comparisons() -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    ce01_config = _load_json(CE01_DIR / "experiment_config.json")
    ce01 = pd.read_csv(CE01_DIR / "arrival_times.csv").rename(
        columns={"layer_index": "layer"})
    expected_ce01 = range(1, int(ce01_config["n_runs"]) + 1)
    reference_name = "FULL_MICRO_REFERENCE"
    variants = [name for name in ce01_config["model_variants"] if name != reference_name]
    for variant in variants:
        ref, var, rc, vc = _paired_terminal(
            ce01,
            id_column="replication_id",
            model_column="model_variant",
            reference=reference_name,
            variant=variant,
            terminal_layer=int(ce01_config["L"]) - 1,
            expected_ids=expected_ce01,
            horizon=float(ce01_config["T_end"]),
        )
        comparisons.append({
            "experiment": "CE-01",
            "scenario": "final_100",
            "reference_name": reference_name,
            "variant_name": variant,
            "horizon": float(ce01_config["T_end"]),
            "reference": ref,
            "variant": var,
            "ref_complete": rc,
            "var_complete": vc,
        })

    configs = {item["name"]: item for item in _load_json(CE06_DIR / "config.json")}
    cell_paths = sorted((CE06_DIR / "cells").glob("arrivals_*.csv"))
    if len(cell_paths) != len(configs):
        raise ValueError(f"CE-06 has {len(configs)} configured cells but {len(cell_paths)} arrival files")
    seen: set[str] = set()
    for path in cell_paths:
        frame = pd.read_csv(path)
        scenarios = frame["scenario"].dropna().unique()
        if len(scenarios) != 1:
            raise ValueError(f"expected one scenario in {path}, got {list(scenarios)}")
        scenario = str(scenarios[0])
        if scenario not in configs:
            raise ValueError(f"unconfigured CE-06 scenario {scenario} in {path}")
        cfg = configs[scenario]
        expected_ids = range(1, int(cfg["n_runs"]) + 1)
        ref, var, rc, vc = _paired_terminal(
            frame,
            id_column="replication_id",
            model_column="model",
            reference="Micro",
            variant="MicroMacro",
            terminal_layer=int(cfg["L"]) - 1,
            expected_ids=expected_ids,
            horizon=float(cfg["T_end"]),
        )
        comparisons.append({
            "experiment": "CE-06",
            "scenario": scenario,
            "reference_name": "Micro",
            "variant_name": "MicroMacro",
            "horizon": float(cfg["T_end"]),
            "reference": ref,
            "variant": var,
            "ref_complete": rc,
            "var_complete": vc,
        })
        seen.add(scenario)
    if seen != set(configs):
        raise ValueError(f"CE-06 scenario mismatch; missing={sorted(set(configs) - seen)}")
    return comparisons


def run(output_dir: Path, bootstrap_reps: int, resume: bool) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    inventory = _source_inventory()
    outputs = ["power_summary.csv", "empirical_power.csv", "completion_discordance.csv"]
    if resume and manifest_path.is_file():
        old = _load_json(manifest_path)
        if (
            old.get("complete") is True
            and old.get("bootstrap_reps") == bootstrap_reps
            and old.get("master_seed") == MASTER_SEED
            and _same_sources(old.get("sources"), inventory)
            and _outputs_match(output_dir, old, outputs)
        ):
            print(f"CE-12 already complete and sources unchanged: {output_dir}")
            return

    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "experiment": "CE-12 power calibration",
        "analysis_only": True,
        "complete": False,
        "started_utc": started,
        "bootstrap_reps": bootstrap_reps,
        "master_seed": MASTER_SEED,
        "alpha_two_sided": ALPHA,
        "effect_margin_fractions_T_end": list(MARGIN_FRACTIONS),
        "target_powers": list(POWERS),
        "sources": inventory,
        "outputs": outputs,
        "formulas": {
            "restricted_terminal": "X*=min(X,T_end) if observed by T_end, else T_end; D=Y*-X*; no censored pair is dropped",
            "paired_normal_N": "ceil(((z_(1-alpha/2)+z_power)*SD(D)/margin)^2)",
            "unpaired_normal_N_per_group": "ceil((z_(1-alpha/2)+z_power)^2*(Var(X*)+Var(Y*))/margin^2)",
            "ci95_halfwidth_N": "ceil((z_(1-alpha/2)*SD(D)/margin)^2)",
            "pairing_benefit": "unpaired N per group / paired N",
            "empirical_power": "paired blocks resampled with replacement; centered D shifted to signed margin; two-sided normal test",
            "completion_discordance": "paired 2x2 completion table plus exact two-sided McNemar/binomial test",
        },
    }
    _atomic_json(manifest_path, manifest)

    normal_rows: list[dict[str, Any]] = []
    empirical_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    comparisons = _load_comparisons()
    for index, item in enumerate(comparisons):
        normal, empirical, binary = _analyse_comparison(
            **item,
            bootstrap_reps=bootstrap_reps,
            comparison_seed=MASTER_SEED + index * 10_000,
        )
        normal_rows.extend(normal)
        empirical_rows.extend(empirical)
        binary_rows.append(binary)

    _write_csv(output_dir / outputs[0], normal_rows)
    _write_csv(output_dir / outputs[1], empirical_rows)
    _write_csv(output_dir / outputs[2], binary_rows)
    manifest.update({
        "complete": True,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_count": len(comparisons),
        "normal_summary_rows": len(normal_rows),
        "empirical_power_rows": len(empirical_rows),
        "completion_diagnostic_rows": len(binary_rows),
        "output_checksums": {name: _sha256(output_dir / name) for name in outputs},
    })
    _atomic_json(manifest_path, manifest)
    print(f"CE-12 complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CE-12 analysis-only paired power calibration from saved CE-01 and CE-06 arrivals."
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=500,
        help="block-resampling replicates per comparison, margin, and N (default: 500)",
    )
    parser.add_argument("--resume", action="store_true", help="skip when outputs are complete and inputs unchanged")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="destination directory")
    args = parser.parse_args()
    if args.bootstrap_reps < 50:
        parser.error("--bootstrap-reps must be at least 50")
    run(args.output_dir, args.bootstrap_reps, args.resume)


if __name__ == "__main__":
    main()
