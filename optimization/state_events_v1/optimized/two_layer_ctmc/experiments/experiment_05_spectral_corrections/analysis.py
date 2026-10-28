"""Replication-level analysis for Experiment 05.

The statistical unit in every confidence interval in this module is a
replication.  Event rows are deliberately not accepted as bootstrap units.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REFERENCE_VARIANT = "V0"
CORRECTED_VARIANTS = ("V1", "V2", "V3", "V4")
REQUIRED_KEYS = ("scenario_id", "variant_id", "replication_id")


def empirical_wasserstein_1(left: np.ndarray, right: np.ndarray) -> float:
    """Exact one-dimensional W1 for equally weighted empirical samples."""
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if not left.size or not right.size:
        raise ValueError("W1 requires two non-empty samples")
    support = np.sort(np.concatenate((left, right)))
    if support.size < 2:
        return 0.0
    cdf_left = np.searchsorted(left, support[:-1], side="right") / left.size
    cdf_right = np.searchsorted(right, support[:-1], side="right") / right.size
    return float(np.sum(np.abs(cdf_left - cdf_right) * np.diff(support)))


@dataclass(frozen=True)
class AnalysisOutputs:
    terminal_metrics: pd.DataFrame
    layer_metrics: pd.DataFrame
    paired_errors: pd.DataFrame
    interaction_contrast: pd.DataFrame


def _first_present(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _parse_vector(value: object) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("arrival vector must be a JSON array or a sequence")
    return [float(item) for item in value]


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"invalid boolean value {value!r}")
    return bool(value)


def _parse_observed_vector(value: object) -> list[bool]:
    """Parse an explicit per-layer observation vector.

    Replication summaries persist this field as JSON, while callers of the
    in-memory analysis API may provide an ordinary sequence.  Keeping this
    separate from the arrival parser makes the precedence rule explicit:
    when an observation vector is present, censoring is not inferred from the
    numerical arrival time.
    """
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("arrival_observed must be a JSON array or a sequence")
    return [_as_bool(item) for item in value]


def normalize_replication_table(frame: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """Return canonical long arrival rows from a replication summary table.

    Accepted arrival encodings are ``arrival_times``/``arrivals`` JSON arrays,
    or wide columns named ``T_0``, ..., ``T_{L-1}``.  Optional observed flags
    can be encoded similarly as ``arrival_observed`` or ``observed_0``, ... .
    Without them an arrival strictly below the horizon is considered observed;
    terminal-layer ``completion`` is authoritative when available.
    """
    aliases = {
        "scenario": "scenario_id", "model_variant": "variant_id",
        "variant": "variant_id", "replication": "replication_id",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns}).copy()
    missing = [column for column in REQUIRED_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"replication table missing columns: {missing}")
    if frame.duplicated(list(REQUIRED_KEYS)).any():
        raise ValueError("replication table has duplicate scenario/variant/replication rows")

    arrival_vector = _first_present(frame, ("arrival_times", "arrivals", "T_i"))
    observed_vector = _first_present(frame, ("arrival_observed", "observed_arrivals"))
    wide = sorted(
        ((int(c[2:]), c) for c in frame.columns if c.startswith("T_") and c[2:].isdigit()),
        key=lambda item: item[0],
    )
    if arrival_vector is None and not wide:
        raise ValueError("replication table needs arrival_times JSON or T_0...T_n columns")

    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        arrivals = _parse_vector(record[arrival_vector]) if arrival_vector else [float(record[c]) for _, c in wide]
        if observed_vector:
            # Explicit flags are authoritative, including the uncommon case
            # of an observed arrival exactly at the analysis horizon.
            observed = _parse_observed_vector(record[observed_vector])
        else:
            observed = [bool(np.isfinite(t) and t < horizon) for t in arrivals]
            for layer, _ in wide:
                flag = f"observed_{layer}"
                if flag in record and pd.notna(record[flag]):
                    observed[layer] = _as_bool(record[flag])
        if len(arrivals) != len(observed):
            raise ValueError("arrival and observed vectors have different lengths")
        completion_col = _first_present(frame, ("completion", "completed", "completion_flag"))
        if completion_col and arrivals:
            observed[-1] = _as_bool(record[completion_col])
        for layer, (arrival, is_observed) in enumerate(zip(arrivals, observed)):
            if not np.isfinite(arrival):
                raise ValueError(f"non-finite arrival for {record['scenario_id']}/{record['variant_id']}")
            rows.append({
                "scenario_id": str(record["scenario_id"]),
                "variant_id": str(record["variant_id"]),
                "replication_id": int(record["replication_id"]),
                "layer_index": layer,
                "arrival_time": min(float(arrival), float(horizon)),
                "observed": bool(is_observed),
                "runtime_seconds": record.get("runtime_seconds", record.get("runtime", np.nan)),
            })
    result = pd.DataFrame(rows)
    counts = result.groupby(list(REQUIRED_KEYS))["layer_index"].max()
    if counts.nunique() != 1:
        raise ValueError("all replications must contain the same number of layers")
    return result


def _metric(value: float | None, status: str = "ok") -> tuple[float | None, str]:
    if value is None or not np.isfinite(value):
        return None, status if status != "ok" else "not_estimable"
    return float(value), "ok"


def _paired_terminal(long: pd.DataFrame, scenario: str, variant: str) -> pd.DataFrame:
    terminal_layer = int(long["layer_index"].max())
    columns = ["replication_id", "arrival_time", "observed"]
    ref = long[(long.scenario_id == scenario) & (long.variant_id == REFERENCE_VARIANT) & (long.layer_index == terminal_layer)][columns]
    var = long[(long.scenario_id == scenario) & (long.variant_id == variant) & (long.layer_index == terminal_layer)][columns]
    if set(ref.replication_id) != set(var.replication_id):
        raise ValueError(f"unpaired replication IDs for {scenario}/{variant}")
    return ref.merge(var, on="replication_id", suffixes=("_ref", "_variant"), validate="one_to_one")


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        sample = rng.integers(0, values.size, values.size)
        estimates[index] = values[sample].mean()
    return tuple(np.quantile(estimates, (0.025, 0.975)))


def _bootstrap_w1(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, reps: int,
) -> tuple[float, float]:
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        sample = rng.integers(0, left.size, left.size)
        estimates[index] = empirical_wasserstein_1(left[sample], right[sample])
    return tuple(np.quantile(estimates, (0.025, 0.975)))


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return centre - radius, centre + radius


def terminal_metrics(
    long: pd.DataFrame, horizon: float, bootstrap_reps: int, rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    terminal_layer = int(long.layer_index.max())
    for scenario in sorted(long.scenario_id.unique()):
        scenario_data = long[(long.scenario_id == scenario) & (long.layer_index == terminal_layer)]
        ref_data = scenario_data[scenario_data.variant_id == REFERENCE_VARIANT]
        ref_completed = ref_data.loc[ref_data.observed, "arrival_time"].to_numpy(float)
        ref_mean = ref_completed.mean() if ref_completed.size else np.nan
        ref_all = ref_data.sort_values("replication_id").arrival_time.to_numpy(float)
        v1_data = scenario_data[scenario_data.variant_id == "V1"].sort_values("replication_id")
        v1_all = v1_data.arrival_time.to_numpy(float)
        v1_bias = float(np.mean(v1_all - ref_all)) if len(v1_all) == len(ref_all) else np.nan
        variants = [v for v in (REFERENCE_VARIANT, *CORRECTED_VARIANTS) if v in set(scenario_data.variant_id)]
        for variant in variants:
            variant_data = scenario_data[scenario_data.variant_id == variant]
            completed = variant_data.loc[variant_data.observed, "arrival_time"].to_numpy(float)
            if ref_completed.size and completed.size and np.isfinite(ref_mean) and ref_mean > 0:
                w1 = empirical_wasserstein_1(ref_completed, completed)
                rel_w1, w1_status = _metric(w1 / ref_mean)
            else:
                w1, rel_w1, w1_status = None, None, "no_completed_arrivals"
            paired = _paired_terminal(long, scenario, variant)
            delta = paired.arrival_time_variant.to_numpy(float) - paired.arrival_time_ref.to_numpy(float)
            rmst, rmst_status = _metric(delta.mean() if delta.size else None, "no_paired_replications")
            ci_low, ci_high = _bootstrap_mean(delta, rng, bootstrap_reps)
            ci_low, _ = _metric(ci_low, rmst_status)
            ci_high, _ = _metric(ci_high, rmst_status)
            epsilon = 0.05 * ref_mean if np.isfinite(ref_mean) else np.nan
            equivalent = bool(np.isfinite(epsilon) and ci_low is not None and ci_high is not None and ci_low >= -epsilon and ci_high <= epsilon)
            restricted_ref = paired.arrival_time_ref.to_numpy(float)
            restricted_variant = paired.arrival_time_variant.to_numpy(float)
            restricted_w1 = empirical_wasserstein_1(restricted_ref, restricted_variant)
            restricted_w1_low, restricted_w1_high = _bootstrap_w1(
                restricted_ref, restricted_variant, rng, bootstrap_reps)
            completion_low, completion_high = _wilson_interval(
                int(variant_data.observed.sum()), len(variant_data))
            delta_vs_v1 = (float(np.mean(restricted_variant - v1_all))
                           if len(v1_all) == len(restricted_variant) else np.nan)
            absolute_error_change = (abs(float(delta.mean())) - abs(v1_bias)
                                     if np.isfinite(v1_bias) else np.nan)
            runtime = pd.to_numeric(variant_data.runtime_seconds, errors="coerce")
            rows.append({
                "scenario_id": scenario, "variant_id": variant,
                "n_replications": int(len(variant_data)),
                "n_completed": int(variant_data.observed.sum()),
                "completion_probability_all_runs": float(variant_data.observed.mean()) if len(variant_data) else None,
                "completion_ci95_low": completion_low,
                "completion_ci95_high": completion_high,
                "mean_T_last_completed": float(completed.mean()) if completed.size else None,
                "W1_T_last_completed": w1,
                "relative_W1_T_last_completed": rel_w1,
                "W1_T_last_restricted_all_runs": restricted_w1,
                "W1_T_last_restricted_ci95_low": restricted_w1_low,
                "W1_T_last_restricted_ci95_high": restricted_w1_high,
                "delta_RMST_all_runs": rmst,
                "delta_RMST_ci95_low": ci_low,
                "delta_RMST_ci95_high": ci_high,
                "delta_RMST_vs_V1_all_runs": delta_vs_v1,
                "absolute_RMST_error_change_vs_V1": absolute_error_change,
                "equivalence_bound": float(epsilon) if np.isfinite(epsilon) else None,
                "rmst_equivalent": equivalent,
                "runtime_seconds_median": float(runtime.median()) if runtime.notna().any() else None,
                "metric_status": w1_status if w1_status != "ok" else rmst_status,
                "bootstrap_reps": bootstrap_reps,
            })
    return pd.DataFrame(rows)


def layer_metrics(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in sorted(long.scenario_id.unique()):
        scenario_data = long[long.scenario_id == scenario]
        for layer in sorted(scenario_data.layer_index.unique()):
            ref = scenario_data[(scenario_data.variant_id == REFERENCE_VARIANT) & (scenario_data.layer_index == layer)]
            ref_completed = ref.loc[ref.observed, "arrival_time"].to_numpy(float)
            ref_mean = ref_completed.mean() if ref_completed.size else np.nan
            ref_all = ref.arrival_time.to_numpy(float)
            ref_all_mean = ref_all.mean() if ref_all.size else np.nan
            for variant in CORRECTED_VARIANTS:
                var = scenario_data[(scenario_data.variant_id == variant) & (scenario_data.layer_index == layer)]
                if var.empty:
                    continue
                completed = var.loc[var.observed, "arrival_time"].to_numpy(float)
                if ref_completed.size and completed.size:
                    w1 = empirical_wasserstein_1(ref_completed, completed)
                    rel, status = _metric(w1 / ref_mean if ref_mean > 0 else (0.0 if w1 == 0 else None))
                else:
                    w1, rel, status = None, None, "no_completed_arrivals"
                var_all = var.arrival_time.to_numpy(float)
                restricted_w1 = (empirical_wasserstein_1(ref_all, var_all)
                                 if ref_all.size and var_all.size else None)
                restricted_rel = (restricted_w1 / ref_all_mean
                                  if restricted_w1 is not None and ref_all_mean > 0 else None)
                rows.append({
                    "scenario_id": scenario, "variant_id": variant, "layer_index": int(layer),
                    "W1_T_i_completed": w1, "relative_W1_T_i_completed": rel,
                    "W1_T_i_restricted_all_runs": restricted_w1,
                    "relative_W1_T_i_restricted_all_runs": restricted_rel,
                    "n_reference_completed": int(ref_completed.size), "n_variant_completed": int(completed.size),
                    "metric_status": status,
                })
    return pd.DataFrame(rows)


def paired_error_metrics(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["replication_id", "layer_index", "arrival_time"]
    for scenario in sorted(long.scenario_id.unique()):
        ref = long[(long.scenario_id == scenario) & (long.variant_id == REFERENCE_VARIANT)][keys]
        for variant in CORRECTED_VARIANTS:
            var = long[(long.scenario_id == scenario) & (long.variant_id == variant)][keys]
            if var.empty:
                continue
            paired = ref.merge(var, on=["replication_id", "layer_index"], suffixes=("_ref", "_variant"), validate="one_to_one")
            paired["cumulative_error"] = paired.arrival_time_variant - paired.arrival_time_ref
            paired["local_delay_ref"] = paired.groupby("replication_id").arrival_time_ref.diff()
            paired["local_delay_variant"] = paired.groupby("replication_id").arrival_time_variant.diff()
            paired["local_delay_shift"] = paired.local_delay_variant - paired.local_delay_ref
            for layer, group in paired.groupby("layer_index", sort=True):
                cumulative = group.cumulative_error.to_numpy(float)
                local = group.local_delay_shift.dropna().to_numpy(float)
                rows.append({
                    "scenario_id": scenario, "variant_id": variant, "layer_index": int(layer),
                    "cumulative_error_median_all_runs": float(np.median(cumulative)),
                    "cumulative_error_q25_all_runs": float(np.quantile(cumulative, 0.25)),
                    "cumulative_error_q75_all_runs": float(np.quantile(cumulative, 0.75)),
                    "local_delay_shift_median_all_runs": float(np.median(local)) if local.size else None,
                    "local_delay_shift_q25_all_runs": float(np.quantile(local, 0.25)) if local.size else None,
                    "local_delay_shift_q75_all_runs": float(np.quantile(local, 0.75)) if local.size else None,
                    "n_paired": int(cumulative.size), "metric_status": "ok",
                })
    return pd.DataFrame(rows)


def factorial_interaction(
    long: pd.DataFrame, bootstrap_reps: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """Estimate the H1-by-H2 interaction on all-run restricted arrival times.

    The replication contrast is V4 - V2 - V3 + V1.  Zero is additive.
    """
    rows: list[dict[str, object]] = []
    terminal = long[long.layer_index == long.layer_index.max()]
    for scenario, group in terminal.groupby("scenario_id", sort=True):
        pivot = group.pivot(index="replication_id", columns="variant_id", values="arrival_time")
        if not set(CORRECTED_VARIANTS).issubset(pivot.columns):
            rows.append({"scenario_id": scenario, "interaction_estimate_all_runs": None,
                         "ci95_low": None, "ci95_high": None, "n_paired": 0,
                         "metric_status": "missing_factorial_cell", "bootstrap_reps": bootstrap_reps})
            continue
        complete = pivot[list(CORRECTED_VARIANTS)].dropna()
        contrast = (complete.V4 - complete.V2 - complete.V3 + complete.V1).to_numpy(float)
        estimate, status = _metric(contrast.mean() if contrast.size else None, "no_paired_replications")
        low, high = _bootstrap_mean(contrast, rng, bootstrap_reps)
        rows.append({"scenario_id": scenario, "interaction_estimate_all_runs": estimate,
                     "ci95_low": _metric(low, status)[0], "ci95_high": _metric(high, status)[0],
                     "n_paired": int(contrast.size), "metric_status": status,
                     "bootstrap_reps": bootstrap_reps})
    return pd.DataFrame(rows)


def analyze_replications(
    frame: pd.DataFrame, horizon: float, bootstrap_reps: int = 1000, bootstrap_seed: int = 0,
) -> tuple[pd.DataFrame, AnalysisOutputs]:
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    long = normalize_replication_table(frame, horizon)
    rng = np.random.default_rng(bootstrap_seed)
    outputs = AnalysisOutputs(
        terminal_metrics(long, horizon, bootstrap_reps, rng),
        layer_metrics(long), paired_error_metrics(long),
        factorial_interaction(long, bootstrap_reps, rng),
    )
    return long, outputs


def write_analysis(
    replication_csv: Path, summary_dir: Path, horizon: float,
    bootstrap_reps: int = 1000, bootstrap_seed: int = 0,
    max_replication_id: int | None = None,
) -> tuple[pd.DataFrame, AnalysisOutputs]:
    frame = pd.read_csv(replication_csv)
    if max_replication_id is not None:
        frame = frame[frame["replication_id"].between(1, max_replication_id)].copy()
        expected = set(range(1, max_replication_id + 1))
        observed = frame.groupby(["scenario_id", "variant_id"])["replication_id"].apply(set)
        incomplete = [
            f"{scenario}/{variant}"
            for (scenario, variant), ids in observed.items()
            if ids != expected
        ]
        if incomplete:
            raise ValueError(
                "prespecified replication window is incomplete for: " + ", ".join(incomplete)
            )
    long, outputs = analyze_replications(frame, horizon, bootstrap_reps, bootstrap_seed)
    summary_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(summary_dir / "arrivals_long.csv", index=False)
    outputs.terminal_metrics.to_csv(summary_dir / "terminal_metrics.csv", index=False)
    outputs.layer_metrics.to_csv(summary_dir / "layer_wise.csv", index=False)
    outputs.paired_errors.to_csv(summary_dir / "paired_errors.csv", index=False)
    outputs.interaction_contrast.to_csv(summary_dir / "interaction_contrast.csv", index=False)
    return long, outputs
