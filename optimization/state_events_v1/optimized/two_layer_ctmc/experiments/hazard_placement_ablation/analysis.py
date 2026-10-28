from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .config import ModelVariant


def summarize(arrivals: pd.DataFrame, runs: pd.DataFrame, horizon: float, bootstrap_reps: int, seed: int):
    terminal = arrivals[arrivals["layer_index"] == arrivals["layer_index"].max()].copy()
    reference_name = ModelVariant.FULL_MICRO_REFERENCE.value
    reference = terminal[terminal["model_variant"] == reference_name].set_index("replication_id")
    rng = np.random.default_rng(seed)
    summary_rows = []
    bootstrap_rows = []

    for variant, group in terminal.groupby("model_variant", sort=False):
        sample = group.set_index("replication_id").reindex(reference.index)
        ref_t = reference["T_i"].to_numpy(float)
        var_t = sample["T_i"].to_numpy(float)
        ref_obs = ref_t[np.isfinite(ref_t)]
        var_obs = var_t[np.isfinite(var_t)]
        completion = np.isfinite(var_t).mean()
        ref_completion = np.isfinite(ref_t).mean()
        restricted_ref = np.where(np.isfinite(ref_t), ref_t, horizon)
        restricted_var = np.where(np.isfinite(var_t), var_t, horizon)
        rmst_shift = float(np.mean(restricted_var - restricted_ref))
        w1 = wasserstein_distance(ref_obs, var_obs) if len(ref_obs) and len(var_obs) else np.nan
        rel_w1 = w1 / np.mean(ref_obs) if len(ref_obs) and np.mean(ref_obs) else np.nan
        summary_rows.append({
            "model_variant": variant,
            "completion_probability": completion,
            "completion_difference_vs_micro": completion - ref_completion,
            "completed_mean_T_last": np.mean(var_obs) if len(var_obs) else np.nan,
            "completed_median_T_last": np.median(var_obs) if len(var_obs) else np.nan,
            "W1_T_last_vs_micro": w1,
            "relative_W1_T_last_vs_micro": rel_w1,
            "RMST": np.mean(restricted_var),
            "delta_RMST_vs_micro": rmst_shift,
            "runtime_ratio_micro_over_variant": _runtime_ratio(runs, reference_name, variant),
        })

        boot = []
        n = len(reference)
        for _ in range(bootstrap_reps):
            idx = rng.integers(0, n, n)
            rt, vt = ref_t[idx], var_t[idx]
            rr = np.where(np.isfinite(rt), rt, horizon)
            vr = np.where(np.isfinite(vt), vt, horizon)
            boot.append(np.mean(vr - rr))
        low, high = np.quantile(boot, [0.025, 0.975])
        bootstrap_rows.append({
            "model_variant": variant,
            "metric": "delta_RMST_vs_micro",
            "estimate": rmst_shift,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_reps": bootstrap_reps,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def layer_metrics(arrivals: pd.DataFrame) -> pd.DataFrame:
    reference_name = ModelVariant.FULL_MICRO_REFERENCE.value
    rows: List[dict] = []
    reference = arrivals[arrivals["model_variant"] == reference_name]
    for variant, variant_data in arrivals.groupby("model_variant", sort=False):
        for layer, ref_layer in reference.groupby("layer_index"):
            var_layer = variant_data[variant_data["layer_index"] == layer]
            merged = ref_layer[["replication_id", "T_i"]].merge(
                var_layer[["replication_id", "T_i"]], on="replication_id", suffixes=("_micro", "_variant")
            )
            ref_obs = merged["T_i_micro"].dropna().to_numpy(float)
            var_obs = merged["T_i_variant"].dropna().to_numpy(float)
            paired = merged.dropna()
            w1 = wasserstein_distance(ref_obs, var_obs) if len(ref_obs) and len(var_obs) else np.nan
            rows.append({
                "model_variant": variant,
                "layer_index": int(layer),
                "W1_T_i": w1,
                "relative_W1_T_i": w1 / np.mean(ref_obs) if len(ref_obs) and np.mean(ref_obs) else np.nan,
                "mean_shift_T_i": np.mean(var_obs) - np.mean(ref_obs) if len(ref_obs) and len(var_obs) else np.nan,
                "median_shift_T_i": np.median(var_obs) - np.median(ref_obs) if len(ref_obs) and len(var_obs) else np.nan,
                "paired_error_mean": (paired["T_i_variant"] - paired["T_i_micro"]).mean() if len(paired) else np.nan,
                "paired_error_median": (paired["T_i_variant"] - paired["T_i_micro"]).median() if len(paired) else np.nan,
            })
    return pd.DataFrame(rows)


def delay_metrics(delays: pd.DataFrame) -> pd.DataFrame:
    reference_name = ModelVariant.FULL_MICRO_REFERENCE.value
    rows = []
    reference = delays[delays["model_variant"] == reference_name]
    for variant, variant_data in delays.groupby("model_variant", sort=False):
        for layer, ref_layer in reference.groupby("layer_index"):
            ref_obs = ref_layer["Delta_T_i"].dropna().to_numpy(float)
            var_obs = variant_data[variant_data["layer_index"] == layer]["Delta_T_i"].dropna().to_numpy(float)
            w1 = wasserstein_distance(ref_obs, var_obs) if len(ref_obs) and len(var_obs) else np.nan
            rows.append({
                "model_variant": variant,
                "layer_index": int(layer),
                "W1_Delta_T_i": w1,
                "mean_shift_Delta_T_i": np.mean(var_obs) - np.mean(ref_obs) if len(ref_obs) and len(var_obs) else np.nan,
                "median_shift_Delta_T_i": np.median(var_obs) - np.median(ref_obs) if len(ref_obs) and len(var_obs) else np.nan,
            })
    return pd.DataFrame(rows)


def _runtime_ratio(runs: pd.DataFrame, reference: str, variant: str) -> float:
    ref = runs[runs["model_variant"] == reference]["runtime_seconds"].sum()
    other = runs[runs["model_variant"] == variant]["runtime_seconds"].sum()
    return float(ref / other) if other else np.nan
