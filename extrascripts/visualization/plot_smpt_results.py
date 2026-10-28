#!/usr/bin/env python3
"""Create the non-clique figures specified in ``RESULTS_ANALYSIS_PLAN.md``.

The clique--bridge--leaf figure is deliberately *not* regenerated here: its
published, validated version is retained as the first Results figure.  This
script reads only the canonical checkpoint files in ``output/data`` and never
imports or runs a simulator.  Incomplete cells are analysed as clearly labelled
previews, with an explicit terminal warning and validation record.

Outputs are written to ``output/analysis/smpt_results_v2``.  The primary
arrival estimand at horizon tau is Y=min(T_last,tau): rW1 and rKS are therefore
explicitly restricted distances whenever a cell has censoring.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extrascripts.runners.checkpoint_io import (  # noqa: E402
    has_trajectories, model_trajectory_columns, read_checkpoint,
)

FULL = "Full Micro"
MM = "Production MicroMacro + independent RNG"
MODELS = (FULL, MM)
COLORS = {FULL: "#2878b5", MM: "#d0604c"}
LABELS = {FULL: "Full Micro", MM: "MicroMacro"}
SIZES = (50, 100, 250, 1000)
K_VALUES = (1, 5, 10)
DEGREES = (10, 6)  # visual order: dense sparse row before sparse row
BETAS = (0.001, 0.003)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "analysis" / "smpt_results_v2")
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS),
                        help="Rates to analyse together; default: 0.001 0.003.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Backward-compatible single-rate shortcut; overrides --betas.")
    parser.add_argument("--detail-beta", type=float, default=0.003,
                        help="Rate used for layer-accumulation and aligned-trajectory figures.")
    parser.add_argument("--tau", type=float, default=20000.0,
                        help="Common administrative horizon for all restricted metrics.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Deprecated compatibility flag; partial cells are now always plotted and labelled.")
    parser.add_argument("--skip-aligned", action="store_true",
                        help="Do not decode trajectories or create the aligned-local-dynamics figure.")
    parser.add_argument("--include-profiles", action="store_true",
                        help=("Also generate redundant one-parameter profile diagnostics. "
                              "They are omitted by default because the accuracy atlases already "
                              "show the complete grid."))
    return parser.parse_args()


def canonical_config(path: Path) -> dict | None:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest.get("benchmark_config")


def scenario_key(config: dict) -> tuple:
    micro = config.get("micro_graph_type")
    degree = None if micro == "complete" else int(round(float(config["mean_internal_degree"])))
    return (float(config["beta"]), micro, degree, int(config["community_size"]), int(config["inter_links"]))


def discover(data_root: Path, betas: tuple[float, ...]) -> dict[tuple, Path]:
    """Discover canonical L=10 chain-grid cells by manifest, never folder name."""
    found: dict[tuple, Path] = {}
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or not (folder / "checkpoints").is_dir():
            continue
        config = canonical_config(folder)
        if not config or config.get("family") != "custom_grid":
            continue
        if (config.get("communities") != 10 or config.get("macro_graph_type") != "chain"
                or not any(math.isclose(float(config.get("beta", math.nan)), beta) for beta in betas)
                or int(config.get("community_size", -1)) not in SIZES
                or int(config.get("inter_links", -1)) not in K_VALUES):
            continue
        micro = config.get("micro_graph_type")
        if micro == "complete":
            key = scenario_key(config)
        elif micro == "random_connected" and int(round(float(config.get("mean_internal_degree", -1)))) in DEGREES:
            key = scenario_key(config)
        else:
            continue
        if key in found:
            raise RuntimeError(f"Duplicate canonical scenario for {key}: {found[key]} and {folder}")
        found[key] = folder
    expected = {(beta, "complete", None, n, k) for beta in betas for n in SIZES for k in K_VALUES}
    expected |= {(beta, "random_connected", d, n, k) for beta in betas for d in DEGREES for n in SIZES for k in K_VALUES}
    missing = expected - set(found)
    if missing:
        print(f"[PREVIEW: missing cells] {len(missing)} of {36 * len(betas)} canonical cells are absent; "
              f"first: {sorted(missing)[:4]}", flush=True)
    if not found:
        raise RuntimeError("No canonical custom-grid cells were found.")
    return found


def checkpoint_id(path: Path) -> int:
    return int(path.name.removesuffix(".json.gz").rsplit("_", 1)[1])


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


CONFIG_FIELDS = ("communities", "community_size", "inter_links", "macro_graph_type",
                 "micro_graph_type", "beta", "gamma", "model", "tau_micro",
                 "macro_T", "initial_node", "mean_internal_degree")


def extract_replication(path: Path, tau: float, retain_payload: bool, expected_config: dict) -> tuple[dict, dict]:
    """Read one paired checkpoint and return restricted terminal/layer data."""
    payload = read_checkpoint(path)
    run_rows = payload.get("runs", [])
    if len(run_rows) != 2 or sorted(row.get("simulator") for row in run_rows) != sorted(MODELS):
        raise ValueError("missing or duplicate model run rows")
    runs = {row["simulator"]: row for row in run_rows}
    if runs[FULL].get("network_seed") != runs[MM].get("network_seed"):
        raise ValueError("Full/MM network seeds differ")
    if runs[FULL].get("initial_node") != runs[MM].get("initial_node"):
        raise ValueError("Full/MM initial nodes differ")
    payload_config = payload.get("benchmark_config", {})
    for field in CONFIG_FIELDS:
        expected = expected_config.get(field)
        actual = payload_config.get(field, expected)
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            equal = math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        else:
            equal = expected == actual
        if not equal:
            raise ValueError(f"checkpoint config mismatch for {field}: {actual!r} != {expected!r}")
    arrivals = pd.DataFrame(payload.get("community_arrivals", []))
    if arrivals.empty:
        raise ValueError("missing community arrivals")
    rep = int(payload.get("replication_id", runs[FULL].get("replication_id")))
    rows: list[dict] = []
    for community in range(10):
        item = {"replication_id": rep, "community": community, "path": str(path)}
        for model in MODELS:
            records = arrivals[(arrivals["simulator"] == model) & (arrivals["community"] == community)]
            if len(records) != 1:
                raise ValueError(f"expected one arrival for {model}, community {community}")
            value = records.iloc[0]["first_infection_time"]
            horizon = float(runs[model].get("T_end", tau))
            observed = finite(value)
            if not observed and horizon + 1e-9 < tau:
                raise ValueError(f"censored before requested tau ({horizon:g} < {tau:g})")
            item[f"time_{'full' if model == FULL else 'mm'}"] = min(float(value), tau) if observed else tau
            item[f"event_{'full' if model == FULL else 'mm'}"] = bool(observed and float(value) <= tau)
            item[f"horizon_{'full' if model == FULL else 'mm'}"] = horizon
        rows.append(item)
    meta = {"replication_id": rep, "path": str(path), "network_seed": runs[FULL].get("network_seed"),
            "event_seed_full": runs[FULL].get("event_seed"), "event_seed_mm": runs[MM].get("event_seed"),
            "horizon_full": float(runs[FULL].get("T_end", tau)),
            "horizon_mm": float(runs[MM].get("T_end", tau)),
            "completion_full": bool(runs[FULL].get("completion")),
            "completion_mm": bool(runs[MM].get("completion")),
            "has_trajectories": has_trajectories(payload)}
    # A compact trajectory payload can be much larger than all observables.
    # Retain it only for the two cases used by the arrival-aligned figure.
    return meta, {"layers": rows, "payload": payload if retain_payload else None}


def load_cell(folder: Path, tau: float, allow_partial: bool, retain_payload: bool,
              expected_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    validation, layer_rows, payloads, manifest_rows = [], [], [], []
    seen: set[int] = set()
    for path in sorted((folder / "checkpoints").glob("rep_*.json.gz"), key=checkpoint_id):
        rep = checkpoint_id(path)
        try:
            meta, extracted = extract_replication(path, tau, retain_payload, expected_config)
            if meta["replication_id"] != rep or rep in seen:
                raise ValueError("replication ID mismatch or duplicate")
            seen.add(rep)
            validation.append({"scenario": folder.name, "replication_id": rep, "valid": True, "reason": "",
                               "network_seed": meta["network_seed"], "horizon_full": meta["horizon_full"],
                               "horizon_mm": meta["horizon_mm"], "completion_full": meta["completion_full"],
                               "completion_mm": meta["completion_mm"],
                               "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
            layer_rows.extend(extracted["layers"])
            if retain_payload:
                payloads.append({"replication_id": rep, "payload": extracted["payload"]})
            manifest_rows.append(meta)
        except Exception as exc:  # report corrupt/incompatible checkpoint instead of silently dropping it
            validation.append({"scenario": folder.name, "replication_id": rep, "valid": False, "reason": str(exc),
                               "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    valid = sum(row["valid"] for row in validation)
    if valid != 100:
        print(f"[PREVIEW: incomplete] {folder.name}: {valid}/100 valid paired checkpoints; "
              "metrics and figures use only available pairs.", flush=True)
    inadequate_followup = any("censored before requested tau" in row["reason"] for row in validation)
    if inadequate_followup:
        print(f"[skip cell: insufficient follow-up] {folder.name}: at least one censored pair ends before tau={tau:g}; "
              "dropping only that pair would select on the outcome.", flush=True)
        return pd.DataFrame(), pd.DataFrame(), validation, []
    if valid < 2:
        print(f"[skip cell] {folder.name}: fewer than two valid pairs", flush=True)
        return pd.DataFrame(), pd.DataFrame(), validation, []
    layers = pd.DataFrame(layer_rows)
    return layers[layers.community == 9].copy(), layers, validation, payloads


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    points = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, points, side="right") / len(a)
                               - np.searchsorted(b, points, side="right") / len(b))))


def metric_values(full: np.ndarray, mm: np.ndarray, event_full: np.ndarray, event_mm: np.ndarray) -> dict:
    categories = {"both": int(np.sum(event_full & event_mm)), "full_only": int(np.sum(event_full & ~event_mm)),
                  "mm_only": int(np.sum(~event_full & event_mm)), "neither": int(np.sum(~event_full & ~event_mm))}
    rmst_full, rmst_mm = float(full.mean()), float(mm.mean())
    w1 = float(wasserstein_distance(full, mm))
    return {"N_pairs": len(full), "p_full": float(event_full.mean()), "p_mm": float(event_mm.mean()),
            "delta_completion": float(event_mm.mean() - event_full.mean()), **categories,
            "rmst_full": rmst_full, "rmst_mm": rmst_mm, "delta_rmst": rmst_mm - rmst_full,
            "relative_delta_rmst": (rmst_mm - rmst_full) / rmst_full if rmst_full else math.nan,
            "restricted_w1": w1,
            "relative_restricted_w1": float(w1 / rmst_full) if rmst_full else math.nan,
            "restricted_ks": ks_distance(full, mm), "all_complete": bool(event_full.all() and event_mm.all())}


def bootstrap(full: np.ndarray, mm: np.ndarray, ef: np.ndarray, em: np.ndarray, count: int,
              rng: np.random.Generator,
              fields: tuple[str, ...] = ("delta_rmst", "relative_delta_rmst", "delta_completion",
                                         "restricted_w1", "relative_restricted_w1", "restricted_ks")) -> dict:
    """Paired bootstrap, vectorized wherever the statistic permits it.

    For equally sized one-dimensional samples W1 is the mean absolute gap
    between their ordered empirical quantiles.  This produces the same W1 as
    ``scipy.stats.wasserstein_distance`` but avoids thousands of Python calls.
    KS remains a small per-resample operation.  Layerwise figures only request
    the signed-shift interval, so they do not compute unused W1/KS intervals.
    """
    n = len(full)
    indices = rng.integers(0, n, size=(count, n))
    draws: dict[str, np.ndarray] = {}
    full_mean = full[indices].mean(axis=1)
    delta = (mm[indices] - full[indices]).mean(axis=1)
    if "delta_rmst" in fields:
        draws["delta_rmst"] = delta
    if "relative_delta_rmst" in fields:
        draws["relative_delta_rmst"] = np.divide(
            delta, full_mean, out=np.full(count, np.nan), where=full_mean != 0)
    if "delta_completion" in fields:
        draws["delta_completion"] = (em[indices].mean(axis=1) - ef[indices].mean(axis=1))
    if "restricted_w1" in fields or "relative_restricted_w1" in fields:
        w1 = np.abs(np.sort(full[indices], axis=1) - np.sort(mm[indices], axis=1)).mean(axis=1)
        if "restricted_w1" in fields:
            draws["restricted_w1"] = w1
        if "relative_restricted_w1" in fields:
            draws["relative_restricted_w1"] = np.divide(
                w1, full_mean, out=np.full(count, np.nan), where=full_mean != 0)
    if "restricted_ks" in fields:
        draws["restricted_ks"] = np.fromiter(
            (ks_distance(np.sort(full[index]), np.sort(mm[index])) for index in indices), float, count=count)
    return {f"{key}_ci_low": float(np.quantile(values, .025)) for key, values in draws.items()} | {
        f"{key}_ci_high": float(np.quantile(values, .975)) for key, values in draws.items()}


def cell_metrics(terminal: pd.DataFrame, config: dict, tau: float, boot: int, rng: np.random.Generator) -> dict:
    terminal = terminal.sort_values("replication_id")
    f, m = terminal.time_full.to_numpy(float), terminal.time_mm.to_numpy(float)
    ef, em = terminal.event_full.to_numpy(bool), terminal.event_mm.to_numpy(bool)
    estimates = metric_values(f, m, ef, em)
    return {"scenario": config["scenario"], "micro_graph_type": config["micro_graph_type"],
            "mean_internal_degree": config["mean_internal_degree"], "community_size": config["community_size"],
            "inter_links": config["inter_links"], "beta": config["beta"], "tau": tau,
            "metric_kind": "ordinary" if estimates["all_complete"] else "restricted",
            **estimates, **bootstrap(f, m, ef, em, boot, rng)}


def layer_metrics(layers: pd.DataFrame, config: dict, tau: float) -> list[dict]:
    result = []
    for community, part in layers.groupby("community", sort=True):
        m = metric_values(part.time_full.to_numpy(float), part.time_mm.to_numpy(float),
                          part.event_full.to_numpy(bool), part.event_mm.to_numpy(bool))
        result.append({"scenario": config["scenario"], "beta": config["beta"],
                       "micro_graph_type": config["micro_graph_type"],
                       "mean_internal_degree": config["mean_internal_degree"],
                       "community_size": config["community_size"], "inter_links": config["inter_links"],
                       "community": community, "tau": tau,
                       "metric_kind": "ordinary" if m["all_complete"] else "restricted", **m})
    return result


def topology_label(micro: str, degree) -> str:
    return "complete" if micro == "complete" else f"random connected, d={int(degree)}"


def heatmap(ax, frame: pd.DataFrame, value: str, cmap, vmin, vmax, title: str,
            percent: bool = False, mark_censored: bool = False,
            completion_annotation: bool = False):
    grid = np.full((len(K_VALUES), len(SIZES)), np.nan)
    samples = np.zeros((len(K_VALUES), len(SIZES)), dtype=int)
    complete = np.ones((len(K_VALUES), len(SIZES)), dtype=bool)
    delta_completion = np.full((len(K_VALUES), len(SIZES)), np.nan)
    for iy, k in enumerate(K_VALUES):
        for ix, n in enumerate(SIZES):
            row = frame[(frame.inter_links == k) & (frame.community_size == n)]
            if len(row):
                grid[iy, ix] = row.iloc[0][value] * (100 if percent else 1)
                samples[iy, ix] = int(row.iloc[0]["N_pairs"])
                complete[iy, ix] = bool(row.iloc[0]["all_complete"])
                delta_completion[iy, ix] = float(row.iloc[0]["delta_completion"])
    image = ax.imshow(grid, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SIZES)), SIZES); ax.set_yticks(range(len(K_VALUES)), [f"k={k}" for k in K_VALUES])
    ax.set_xlabel("community size n"); ax.set_title(title, fontsize=10)
    for y in range(len(K_VALUES)):
        for x in range(len(SIZES)):
            if np.isfinite(grid[y, x]):
                if completion_annotation:
                    label = f"{grid[y,x]:.2f}\n$\\Delta p$={delta_completion[y,x]:+.2f}"
                else:
                    label = f"{grid[y,x]:.1f}" if percent else f"{grid[y,x]:.2f}"
                if samples[y, x] < 100:
                    label += f"\nN={samples[y, x]}"
                    ax.add_patch(Rectangle((x - .49, y - .49), .98, .98, fill=False,
                                           edgecolor="#222222", linewidth=1.2, linestyle="--"))
                if mark_censored and not complete[y, x]:
                    ax.add_patch(Rectangle((x - .47, y - .47), .94, .94, fill=False,
                                           edgecolor="#111111", linewidth=1.4, linestyle=(0, (4, 2))))
                red, green, blue, _ = image.cmap(image.norm(grid[y, x]))
                luminance = .2126 * red + .7152 * green + .0722 * blue
                ax.text(x, y, label, ha="center", va="center", fontsize=7,
                        color="white" if luminance < .45 else "black")
    return image


def beta_tag(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def plot_atlas(metrics: pd.DataFrame, output: Path, beta: float,
               signed_limit: float, error_limit: float):
    censored = not metrics.all_complete.all()
    signed_title = "relative $\\Delta\\mathrm{RMST}_{\\tau}$ (%)" if censored else "relative mean shift (%)"
    w1_title = "relative restricted $W_{1,\\tau}$ (%)" if censored else "relative $W_1$ (%)"
    columns = [("relative_delta_rmst", signed_title, "RdBu_r", True),
               ("relative_restricted_w1", w1_title, "viridis", True)]
    if censored:
        columns.append(("p_min", "minimum completion; annotation: $\\Delta p$", "Blues", False))
    fig, axes = plt.subplots(3, len(columns), figsize=(4.0 * len(columns), 8.4), layout="constrained")
    axes = np.atleast_2d(axes)
    groups = [("complete", None), ("random_connected", 10), ("random_connected", 6)]
    for r, (micro, degree) in enumerate(groups):
        subset = metrics[metrics.micro_graph_type == micro]
        if degree is not None: subset = subset[subset.mean_internal_degree == degree]
        for c, (field, title, cmap, pct) in enumerate(columns):
            lim = signed_limit if field == "relative_delta_rmst" else (error_limit if field != "p_min" else 1.0)
            image = heatmap(axes[r, c], subset, field, cmap, -lim if field == "relative_delta_rmst" else 0, lim,
                            title if r == 0 else "", pct,
                            mark_censored=censored and field != "p_min",
                            completion_annotation=field == "p_min")
            if c == 0: axes[r, c].set_ylabel(topology_label(micro, degree))
            fig.colorbar(image, ax=axes[r, c], shrink=.82, label="%" if pct else "probability")
    tau = float(metrics.tau.iloc[0])
    title = (f"Accuracy atlas, $\\beta={beta:g}$: restricted metrics at $\\tau={tau:g}$"
             if censored else f"Accuracy atlas, $\\beta={beta:g}$: all terminal arrivals observed")
    if censored:
        title += "\ndashed cells: at least one censored terminal arrival"
    if (metrics.N_pairs < 100).any(): title += " [PREVIEW: incomplete cells]"
    fig.suptitle(title, fontsize=13)
    tag = beta_tag(beta)
    fig.savefig(output / f"fig_accuracy_atlas_beta{tag}.pdf")
    fig.savefig(output / f"fig_accuracy_atlas_beta{tag}.png", dpi=220)
    plt.close(fig)


def selected_key(micro: str, degree, n: int, k: int) -> tuple:
    return (micro, degree, n, k)


def paired_mean_interval(values: np.ndarray, count: int,
                         rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and paired-bootstrap percentile interval for one contrast vector."""
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def plot_accumulation(layers: pd.DataFrame, configs: dict, output: Path, boot: int,
                      rng: np.random.Generator, detail_beta: float):
    desired = [(detail_beta, "complete", None, 100, 5),
               (detail_beta, "random_connected", 6, 100, 5),
               (detail_beta, "random_connected", 10, 1000, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    loaded_scenarios = set(layers["scenario"])
    for key in desired:
        if key not in configs or configs[key]["scenario"] not in loaded_scenarios:
            print(f"[skip] accumulation panel unavailable: {key}", flush=True)
            continue
        config = configs[key]; part = layers[layers.scenario == config["scenario"]]
        x, shifts, shift_lo, shift_hi, w1, w1_lo, w1_hi = [], [], [], [], [], [], []
        for community, row in part.groupby("community", sort=True):
            if community == 0:
                continue
            full, mm = row.time_full.to_numpy(float), row.time_mm.to_numpy(float)
            ef, em = row.event_full.to_numpy(bool), row.event_mm.to_numpy(bool)
            point = metric_values(full, mm, ef, em)
            ci = bootstrap(full, mm, ef, em, boot, rng,
                           fields=("delta_rmst", "relative_restricted_w1"))
            x.append(community); shifts.append(point["delta_rmst"])
            shift_lo.append(ci["delta_rmst_ci_low"]); shift_hi.append(ci["delta_rmst_ci_high"])
            w1.append(point["relative_restricted_w1"] * 100)
            w1_lo.append(ci["relative_restricted_w1_ci_low"] * 100)
            w1_hi.append(ci["relative_restricted_w1_ci_high"] * 100)
        label = f"{topology_label(key[1], key[2])}, n={key[3]}, k={key[4]}"
        line = axes[0].plot(x, shifts, marker="o", label=label)[0]
        axes[0].fill_between(x, shift_lo, shift_hi, color=line.get_color(), alpha=.16)
        line = axes[1].plot(x, w1, marker="o", label=label)[0]
        axes[1].fill_between(x, w1_lo, w1_hi, color=line.get_color(), alpha=.16)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(title="Cumulative arrival-time shift", xlabel="community index",
                ylabel="time shift (MM − Full)")
    axes[1].set(title="Layerwise distributional error", xlabel="community index",
                ylabel="relative W1 (%)")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"Arrival-time discrepancy along the chain, $\\beta={detail_beta:g}$\n"
                 "shading: paired-bootstrap 95% intervals", fontsize=12)
    tag = beta_tag(detail_beta)
    fig.savefig(output / f"fig_arrival_error_accumulation_beta{tag}.pdf")
    fig.savefig(output / f"fig_arrival_error_accumulation_beta{tag}.png", dpi=220); plt.close(fig)


def plot_local_hop_delay(layers: pd.DataFrame, configs: dict, output: Path, boot: int,
                         rng: np.random.Generator, detail_beta: float) -> pd.DataFrame:
    """Plot local hop-delay contrasts without conditioning on model-specific completion.

    The diagnostic is generated only for selected scenarios whose arrivals are
    fully observed in both simulators at every displayed layer.  This prevents
    completed-only selection from entering a local-delay comparison.
    """
    desired = [(detail_beta, "complete", None, 100, 5),
               (detail_beta, "random_connected", 6, 100, 5),
               (detail_beta, "random_connected", 10, 1000, 1)]
    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    table: list[dict] = []
    for key in desired:
        if key not in configs:
            print(f"[skip] local-hop profile unavailable: {key}", flush=True)
            continue
        config = configs[key]
        part = layers[layers.scenario == config["scenario"]]
        wide = part.pivot(index="replication_id", columns="community",
                          values=["time_full", "time_mm", "event_full", "event_mm"])
        if wide.empty:
            continue
        x, estimate, low, high = [], [], [], []
        for community in range(1, 10):
            observed = (wide[("event_full", community)].astype(bool)
                        & wide[("event_full", community - 1)].astype(bool)
                        & wide[("event_mm", community)].astype(bool)
                        & wide[("event_mm", community - 1)].astype(bool))
            if not observed.all():
                print(f"[skip] local-hop profile for {config['scenario']}: "
                      f"community {community} is censored in at least one paired run", flush=True)
                x = []
                break
            full_delay = (wide[("time_full", community)] - wide[("time_full", community - 1)]).to_numpy(float)
            mm_delay = (wide[("time_mm", community)] - wide[("time_mm", community - 1)]).to_numpy(float)
            point, lo, hi = paired_mean_interval(mm_delay - full_delay, boot, rng)
            x.append(community); estimate.append(point); low.append(lo); high.append(hi)
            table.append({"scenario": config["scenario"], "beta": detail_beta,
                          "micro_graph_type": config["micro_graph_type"],
                          "mean_internal_degree": config.get("mean_internal_degree"),
                          "community_size": config["community_size"], "inter_links": config["inter_links"],
                          "destination_community": community, "N_pairs": len(full_delay),
                          "mean_hop_delay_difference": point, "ci_low": lo, "ci_high": hi})
        if not x:
            table = [row for row in table if row["scenario"] != config["scenario"]]
            continue
        label = f"{topology_label(key[1], key[2])}, n={key[3]}, k={key[4]}"
        line = ax.plot(x, estimate, marker="o", label=label)[0]
        ax.fill_between(x, low, high, color=line.get_color(), alpha=.16)
    ax.axhline(0, color="black", lw=.8)
    ax.set(title=f"Local inter-community hop-delay error, $\\beta={detail_beta:g}$",
           xlabel="destination community index",
           ylabel=r"mean $[(T_j-T_{j-1})^{MM}-(T_j-T_{j-1})^{Full}]$")
    ax.grid(alpha=.2)
    if table:
        ax.legend(fontsize=8)
    tag = beta_tag(detail_beta)
    fig.savefig(output / f"fig_local_hop_delay_beta{tag}.pdf")
    fig.savefig(output / f"fig_local_hop_delay_beta{tag}.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(table)


PROFILE_METRICS = (
    ("relative_delta_rmst", "signed timing bias (%)", "relative_delta_rmst_ci_low", "relative_delta_rmst_ci_high"),
    ("relative_restricted_w1", "relative distributional error (%)", "relative_restricted_w1_ci_low", "relative_restricted_w1_ci_high"),
    ("p_min", "minimum completion (%)", None, None),
)
TOPOLOGY_STYLES = {
    ("complete", None): ("complete", "#222222", "o"),
    ("random_connected", 10): ("random connected d=10", "#2878b5", "s"),
    ("random_connected", 6): ("random connected d=6", "#d0604c", "^"),
}


def profile_y(frame: pd.DataFrame, metric: tuple) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    field, _, low, high = metric
    scale = 100.0
    values = frame[field].to_numpy(float) * scale
    if low is None:
        return values, None, None
    return values, frame[low].to_numpy(float) * scale, frame[high].to_numpy(float) * scale


def plot_profile_line(ax, frame: pd.DataFrame, x_field: str, metric: tuple, label: str, color: str, marker: str):
    frame = frame.sort_values(x_field)
    x = frame[x_field].to_numpy(float)
    y, low, high = profile_y(frame, metric)
    incomplete = frame.N_pairs.to_numpy(int) < 100
    ax.plot(x, y, color=color, marker=marker, lw=1.5, ms=4, label=label)
    if low is not None:
        ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
    if incomplete.any():
        ax.scatter(x[incomplete], y[incomplete], facecolors="none", edgecolors=color, s=55, linewidths=1.2, zorder=3)


def plot_parameter_profiles(metrics: pd.DataFrame, output: Path):
    """Optional beta-specific profiles; never connect unlike estimands.

    A rate with censoring is plotted on a separate page and explicitly labelled
    as restricted.  Open markers identify cells with at least one censored
    terminal arrival.  These figures are diagnostic supplements, not main-text
    replacements for the complete accuracy atlases.
    """
    topology_rows = [("complete", None), ("random_connected", 10), ("random_connected", 6)]
    figure_specs = ((PROFILE_METRICS[1], "fig_relative_w1_profiles", "Relative terminal-arrival distribution error"),
                    (PROFILE_METRICS[0], "fig_signed_bias_profiles", "Signed terminal-arrival timing bias"))
    for beta in sorted(metrics.beta.unique()):
        beta_metrics = metrics[np.isclose(metrics.beta, beta)]
        restricted = not beta_metrics.all_complete.all()
        for metric, filename, heading in figure_specs:
            fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.2), sharex=True, sharey=True)
            for row, (micro, degree) in enumerate(topology_rows):
                for col, k in enumerate(K_VALUES):
                    ax = axes[row, col]
                    part = beta_metrics[(beta_metrics.micro_graph_type == micro)
                                        & (beta_metrics.inter_links == k)]
                    if degree is not None:
                        part = part[part.mean_internal_degree == degree]
                    part = part.sort_values("community_size")
                    if part.empty:
                        continue
                    x = part.community_size.to_numpy(float)
                    y, low, high = profile_y(part, metric)
                    color = "#2878b5" if beta < .002 else "#d0604c"
                    ax.plot(x, y, color=color, marker="o", lw=1.5, ms=4)
                    ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
                    censored = ~part.all_complete.to_numpy(bool)
                    if censored.any():
                        ax.scatter(x[censored], y[censored], facecolors="none", edgecolors="#111111",
                                   s=70, linewidths=1.2, zorder=3)
                    if row == 0:
                        ax.set_title(f"k={k}")
                    if col == 0:
                        ax.set_ylabel(topology_label(micro, degree) + "\n" + metric[1])
                    if row == 2:
                        ax.set_xlabel("community size n")
                    ax.set_xscale("log"); ax.set_xticks(SIZES, [str(n) for n in SIZES]); ax.grid(alpha=.2)
                    if metric[0] == "relative_delta_rmst":
                        ax.axhline(0, color="#222222", lw=.8)
            qualifier = f"restricted at $\\tau={float(beta_metrics.tau.iloc[0]):g}$" if restricted else "all arrivals observed"
            fig.suptitle(f"{heading}, $\\beta={beta:g}$ ({qualifier})\n"
                         "shading: paired-bootstrap 95% intervals; open markers: censored cells",
                         fontsize=11, y=.995)
            fig.subplots_adjust(top=.88, hspace=.12, wspace=.10)
            tag = beta_tag(beta)
            fig.savefig(output / f"{filename}_beta{tag}.pdf")
            fig.savefig(output / f"{filename}_beta{tag}.png", dpi=220)
            plt.close(fig)

    profiles = metrics.copy(); profiles["preview"] = profiles.N_pairs < 100
    profiles.to_csv(output.parent / "parameter_profiles.csv", index=False)


def plot_beta_robustness(metrics: pd.DataFrame, output: Path):
    """Compare dimensionless errors in matched, fully observed cells across rates."""
    if not all(any(np.isclose(metrics.beta, beta)) for beta in BETAS):
        print("[skip] beta robustness figure requires both beta grids", flush=True)
        return
    fields = ["relative_restricted_w1", "relative_delta_rmst"]
    join = ["micro_graph_type", "degree_key", "community_size", "inter_links"]
    prepared = metrics.copy(); prepared["degree_key"] = prepared.mean_internal_degree.fillna(-1)
    comparison_fields = fields + ["all_complete"]
    left = prepared[np.isclose(prepared.beta, 0.001)][join + comparison_fields].rename(
        columns={field: f"{field}_001" for field in comparison_fields})
    right = prepared[np.isclose(prepared.beta, 0.003)][join + comparison_fields].rename(
        columns={field: f"{field}_003" for field in comparison_fields})
    wide = left.merge(right, on=join, how="inner", validate="one_to_one")
    wide = wide[wide.all_complete_001 & wide.all_complete_003]
    if wide.empty:
        print("[skip] beta robustness figure has no cells fully observed at both rates", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")
    topology_marks = {("complete", None): ("complete", "o", "#222222"),
                      ("random_connected", 10): ("random d=10", "s", "#2878b5"),
                      ("random_connected", 6): ("random d=6", "^", "#d0604c")}
    for ax, field, title in zip(axes, fields, ("Relative W1", "Signed relative timing bias")):
        for (micro, degree), (label, marker, color) in topology_marks.items():
            rows = wide[wide.micro_graph_type == micro]
            rows = rows[rows.degree_key == (-1 if degree is None else degree)]
            x = rows[f"{field}_001"].to_numpy(float) * 100
            y = rows[f"{field}_003"].to_numpy(float) * 100
            sizes = 22 + 5 * np.sqrt(rows.community_size.to_numpy(float))
            ax.scatter(x, y, marker=marker, c=color, s=sizes, alpha=.78, label=label,
                       edgecolors="white", linewidths=.5)
        finite = np.concatenate([ax.collections[i].get_offsets().data.ravel() for i in range(len(ax.collections))])
        finite = finite[np.isfinite(finite)]
        if len(finite):
            lo, hi = float(finite.min()), float(finite.max()); margin = max((hi - lo) * .08, .2)
            ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], color="#666666", lw=1, linestyle=":")
            ax.set_xlim(lo-margin, hi+margin); ax.set_ylim(lo-margin, hi+margin)
        ax.set(xlabel=r"$\beta=0.001$ (%)", ylabel=r"$\beta=0.003$ (%)", title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Cross-rate sensitivity relative to the fixed scheduler\n"
                 f"{len(wide)} cells fully observed at both rates; marker area increases with n", fontsize=11)
    fig.savefig(output / "fig_beta_robustness.pdf"); fig.savefig(output / "fig_beta_robustness.png", dpi=220); plt.close(fig)


def absorbing_extension(payload: dict, model: str, community: int) -> float | None:
    termination = {row.get("simulator"): row for row in payload.get("termination", [])}
    row = termination.get(model)
    if not row or row.get("stop_reason") != "all_nodes_infected":
        return None
    states = row.get("final_state_by_community", [])
    state = next((item for item in states if int(item.get("community", -1)) == community), None)
    return None if state is None else float(state["I"])


def held_values(columns: dict, community: int, target: np.ndarray,
                extension: float | None = None) -> np.ndarray | None:
    raw = pd.DataFrame(columns)
    raw = raw[raw.community == community].sort_values("time")
    if raw.empty: return None
    result = np.full(len(target), np.nan)
    supported = target <= float(raw.time.max()) + 1e-9
    ix = np.searchsorted(raw.time.to_numpy(float), target, side="right") - 1
    supported &= ix >= 0
    result[supported] = raw.I.to_numpy(float)[ix[supported]]
    if extension is not None:
        result[target > float(raw.time.max())] = extension
    return result


def plot_aligned(payloads: dict, configs: dict, output: Path, detail_beta: float):
    desired = [(detail_beta, "complete", None, 100, 5),
               (detail_beta, "random_connected", 6, 100, 5)]
    communities = (1, 5, 9); window = np.arange(0.0, 1001.0, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=False, sharey=True, layout="constrained")
    summary, panels = [], {}
    for row, key in enumerate(desired):
        if key not in configs:
            for ax in axes[row]: ax.set_axis_off()
            print(f"[skip] aligned-trajectory row unavailable: {key}", flush=True)
            continue
        config = configs[key]
        for col, community in enumerate(communities):
            curves = {model: [] for model in MODELS}
            for item in payloads[config["scenario"]]:
                payload = item["payload"]
                arrivals = pd.DataFrame(payload["community_arrivals"])
                trial = {}
                for model in MODELS:
                    arrival = arrivals[(arrivals.simulator == model) & (arrivals.community == community)].first_infection_time.iloc[0]
                    if not finite(arrival) or not has_trajectories(payload): continue
                    target = float(arrival) + window
                    values = held_values(model_trajectory_columns(payload, model), community, target,
                                         absorbing_extension(payload, model, community))
                    if values is not None: trial[model] = values / config["community_size"]
                if set(trial) == set(MODELS):
                    for model in MODELS: curves[model].append(trial[model])
            if curves[FULL]:
                full_stack, mm_stack = np.asarray(curves[FULL]), np.asarray(curves[MM])
                coverage = np.sum(np.isfinite(full_stack) & np.isfinite(mm_stack), axis=0)
                required = min(90, max(2, math.ceil(.9 * len(full_stack))))
                panels[(row, col)] = (config, community, full_stack, mm_stack, coverage, required)

    if not panels:
        print("[skip] no paired trajectories available for arrival alignment", flush=True)
        plt.close(fig)
        return pd.DataFrame()
    # Keep one common window within each topology row, but do not force the
    # rapidly saturating clique onto the much longer sparse-graph time scale.
    row_end_index: dict[int, int] = {}
    for row in range(2):
        row_panels = [value for (panel_row, _), value in panels.items() if panel_row == row]
        if not row_panels:
            continue
        valid_window = np.ones(len(window), dtype=bool)
        for _, _, _, _, coverage, required in row_panels:
            valid_window &= coverage >= required
        invalid = np.flatnonzero(~valid_window)
        coverage_end = int(invalid[0] - 1) if len(invalid) else len(window) - 1
        if coverage_end < 1:
            print(f"[skip] aligned trajectory row {row} has insufficient post-arrival coverage", flush=True)
            continue
        saturation_times: list[int] = []
        for _, _, full_stack, mm_stack, _, _ in row_panels:
            for stack in (full_stack, mm_stack):
                median = np.nanmedian(stack[:, :coverage_end + 1], axis=0)
                reached = np.flatnonzero(median >= .95)
                if len(reached):
                    saturation_times.append(int(reached[0]))
        informative_end = max(saturation_times) if saturation_times else coverage_end
        # Leave visible context after saturation while retaining the coverage gate.
        display_end = min(coverage_end, max(50, int(math.ceil(1.15 * informative_end))))
        row_end_index[row] = display_end

    for row in range(2):
        for col in range(3):
            if (row, col) not in panels:
                axes[row, col].set_axis_off()

    for (row, col), (config, community, full_stack, mm_stack, coverage, required) in panels.items():
        if row not in row_end_index:
            axes[row, col].set_axis_off()
            continue
        end_index = row_end_index[row]
        used_window = window[:end_index + 1]
        ax = axes[row, col]
        curves = {FULL: full_stack[:, :end_index + 1], MM: mm_stack[:, :end_index + 1]}
        for model in MODELS:
            stack = curves[model]
            if len(stack):
                median, q1, q3 = np.nanquantile(stack, [.5, .25, .75], axis=0)
                ax.plot(used_window, median, color=COLORS[model], label=LABELS[model])
                ax.fill_between(used_window, q1, q3, color=COLORS[model], alpha=.15)
                summary.extend({"scenario": config["scenario"], "community": community, "simulator": model,
                                "relative_time": t, "median_I_fraction": a, "q25": b, "q75": c,
                                "paired_coverage": int(n)}
                               for t, a, b, c, n in zip(used_window, median, q1, q3, coverage[:end_index + 1]))
        ax.set(title=f"{topology_label(config['micro_graph_type'], config.get('mean_internal_degree'))}; community {community}",
               xlabel="time since own arrival", ylabel="infected fraction" if col == 0 else "",
               xlim=(0, float(used_window[-1])), ylim=(0, 1.02))
        ax.grid(alpha=.2); ax.text(.98, .04, f"N(s)≥{required}", ha="right", transform=ax.transAxes, fontsize=7)
        if row == 0 and col == 0: ax.legend(fontsize=8)
    fig.suptitle(f"Local dynamics aligned by each model's community arrival; median and IQR; $\\beta={detail_beta:g}$\n"
                 "common horizontal scale within each topology row", fontsize=12)
    tag = beta_tag(detail_beta)
    fig.savefig(output / f"fig_arrival_aligned_local_dynamics_beta{tag}.pdf")
    fig.savefig(output / f"fig_arrival_aligned_local_dynamics_beta{tag}.png", dpi=220); plt.close(fig)
    return pd.DataFrame(summary)


def main() -> None:
    args = arguments(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    betas = (args.beta,) if args.beta is not None else tuple(dict.fromkeys(args.betas))
    scenarios = discover(args.data_root.resolve(), betas)
    rng = np.random.default_rng(args.seed)
    validations, all_layers, payloads, configs, metrics = [], [], {}, {}, []
    aligned_keys = {(args.detail_beta, "complete", None, 100, 5),
                    (args.detail_beta, "random_connected", 6, 100, 5)}
    for key, folder in scenarios.items():
        config = canonical_config(folder).copy(); config["scenario"] = folder.name
        terminal, layers, validation, loaded_payloads = load_cell(
            folder, args.tau, args.allow_partial,
            retain_payload=(not args.skip_aligned and key in aligned_keys), expected_config=config)
        configs[key] = config
        if loaded_payloads:
            payloads[config["scenario"]] = loaded_payloads
        validations.extend(validation)
        if terminal.empty:
            continue
        all_layers.append(layers.assign(scenario=config["scenario"]))
        metrics.append(cell_metrics(terminal, config, args.tau, args.bootstrap, rng))
        print(f"[loaded] {folder.name}: {len(terminal)}/100 valid paired checkpoints", flush=True)
    if not metrics:
        raise RuntimeError("No cells have sufficient valid follow-up for analysis.")
    metric_frame = pd.DataFrame(metrics); metric_frame["p_min"] = metric_frame[["p_full", "p_mm"]].min(axis=1)
    for beta in betas:
        subset = metric_frame[np.isclose(metric_frame.beta, beta)]
        if subset.empty:
            print(f"[summary] beta={beta:g}: no analysable cells", flush=True)
        else:
            print(f"[summary] beta={beta:g}: cells={len(subset)}/36; "
                  f"fully observed terminal arrivals={int(subset.all_complete.sum())}/{len(subset)}; "
                  f"cells with 100 pairs={int((subset.N_pairs == 100).sum())}/{len(subset)}", flush=True)
    layers = pd.concat(all_layers, ignore_index=True)
    layer_table = pd.DataFrame(sum((layer_metrics(layers[layers.scenario == c["scenario"]], c, args.tau) for c in configs.values()), []))
    pd.DataFrame(validations).to_csv(output / "checkpoint_validation.csv", index=False)
    metric_frame.to_csv(output / "cell_metrics.csv", index=False)
    layer_table.to_csv(output / "layer_metrics.csv", index=False)
    signed_limit = max(1.0, float(metric_frame.relative_delta_rmst.abs().max() * 100))
    error_limit = max(1.0, float(metric_frame.relative_restricted_w1.max() * 100))
    for beta in betas:
        subset = metric_frame[np.isclose(metric_frame.beta, beta)]
        if not subset.empty:
            plot_atlas(subset, figures, beta, signed_limit, error_limit)
    if args.include_profiles:
        plot_parameter_profiles(metric_frame, figures)
    else:
        print("[skip] redundant one-parameter profiles (use --include-profiles for beta-specific diagnostics)",
              flush=True)
    plot_beta_robustness(metric_frame, figures)
    plot_accumulation(layers, configs, figures, args.bootstrap, rng, args.detail_beta)
    plot_local_hop_delay(layers, configs, figures, args.bootstrap, rng, args.detail_beta).to_csv(
        output / "local_hop_delay.csv", index=False)
    if not args.skip_aligned:
        plot_aligned(payloads, configs, figures, args.detail_beta).to_csv(
            output / "aligned_trajectory_summary.csv", index=False)
    frozen = {"script": str(Path(__file__).relative_to(ROOT)), "tau": args.tau, "betas": betas,
              "detail_beta": args.detail_beta,
              "parameter_profiles_generated": bool(args.include_profiles),
              "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
              "partial_cell_policy": "Plot available valid pairs and emit a PREVIEW warning; do not treat as final.",
              "beta_interpretation": (
                  "beta_in=beta_out changes the epidemic time scale while tau_micro=macro_T=1 remains fixed; "
                  "cross-beta figures therefore measure sensitivity to process rate relative to the scheduler."
              ),
              "metric_policy": (
                  "Use ordinary mean shift/W1/KS when every terminal arrival is observed in a cell; "
                  "otherwise use completion plus capped/restricted RMST, W1 and KS at the common tau."
              ),
              "clique_bridge_leaf": "Retained existing validated figure; not regenerated by this script.",
              "scenarios": [{"folder": p.name, "config": configs[k]} for k, p in scenarios.items()]}
    (output / "analysis_manifest.json").write_text(json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Figures saved to: {figures}\nTables and manifest saved to: {output}")


if __name__ == "__main__":
    main()
