"""Summary metrics for the SMPT two-scale manuscript (ten-community chain size sweep).

Reads saved checkpoints only; no simulation is run. Outputs go to
output/analysis/smpt_twoscale_paper/{per_run.csv.gz,cell_summary.csv}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "analysis" / "smpt_twoscale_paper"
SIZES = (50, 100, 250, 1000)
BETA = 0.003
L = 10
N_PERM = 2000
N_BOOT = 2000
SEED = 20260930
SCENARIO_RE = re.compile(
    r"__n(?P<n>\d+)__(?P<family>clique|tree_augmented)__d(?P<d>[0-9p]+)__aspl(?P<aspl>\d+)__k(?P<k>\d+)__"
)


def experiment_root(n: int) -> Path:
    return ROOT / "output" / "data" / (
        f"l10_n{n}_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
    )


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def infection_times(events: list) -> dict[int, float]:
    times: dict[int, float] = {}
    for t, _, node in events:
        times.setdefault(int(node), float(t))
    return times


def per_run(path_str: str) -> dict:
    path = Path(path_str)
    match = SCENARIO_RE.search(path.parent.parent.name)
    n, k, aspl = int(match["n"]), int(match["k"]), int(match["aspl"])
    degree = float(match["d"].replace("p", "."))
    payload = read(path)
    task = payload["task"]
    full, mm = payload["runs"]
    if not full["simulator"].startswith("Full") or "MicroMacro" not in mm["simulator"]:
        raise RuntimeError(f"unexpected simulator order in {path}")
    extended = path.parent.parent / "checkpoints_full_to40000" / path.name
    if extended.exists():
        ext = read(extended)
        if ext["task"] != task:
            raise RuntimeError(f"extension task mismatch: {extended}")
        full = ext["full_run"]

    bridges: dict[int, set[int]] = {c: set() for c in range(L - 1)}
    for u, v in task["interface_edges"]:
        cu, cv = int(u) // n, int(v) // n
        if cv != cu + 1:
            raise RuntimeError(f"unexpected interface orientation in {path}")
        bridges[cu].add(int(u))

    row = {
        "n": n, "k": k, "aspl": aspl, "mean_degree": degree,
        "network_id": int(task["network_id"]), "replicate_id": int(task["replicate_id"]),
        "full_runtime_s": float(full["runtime_seconds"]), "mm_runtime_s": float(mm["runtime_seconds"]),
        "extended": extended.exists(),
    }
    for label, run in (("full", full), ("mm", mm)):
        arrivals = run["arrivals"]
        for c in range(L):
            row[f"{label}_T{c}"] = np.nan if arrivals[c] is None else float(arrivals[c])
        tau = infection_times(run["infection_events"])
        for c in range(L - 1):
            hits = [tau[b] for b in bridges[c] if b in tau]
            row[f"{label}_TB{c}"] = min(hits) if hits else np.nan
        if label == "full":
            nodes = np.fromiter(tau.keys(), dtype=np.int64, count=len(tau))
            times = np.fromiter(tau.values(), dtype=float, count=len(tau))
            community = nodes // n
            spreads = [float(times[community == c].std()) for c in range(L - 1)
                       if np.count_nonzero(community == c) == n]
            row["full_sd"] = float(np.mean(spreads)) if spreads else np.nan
    return row


def collect(workers: int, sizes: tuple[int, ...]) -> pd.DataFrame:
    paths = []
    for n in sizes:
        for scenario in sorted((experiment_root(n) / "scenarios").iterdir()):
            paths.extend(str(p) for p in sorted((scenario / "checkpoints").glob("*.json.gz")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(per_run, paths, chunksize=16))
    return pd.DataFrame(rows).sort_values(["n", "aspl", "k", "network_id", "replicate_id"]).reset_index(drop=True)


def rel_w1(full: np.ndarray, mm: np.ndarray) -> float:
    return 100.0 * np.mean(np.abs(np.sort(mm) - np.sort(full))) / np.mean(full)


def median_shift(full: np.ndarray, mm: np.ndarray) -> float:
    return 100.0 * (np.median(mm) - np.median(full)) / np.median(full)


def mean_shift(full: np.ndarray, mm: np.ndarray) -> float:
    return 100.0 * (np.mean(mm) - np.mean(full)) / np.mean(full)


def summarise(cell: pd.DataFrame, rng: np.random.Generator) -> dict:
    full_t9, mm_t9 = cell["full_T9"].to_numpy(), cell["mm_T9"].to_numpy()
    if np.isnan(full_t9).any() or np.isnan(mm_t9).any():
        raise RuntimeError("unobserved terminal arrival in a summarised cell")
    obs_w, obs_b = rel_w1(full_t9, mm_t9), median_shift(full_t9, mm_t9)

    # Within-pair label swaps: the null that paired Full/MM times are exchangeable.
    swap = rng.random((N_PERM, full_t9.size)) < 0.5
    perm_f = np.where(swap, mm_t9, full_t9)
    perm_m = np.where(swap, full_t9, mm_t9)
    null_w = 100.0 * np.mean(np.abs(np.sort(perm_m, axis=1) - np.sort(perm_f, axis=1)), axis=1) / perm_f.mean(axis=1)
    null_b = 100.0 * (np.median(perm_m, axis=1) - np.median(perm_f, axis=1)) / np.median(perm_f, axis=1)

    first_f = cell["full_T1"].to_numpy()
    first_m = cell["mm_T1"].to_numpy()
    later_f = full_t9 - first_f
    later_m = mm_t9 - first_m
    s_first, s_later = mean_shift(first_f, first_m), mean_shift(later_f, later_m)

    blocks = cell["network_id"].to_numpy()
    ids = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ids]
    boot_b, boot_first, boot_later = [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, ids.size, ids.size)])
        boot_b.append(median_shift(full_t9[idx], mm_t9[idx]))
        boot_first.append(mean_shift(first_f[idx], first_m[idx]))
        boot_later.append(mean_shift(later_f[idx], later_m[idx]))
    boot_b, boot_first, boot_later = map(np.asarray, (boot_b, boot_first, boot_later))
    boot_d = boot_later - boot_first

    access, bypass_mm, bypass_full = [], [], []
    for c in range(L - 1):
        access.append(cell[f"full_TB{c}"] - cell[f"full_T{c}"])
        bypass_mm.append(cell[f"mm_T{c + 1}"] < cell[f"mm_TB{c}"])
        bypass_full.append(cell[f"full_T{c + 1}"] < cell[f"full_TB{c}"])
    access = pd.concat(access).to_numpy()
    k = int(cell["k"].iloc[0])

    return {
        "R": int(full_t9.size),
        "mean_full_T9": float(full_t9.mean()), "mean_mm_T9": float(mm_t9.mean()),
        "median_full_T9": float(np.median(full_t9)), "median_mm_T9": float(np.median(mm_t9)),
        "rel_w1_pct": obs_w, "rel_w1_null95_pct": float(np.quantile(null_w, 0.95)),
        "rel_w1_null_median_pct": float(np.median(null_w)),
        "rel_w1_perm_p": float((1 + np.sum(null_w >= obs_w)) / (N_PERM + 1)),
        "median_shift_pct": obs_b,
        "median_shift_ci_low": float(np.quantile(boot_b, 0.025)),
        "median_shift_ci_high": float(np.quantile(boot_b, 0.975)),
        "median_shift_perm_p": float((1 + np.sum(np.abs(null_b) >= abs(obs_b))) / (N_PERM + 1)),
        "mean_shift_pct": mean_shift(full_t9, mm_t9),
        "first_cross_shift_pct": s_first,
        "first_cross_ci_low": float(np.quantile(boot_first, 0.025)),
        "first_cross_ci_high": float(np.quantile(boot_first, 0.975)),
        "later_cross_shift_pct": s_later,
        "later_cross_ci_low": float(np.quantile(boot_later, 0.025)),
        "later_cross_ci_high": float(np.quantile(boot_later, 0.975)),
        "later_minus_first_pct": s_later - s_first,
        "later_minus_first_ci_low": float(np.quantile(boot_d, 0.025)),
        "later_minus_first_ci_high": float(np.quantile(boot_d, 0.975)),
        "mean_access_full": float(np.nanmean(access)),
        "rho": float(BETA * k * np.nanmean(access)),
        "sigma_full": float(cell["full_sd"].mean()),
        "rho_sigma": float(BETA * np.sqrt(k) * cell["full_sd"].mean()),
        "mm_chain_bypass_pct": float(100.0 * pd.concat(bypass_mm).mean()),
        "full_chain_bypass_pct": float(100.0 * pd.concat(bypass_full).mean()),
        "median_full_runtime_s": float(cell["full_runtime_s"].median()),
        "median_mm_runtime_s": float(cell["mm_runtime_s"].median()),
        "extended_runs": int(cell["extended"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reuse", action="store_true", help="reuse per_run.csv.gz if present")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    per_run_path = out / "per_run.csv.gz"
    if args.reuse and per_run_path.exists():
        runs = pd.read_csv(per_run_path)
    else:
        runs = collect(args.workers, tuple(args.sizes))
        runs.to_csv(per_run_path, index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for (n, aspl, k), cell in runs.groupby(["n", "aspl", "k"], sort=True):
        rows.append({"n": n, "aspl": aspl, "k": k, "mean_degree": cell["mean_degree"].iloc[0],
                     **summarise(cell, rng)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cell_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary[["n", "aspl", "k", "rel_w1_pct", "rel_w1_null95_pct", "rel_w1_perm_p",
                       "median_shift_pct", "median_shift_ci_low", "median_shift_ci_high",
                       "first_cross_shift_pct", "later_cross_shift_pct", "later_minus_first_ci_low",
                       "later_minus_first_ci_high", "rho", "rho_sigma", "mm_chain_bypass_pct",
                       "median_full_runtime_s", "median_mm_runtime_s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
