import argparse
import math
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.config import load_config
from sim_db import init_db, resolve_path


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required; install two-layer-ctmc[analysis]") from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("matplotlib is required; install two-layer-ctmc[analysis]") from exc
    return plt


def _default_paths_from_config() -> Tuple[Path, Path]:
    cfg, base_dir = load_config()
    db_path = resolve_path(cfg.get("storage", {}).get("db_path", "simulations.db"), base_dir=base_dir)

    net_cfg = cfg["network"]
    community_size = int(net_cfg["community_size"])
    edge_prob_str = str(net_cfg["edge_prob"]).replace(".", "p")
    folder_name = (
        f"k_{net_cfg['communities']}_"
        f"n_{community_size}_"
        f"inter_{net_cfg['inter_links']}_"
        f"macro_{net_cfg['macro_graph_type']}_"
        f"micro_{net_cfg['micro_graph_type']}_"
        f"p{edge_prob_str}"
    )
    out_dir = resolve_path(Path("plots") / folder_name, base_dir=base_dir)
    return db_path, out_dir


def _quantiles(values: np.ndarray) -> Tuple[float, float, float]:
    return (
        float(np.quantile(values, 0.1)),
        float(np.quantile(values, 0.5)),
        float(np.quantile(values, 0.9)),
    )


def _cdf_points(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values.astype(float))
    n = xs.size
    ys = np.arange(1, n + 1, dtype=float) / float(n)
    return xs, ys


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / float(a.size)
    fb = np.searchsorted(np.sort(b), grid, side="right") / float(b.size)
    return float(np.max(np.abs(fa - fb)))


def _w1_distance(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.sort(a)
    xb = np.sort(b)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    if xa.size == xb.size:
        return float(np.mean(np.abs(xa - xb)))

    grid = np.unique(np.concatenate([xa, xb]))
    if grid.size <= 1:
        return 0.0
    fa = np.searchsorted(xa, grid, side="right") / float(xa.size)
    fb = np.searchsorted(xb, grid, side="right") / float(xb.size)
    dx = np.diff(grid)
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * dx))


def _boxplot_stats(values: np.ndarray, n_total: int, n_reached: int) -> Dict[str, float]:
    q25 = float(np.quantile(values, 0.25))
    q50 = float(np.quantile(values, 0.50))
    q75 = float(np.quantile(values, 0.75))
    iqr = q75 - q25
    low_bound = q25 - 1.5 * iqr
    high_bound = q75 + 1.5 * iqr
    inliers = values[(values >= low_bound) & (values <= high_bound)]
    whisker_low = float(np.min(inliers)) if inliers.size else q25
    whisker_high = float(np.max(inliers)) if inliers.size else q75
    outlier_count = int(values.size - inliers.size)
    return {
        "n_total": int(n_total),
        "n_reached": int(n_reached),
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "iqr": iqr,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outlier_count": outlier_count,
    }


def _load_raw(db_path: Path, scenario_ids: Optional[List[str]]):
    pd = _require_pandas()
    conn = sqlite3.connect(db_path)
    try:
        where = "WHERE r.simulator IN ('Micro', 'MicroMacro')"
        params: List[object] = []
        if scenario_ids:
            placeholders = ",".join(["?"] * len(scenario_ids))
            where += f" AND r.scenario_id IN ({placeholders})"
            params.extend(scenario_ids)
        query = f"""
            SELECT
                r.run_uid,
                r.scenario_id,
                r.simulator,
                r.replication_id,
                r.communities,
                r.T_end,
                cm.community,
                cm.t0
            FROM runs r
            JOIN community_metrics cm ON cm.run_uid = r.run_uid
            {where}
            ORDER BY r.scenario_id, r.simulator, r.replication_id, cm.community
        """
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


def _tlast_sample(values: np.ndarray, reached: np.ndarray, tmax: np.ndarray, mode: str) -> np.ndarray:
    if mode == "completed_only":
        return values[reached]
    if mode == "right_censor":
        out = values.copy()
        out[~reached] = tmax[~reached]
        return out
    # keep_flag: distribution metrics use completed values; reached probability is reported separately
    return values[reached]


def run_analysis(
    *,
    db_path: Path,
    output_dir: Path,
    scenario_ids: Optional[List[str]],
    censoring_mode: str,
    paired_mode: bool,
    bins: int,
    tmax_override: Optional[float],
) -> str:
    pd = _require_pandas()
    plt = _require_matplotlib()
    init_db(db_path)

    df = _load_raw(db_path, scenario_ids)
    if df.empty:
        raise RuntimeError("No runs found for selected scenarios/models.")

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_uid = f"bench_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().isoformat() + "Z"

    summary_rows: List[dict] = []
    cdf_rows: List[dict] = []
    density_rows: List[dict] = []
    box_rows: List[dict] = []
    paired_rows: List[dict] = []

    for scenario_id, sdf in df.groupby("scenario_id"):
        scenario_slug = str(scenario_id)
        scenario_dir = output_dir / scenario_slug
        scenario_dir.mkdir(parents=True, exist_ok=True)

        pivot = (
            sdf.pivot_table(
                index=["run_uid", "simulator", "replication_id", "communities", "T_end"],
                columns="community",
                values="t0",
                aggfunc="first",
            )
            .reset_index()
        )

        if pivot.empty:
            continue
        max_comm = int(np.nanmax(pivot["communities"].to_numpy()) - 1)
        if max_comm < 0:
            continue
        if max_comm not in pivot.columns:
            pivot[max_comm] = np.nan

        model_data: Dict[str, dict] = {}
        for model in ("Micro", "MicroMacro"):
            mdf = pivot[pivot["simulator"] == model].copy()
            if mdf.empty:
                continue
            t_last = mdf[max_comm].to_numpy(dtype=float)
            t_end = mdf["T_end"].to_numpy(dtype=float)
            reached = ~np.isnan(t_last)
            tmax = np.full_like(t_end, float(tmax_override)) if tmax_override is not None else t_end
            tlast_sample = _tlast_sample(t_last, reached, tmax, censoring_mode)

            comm_samples: Dict[int, np.ndarray] = {}
            for c in range(max_comm + 1):
                if c not in mdf.columns:
                    continue
                vals = mdf[c].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    comm_samples[c] = vals

            model_data[model] = {
                "frame": mdf,
                "t_last_raw": t_last,
                "reached": reached,
                "tmax": tmax,
                "t_last": tlast_sample,
                "comm_samples": comm_samples,
            }
            p_reached = float(np.mean(reached)) if reached.size else float("nan")
            summary_rows.append(
                {
                    "analysis_uid": analysis_uid,
                    "scenario_id": scenario_slug,
                    "variable": "T_last",
                    "community": None,
                    "metric": f"p_reached_{model}",
                    "value": p_reached,
                }
            )

        if "Micro" not in model_data or "MicroMacro" not in model_data:
            continue

        micro_last = model_data["Micro"]["t_last"]
        mm_last = model_data["MicroMacro"]["t_last"]
        if micro_last.size and mm_last.size:
            mu_shift = float(np.mean(mm_last) - np.mean(micro_last))
            q10m, q50m, q90m = _quantiles(micro_last)
            q10a, q50a, q90a = _quantiles(mm_last)

            summary_rows.extend(
                [
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "mean_shift", "value": mu_shift},
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "median_shift", "value": q50a - q50m},
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "q10_shift", "value": q10a - q10m},
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "q90_shift", "value": q90a - q90m},
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "wasserstein_w1", "value": _w1_distance(micro_last, mm_last)},
                    {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_last", "community": None, "metric": "ks_distance", "value": _ks_distance(micro_last, mm_last)},
                ]
            )

            for model_name, arr in (("Micro", micro_last), ("MicroMacro", mm_last)):
                xs, ys = _cdf_points(arr)
                cdf_rows.extend(
                    [
                        {
                            "analysis_uid": analysis_uid,
                            "scenario_id": scenario_slug,
                            "model": model_name,
                            "curve_kind": "cdf",
                            "community": max_comm,
                            "x": float(x),
                            "y": float(y),
                            "bin_left": None,
                            "bin_right": None,
                        }
                        for x, y in zip(xs, ys)
                    ]
                )

                hist, edges = np.histogram(arr, bins=max(2, bins), density=True)
                density_rows.extend(
                    [
                        {
                            "analysis_uid": analysis_uid,
                            "scenario_id": scenario_slug,
                            "model": model_name,
                            "curve_kind": "density_hist",
                            "community": max_comm,
                            "x": float((edges[i] + edges[i + 1]) * 0.5),
                            "y": float(hist[i]),
                            "bin_left": float(edges[i]),
                            "bin_right": float(edges[i + 1]),
                        }
                        for i in range(hist.size)
                    ]
                )

            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            for model_name, arr, color in (("Micro", micro_last, "#2ca02c"), ("MicroMacro", mm_last, "#1f77b4")):
                xs, ys = _cdf_points(arr)
                ax.step(xs, ys, where="post", label=model_name, color=color, linewidth=2.0)
            ax.set_title(f"CDF of T_last ({scenario_slug})")
            ax.set_xlabel("T_last")
            ax.set_ylabel("F(t)")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(scenario_dir / "tlast_cdf.png", dpi=220)
            plt.close(fig)

            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.hist(micro_last, bins=max(2, bins), density=True, alpha=0.45, color="#2ca02c", label="Micro")
            ax.hist(mm_last, bins=max(2, bins), density=True, alpha=0.45, color="#1f77b4", label="MicroMacro")
            ax.set_title(f"Density of T_last ({scenario_slug})")
            ax.set_xlabel("T_last")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(scenario_dir / "tlast_density.png", dpi=220)
            plt.close(fig)

        # Community-wise summaries + boxplot stats
        for c in range(max_comm + 1):
            micro_c = model_data["Micro"]["comm_samples"].get(c, np.array([], dtype=float))
            mm_c = model_data["MicroMacro"]["comm_samples"].get(c, np.array([], dtype=float))
            if micro_c.size and mm_c.size:
                summary_rows.extend(
                    [
                        {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_i", "community": c, "metric": "mean_shift", "value": float(np.mean(mm_c) - np.mean(micro_c))},
                        {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_i", "community": c, "metric": "median_shift", "value": float(np.quantile(mm_c, 0.5) - np.quantile(micro_c, 0.5))},
                        {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "T_i", "community": c, "metric": "q90_shift", "value": float(np.quantile(mm_c, 0.9) - np.quantile(micro_c, 0.9))},
                    ]
                )

            for model_name, vals, n_total, n_reached in (
                ("Micro", micro_c, int(model_data["Micro"]["frame"].shape[0]), int(micro_c.size)),
                ("MicroMacro", mm_c, int(model_data["MicroMacro"]["frame"].shape[0]), int(mm_c.size)),
            ):
                if vals.size == 0:
                    continue
                stats = _boxplot_stats(vals, n_total=n_total, n_reached=n_reached)
                box_rows.append(
                    {
                        "analysis_uid": analysis_uid,
                        "scenario_id": scenario_slug,
                        "model": model_name,
                        "community": c,
                        **stats,
                    }
                )

        # Paired deltas (optional but enabled when keys exist)
        if paired_mode:
            m1 = model_data["Micro"]["frame"].set_index("replication_id")
            m2 = model_data["MicroMacro"]["frame"].set_index("replication_id")
            common_rep = sorted(set(m1.index.dropna().tolist()) & set(m2.index.dropna().tolist()))
            for rep_id in common_rep:
                row_micro = m1.loc[rep_id]
                row_mm = m2.loc[rep_id]
                if hasattr(row_micro, "ndim") and row_micro.ndim > 1:
                    row_micro = row_micro.iloc[0]
                if hasattr(row_mm, "ndim") and row_mm.ndim > 1:
                    row_mm = row_mm.iloc[0]
                for c in range(max_comm + 1):
                    vm = row_micro.get(c, np.nan)
                    va = row_mm.get(c, np.nan)
                    if np.isnan(vm) or np.isnan(va):
                        continue
                    paired_rows.append(
                        {
                            "analysis_uid": analysis_uid,
                            "scenario_id": scenario_slug,
                            "replication_id": int(rep_id),
                            "community": c,
                            "delta_t": float(va - vm),
                        }
                    )

            if paired_rows:
                deltas = [r for r in paired_rows if r["scenario_id"] == scenario_slug]
                for c in range(max_comm + 1):
                    arr = np.array([r["delta_t"] for r in deltas if r["community"] == c], dtype=float)
                    if arr.size == 0:
                        continue
                    summary_rows.extend(
                        [
                            {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "delta_i", "community": c, "metric": "paired_mean", "value": float(np.mean(arr))},
                            {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "delta_i", "community": c, "metric": "paired_median", "value": float(np.quantile(arr, 0.5))},
                            {"analysis_uid": analysis_uid, "scenario_id": scenario_slug, "variable": "delta_i", "community": c, "metric": "paired_q90", "value": float(np.quantile(arr, 0.9))},
                        ]
                    )

        # Boxplot figure
        fig = plt.figure(figsize=(max(8, (max_comm + 1) * 0.8), 4.8))
        ax = fig.add_subplot(111)
        data = []
        positions = []
        labels = []
        for c in range(max_comm + 1):
            m = model_data["Micro"]["comm_samples"].get(c, np.array([], dtype=float))
            a = model_data["MicroMacro"]["comm_samples"].get(c, np.array([], dtype=float))
            if m.size:
                data.append(m)
                positions.append(c * 2 + 1)
                labels.append(f"{c}:M")
            if a.size:
                data.append(a)
                positions.append(c * 2 + 2)
                labels.append(f"{c}:MM")
        if data:
            ax.boxplot(data, positions=positions, widths=0.6, whis=1.5, patch_artist=False)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=70, fontsize=8)
            ax.set_ylabel("Arrival time T_i")
            ax.set_title(f"Community-wise arrival boxplots ({scenario_slug})")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(scenario_dir / "arrival_boxplots.png", dpi=220)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    cdf_df = pd.DataFrame(cdf_rows)
    density_df = pd.DataFrame(density_rows)
    box_df = pd.DataFrame(box_rows)
    paired_df = pd.DataFrame(paired_rows)

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    cdf_df.to_csv(output_dir / "tlast_cdf_points.csv", index=False)
    density_df.to_csv(output_dir / "tlast_density_hist.csv", index=False)
    box_df.to_csv(output_dir / "boxplot_stats.csv", index=False)
    paired_df.to_csv(output_dir / "paired_deltas.csv", index=False)

    # Persist benchmark outputs to DB tables.
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO benchmark_analyses (analysis_uid, created_at, censoring_mode, t_max, paired_mode, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_uid,
                created_at,
                censoring_mode,
                tmax_override,
                1 if paired_mode else 0,
                f"output_dir={str(output_dir)}",
            ),
        )

        if not summary_df.empty:
            cur.executemany(
                """
                INSERT INTO benchmark_summary_metrics
                (analysis_uid, scenario_id, variable, community, metric, value)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.analysis_uid),
                        str(r.scenario_id),
                        str(r.variable),
                        int(r.community) if not pd.isna(r.community) else None,
                        str(r.metric),
                        float(r.value) if not pd.isna(r.value) else None,
                    )
                    for r in summary_df.itertuples(index=False)
                ],
            )

        curve_df = pd.concat([cdf_df, density_df], ignore_index=True) if (not cdf_df.empty or not density_df.empty) else pd.DataFrame()
        if not curve_df.empty:
            cur.executemany(
                """
                INSERT INTO benchmark_curve_points
                (analysis_uid, scenario_id, model, curve_kind, community, x, y, bin_left, bin_right)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.analysis_uid),
                        str(r.scenario_id),
                        str(r.model),
                        str(r.curve_kind),
                        int(r.community) if not pd.isna(r.community) else None,
                        float(r.x) if not pd.isna(r.x) else None,
                        float(r.y) if not pd.isna(r.y) else None,
                        float(r.bin_left) if not pd.isna(r.bin_left) else None,
                        float(r.bin_right) if not pd.isna(r.bin_right) else None,
                    )
                    for r in curve_df.itertuples(index=False)
                ],
            )

        if not box_df.empty:
            cur.executemany(
                """
                INSERT INTO benchmark_boxplot_stats
                (analysis_uid, scenario_id, model, community, n_total, n_reached, q25, q50, q75, iqr, whisker_low, whisker_high, outlier_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.analysis_uid),
                        str(r.scenario_id),
                        str(r.model),
                        int(r.community),
                        int(r.n_total),
                        int(r.n_reached),
                        float(r.q25),
                        float(r.q50),
                        float(r.q75),
                        float(r.iqr),
                        float(r.whisker_low),
                        float(r.whisker_high),
                        int(r.outlier_count),
                    )
                    for r in box_df.itertuples(index=False)
                ],
            )

        if not paired_df.empty:
            cur.executemany(
                """
                INSERT INTO benchmark_paired_deltas
                (analysis_uid, scenario_id, replication_id, community, delta_t)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.analysis_uid),
                        str(r.scenario_id),
                        int(r.replication_id),
                        int(r.community),
                        float(r.delta_t),
                    )
                    for r in paired_df.itertuples(index=False)
                ],
            )

        conn.commit()
    finally:
        conn.close()

    return analysis_uid


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark timing analysis for Micro vs MicroMacro.")
    parser.add_argument("--db", default=None, help="SQLite DB path. Default: from config.json storage.db_path.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for CSV/PNG artifacts. Default: same simulation-specific plots folder as visualization.py.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=None,
        help="Filter by scenario_id (repeatable). If omitted, analyze all scenarios with both models.",
    )
    parser.add_argument(
        "--censoring-mode",
        choices=["completed_only", "right_censor", "keep_flag"],
        default="completed_only",
        help="How to handle runs that do not reach last community before cutoff.",
    )
    parser.add_argument("--paired", action="store_true", help="Enable paired delta analysis by replication_id.")
    parser.add_argument("--bins", type=int, default=30, help="Histogram bins for T_last density.")
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Override censoring horizon (used in right_censor mode). Default: run-specific T_end.",
    )
    args = parser.parse_args()

    default_db_path, default_out_dir = _default_paths_from_config()
    db_path = resolve_path(args.db) if args.db else default_db_path
    out_dir = resolve_path(args.out) if args.out else default_out_dir

    analysis_uid = run_analysis(
        db_path=db_path,
        output_dir=out_dir,
        scenario_ids=args.scenario_id,
        censoring_mode=args.censoring_mode,
        paired_mode=bool(args.paired),
        bins=int(max(2, args.bins)),
        tmax_override=args.tmax,
    )
    print(f"benchmark_analysis_uid={analysis_uid}")
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
