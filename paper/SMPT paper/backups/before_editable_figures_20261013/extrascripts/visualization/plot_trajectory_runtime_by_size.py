#!/usr/bin/env python3
"""Plot trajectory-by-community-size matrices with paired timing shifts.

The left column contains median aggregate infected trajectories with IQR bands;
the right column contains paired arrival-time shifts
``T_i(MicroMacro) - T_i(Full Micro)`` with median/IQR bands.  This script only
reads completed checkpoints (legacy rows or compact numeric columns), ignores
historical consolidated CSVs, and never imports a simulator.

Edit FIGURES below to select scenarios.  Each case points to one canonical
scenario directory under ``output/data``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extrascripts.runners.checkpoint_io import read_checkpoint, model_trajectory_columns

OUT = ROOT / "plots" / "trajectory_runtime_by_size"
PAPER = ROOT / "paper" / "Images" / "size_dynamics"

FULL = "Full Micro"
MM = "Production MicroMacro + independent RNG"
MODELS = (FULL, MM)
SHORT = {FULL: "Full Micro", MM: "MicroMacro (independent RNG)"}
COLORS = {FULL: "#2878b5", MM: "#d0604c"}


@dataclass(frozen=True)
class Case:
    data_dir: str
    benchmark: str | None = None


# Current custom-grid dimensions.  Discovery below uses manifests, so the
# plotting script automatically picks up every completed point without a
# manually maintained list of folder names.
GRID_COMMUNITIES = 10
GRID_SIZES = (50, 100, 250, 1000)
GRID_BETAS = (0.001, 0.003)
GRID_K = (1, 5, 10)
GRID_D = (6, 10)


def discover_figures() -> dict[str, list[Case]]:
    """Discover all current custom-grid scenarios and group comparable panels."""
    groups: dict[str, list[Case]] = {}
    data_root = ROOT / "output" / "data"
    for data_dir in sorted(data_root.iterdir()):
        if not data_dir.is_dir() or not (data_dir / "manifest.json").exists():
            continue
        try:
            config = read_manifest(data_dir).get("benchmark_config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if (
            config.get("family") != "custom_grid"
            or config.get("communities") != GRID_COMMUNITIES
            or config.get("macro_graph_type") != "chain"
            or config.get("community_size") not in GRID_SIZES
            or config.get("inter_links") not in GRID_K
            or config.get("beta") not in GRID_BETAS
            or not config.get("store_trajectories", False)
        ):
            continue
        micro_type = config.get("micro_graph_type")
        if micro_type == "complete" and config.get("mean_internal_degree") is None:
            group = f"chain_cliques_grid_beta{config['beta']:g}".replace(".", "p")
        elif micro_type == "random_connected":
            degree = config.get("mean_internal_degree")
            if degree not in GRID_D:
                continue
            group = (
                f"chain_random_connected_grid_d{degree:g}_beta{config['beta']:g}"
            ).replace(".", "p")
        else:
            continue
        groups.setdefault(group, []).append(
            Case(str(data_dir.relative_to(ROOT)))
        )

    def sort_key(case: Case):
        config = read_manifest(ROOT / case.data_dir)["benchmark_config"]
        # Numerical order is deliberate: k=1 rows precede k=5 and k=10;
        # within each k, community sizes increase from 50 to 1000.
        return (int(config["inter_links"]), int(config["community_size"]))

    for cases in groups.values():
        cases.sort(key=sort_key)

    def group_sort(item):
        name, cases = item
        config = read_manifest(ROOT / cases[0].data_dir)["benchmark_config"]
        topology_order = 0 if config.get("micro_graph_type") == "complete" else 1
        degree = float(config.get("mean_internal_degree") or 0.0)
        return topology_order, float(config["beta"]), degree

    return dict(sorted(groups.items(), key=group_sort))

_CASE_CACHE = {}
_PAYLOAD_CACHE = {}


def read_manifest(data_dir: Path) -> dict:
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def scenario(data_dir: Path, benchmark: str) -> dict:
    manifest = read_manifest(data_dir)
    config = manifest.get("benchmark_config")
    if not config:
        raise RuntimeError(f"Missing benchmark_config in {data_dir / 'manifest.json'}")
    return config


def checkpoint_observable_rows(data_dir: Path, key: str) -> pd.DataFrame:
    """Checkpoints are authoritative; load small observables together once."""
    cache_key = str(data_dir.resolve())
    cached = _PAYLOAD_CACHE.setdefault(cache_key, {})
    if key not in cached:
        rows = {name: [] for name in {"runtime", "community_arrivals", key}}
        for path in sorted((data_dir / "checkpoints").glob("rep_*.json.gz")):
            payload = read_checkpoint(path)
            if "error" in payload:
                continue
            for name in rows:
                rows[name].extend(payload.get(name, []))
        cached.update({name: pd.DataFrame(values) for name, values in rows.items()})
    return cached[key].copy()


def runtime_rows(data_dir: Path, benchmark: str) -> pd.DataFrame:
    rows = checkpoint_observable_rows(data_dir, "runtime")
    # One canonical folder is one scenario. A reused legacy pair can retain an
    # old benchmark label alongside new pairs with the canonical label.
    # Filtering by that label would silently discard valid reused pairs.
    if rows.empty:
        raise RuntimeError(f"No runtime rows for {benchmark} in {data_dir / 'checkpoints'}")
    return rows


def arrival_rows(data_dir: Path, benchmark: str) -> pd.DataFrame:
    rows = checkpoint_observable_rows(data_dir, "community_arrivals")
    if rows.empty:
        raise RuntimeError(f"No arrival rows for {benchmark} in {data_dir / 'checkpoints'}")
    return rows


def checkpoint_paths(data_dir: Path, benchmark: str, expected_reps: set[int]) -> list[Path]:
    root = data_dir / "checkpoints"
    paths = sorted(root.glob("rep_*.json.gz"))
    found = {int(path.stem.split("_")[-1].split(".")[0]) for path in paths}
    missing = sorted(expected_reps - found)
    if missing:
        raise RuntimeError(f"{benchmark}: missing {len(missing)} checkpoints; first IDs: {missing[:8]}")
    by_id = {int(path.stem.split("_")[-1].split(".")[0]): path for path in paths}
    return [by_id[rep] for rep in sorted(expected_reps)]


def aggregate_stack(data_dir: Path, benchmark: str, reps: list[int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    paths = checkpoint_paths(data_dir, benchmark, set(reps))
    series = {model: [] for model in MODELS}
    for index, path in enumerate(paths, 1):
        payload = read_checkpoint(path)
        termination = {
            row.get("simulator"): row for row in payload.get("termination", [])
        }
        for model in MODELS:
            part = pd.DataFrame(model_trajectory_columns(payload, model))
            if part.empty:
                raise RuntimeError(f"{path}: no trajectory for {model}")
            total = part.groupby("time", sort=True)["I"].sum()
            terminal = termination.get(model)
            if terminal:
                config = payload.get("benchmark_config", {})
                try:
                    horizon = float(config["T_end"])
                    dt_out = float(config["dt_out"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(f"{path}: termination lacks a valid reporting horizon") from error
                # The runner intentionally omits held rows after termination.
                # Reconstruct them only in memory: the terminal state is
                # absorbing/no-event or the state held through this T_end.
                stop_time = float(terminal["stop_time"])
                terminal_I = sum(int(state["I"])
                                 for state in terminal["final_state_by_community"])
                extension = pd.Series([terminal_I, terminal_I],
                                      index=[stop_time, horizon], dtype=float)
                total = pd.concat([total, extension]).groupby(level=0).last().sort_index()
                grid = np.arange(0.0, horizon + 0.5 * dt_out, dt_out)
                total = total.reindex(total.index.union(grid)).sort_index().ffill().reindex(grid)
            times = total.index.to_numpy(float)
            series[model].append((times, total.to_numpy(float)))
        if index % 25 == 0 or index == len(paths):
            print(f"[{benchmark}] checkpoints {index}/{len(paths)}", flush=True)
    # Selective horizon extension intentionally leaves already-completed pairs
    # at their old observation horizon. Plot their common exact prefix only;
    # never fill or extrapolate post-horizon states.
    common_end = min(float(times[-1]) for values in series.values() for times, _ in values)
    template = None
    stacked = {}
    heterogeneous = False
    for model, values in series.items():
        vectors = []
        for times, infected in values:
            keep = times <= common_end + 1e-9
            prefix_t, prefix_i = times[keep], infected[keep]
            if template is None:
                template = prefix_t
            elif not np.array_equal(prefix_t, template):
                raise RuntimeError(f"{benchmark}: incompatible reporting grids before common horizon")
            heterogeneous |= float(times[-1]) > common_end + 1e-9
            vectors.append(prefix_i)
        stacked[model] = np.vstack(vectors)
    if heterogeneous:
        print(
            f"[{benchmark}] trajectories have mixed horizons; plotting only the "
            f"common exact prefix through t={common_end:g}", flush=True,
        )
    return template, stacked


def row_label(meta: dict, replications: int, trajectory_horizon: float) -> str:
    graph = "complete" if meta["micro_graph_type"] == "complete" else "random-connected"
    degree = "" if graph == "complete" else rf", $d\approx{meta.get('mean_internal_degree') or '?'}$"
    return (rf"$n={meta['community_size']}$, {graph}{degree}, $L={meta['communities']}$, "
            rf"$k={meta['inter_links']}$, $\beta={meta['beta']:g}$, "
            rf"$T_{{arr,max}}={meta['T_end']:g}$, $T_{{traj}}\leq{trajectory_horizon:g}$, $N={replications}$")


def make_figure(name: str, cases: list[Case]) -> dict:
    prepared = []
    for case in cases:
        cache_key = (case.data_dir, case.benchmark)
        if cache_key in _CASE_CACHE:
            prepared.append(_CASE_CACHE[cache_key])
            continue
        data_dir = ROOT / case.data_dir
        meta = scenario(data_dir, case.benchmark)
        benchmark_name = case.benchmark or str(meta.get("name", data_dir.name))
        runtime = runtime_rows(data_dir, benchmark_name)
        arrivals = arrival_rows(data_dir, benchmark_name)
        complete = runtime.groupby("replication_id")["simulator"].nunique()
        reps = sorted(int(rep) for rep in complete[complete.eq(len(MODELS))].index)
        if not reps:
            raise RuntimeError(f"{benchmark_name}: no complete simulator pairs")
        t, stacks = aggregate_stack(data_dir, benchmark_name, reps)
        item = (case, benchmark_name, meta, runtime[runtime.replication_id.isin(reps)], arrivals,
                reps, t, stacks)
        _CASE_CACHE[cache_key] = item
        prepared.append(item)

    fig, axes = plt.subplots(len(prepared), 2, figsize=(12.2, 3.05 * len(prepared)),
                             squeeze=False, gridspec_kw={"width_ratios": [3.25, 1.15]})
    manifest_rows = []
    for row, (case, benchmark_name, meta, runtime, arrivals, reps, t, stacks) in enumerate(prepared):
        ax = axes[row, 0]
        for model in MODELS:
            q25, median, q75 = np.quantile(stacks[model], [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(t, q25, q75, color=COLORS[model], alpha=0.18, linewidth=0)
            ax.plot(t, median, color=COLORS[model], lw=1.65,
                    label=SHORT[model] if row == 0 else None)
        ax.set_ylabel(r"total infected $I(t)$")
        ax.set_title(row_label(meta, len(reps), float(t[-1])), loc="left", fontsize=10)
        ax.grid(alpha=0.22)
        if row == 0:
            ax.legend(frameon=False, ncol=2)

        ax = axes[row, 1]
        paired = arrivals.pivot_table(
            index=["replication_id", "community"], columns="simulator",
            values="first_infection_time", aggfunc="first"
        ).reset_index()
        paired = paired[paired.replication_id.isin(reps)]
        if FULL not in paired or MM not in paired:
            raise RuntimeError(f"{benchmark_name}: paired arrival columns are incomplete")
        paired["error"] = paired[MM] - paired[FULL]
        stats = paired.dropna(subset=["error"]).groupby("community")["error"].agg(
            q25=lambda x: np.quantile(x, 0.25),
            median="median",
            q75=lambda x: np.quantile(x, 0.75),
        ).reset_index()
        ax.fill_between(stats["community"], stats["q25"], stats["q75"],
                        color=COLORS[MM], alpha=0.22, linewidth=0)
        ax.plot(stats["community"], stats["median"], marker="o", ms=4.5,
                color=COLORS[MM], lw=1.55)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel(r"$T_i^{MM}-T_i^{Micro}$")
        ax.set_xlabel("community index")
        ax.grid(axis="y", alpha=0.22)
        manifest_rows.append({"data_dir": case.data_dir, "benchmark": benchmark_name,
                              "replications": reps, "scenario": meta,
                              "arrival_shift_median_by_community": {
                                  str(int(c)): float(v)
                                  for c, v in zip(stats["community"], stats["median"])}})

    axes[-1, 0].set_xlabel("time")
    fig.suptitle("Epidemic dynamics and paired arrival-time shift by community size", y=1.002)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f"{name}.png", OUT / f"{name}.pdf",
               PAPER / f"{name}.png", PAPER / f"{name}.pdf"]
    for target in targets:
        fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {targets[0]}", flush=True)
    return {"name": name, "targets": [str(path.relative_to(ROOT)) for path in targets],
            "rows": manifest_rows}


def main() -> None:
    figures = discover_figures()
    if not figures:
        raise RuntimeError("No completed custom-grid scenarios found under output/data")
    print(
        "Discovered grid: "
        + ", ".join(f"{name} ({len(cases)} cases)" for name, cases in figures.items()),
        flush=True,
    )
    records = [make_figure(name, cases) for name, cases in figures.items()]
    manifest = {"script": str(Path(__file__).relative_to(ROOT)),
                "note": "Right panels show paired arrival shift MM minus Full Micro; subtitles disclose non-controlled settings.",
                "grid": {
                    "communities": GRID_COMMUNITIES,
                    "sizes": GRID_SIZES,
                    "betas": GRID_BETAS,
                    "k": GRID_K,
                    "d": GRID_D,
                },
                "figures": records}
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(records)} figures; manifest: {OUT / 'plot_manifest.json'}")


if __name__ == "__main__":
    main()
