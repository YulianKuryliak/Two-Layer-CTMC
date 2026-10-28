"""Collect a calibrated SI topology/size matrix without changing production sources.

Default launch: python extrascripts/runners/run_topology_size_matrix.py
See docs/topology_size_matrix.md for the scientific design and NPZ schema.
"""
from __future__ import annotations

import os
# Set before numpy import; each worker uses a single BLAS thread.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import asdict, replace
import csv
import hashlib
import inspect
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import random
import signal
import statistics
import sys
import time

import networkx as nx
import numpy as np
from extrascripts.runners import run_original_benchmarks_independent_rng as legacy
import two_layer_ctmc.network as network

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "extrascripts/runners/configs/topology_size_matrix.json"
VERSION = 2


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seeds(cfg, cell, phase, rep):
    def one(label):
        value = f"{cfg['seed']}:{cell['id']}:{phase}:{rep}:{label}"
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (2**31)
    return one("network"), one("events")


def cells_for(cfg):
    return [dict(id=f"L{cfg['communities']}_n{n}_d{d}_k{k}", n=n, d=d, k=k)
            for n in cfg["sizes"] for d in cfg["degrees"] for k in cfg["inter_links"]]


def validate_config(cfg):
    expected = set(load(DEFAULT_CONFIG))
    if set(cfg) != expected:
        raise ValueError(f"Config keys differ: missing={expected-set(cfg)}, extra={set(cfg)-expected}")
    for name in ("communities", "runs", "calibration_runs", "report_points", "workers", "network_timeout_seconds", "pilot_horizon_attempts"):
        if not isinstance(cfg[name], int) or cfg[name] < (2 if name in ("communities", "report_points") else 1):
            raise ValueError(f"Invalid {name}")
    for name in ("sizes", "degrees", "inter_links"):
        if not cfg[name] or len(set(cfg[name])) != len(cfg[name]):
            raise ValueError(f"Empty/duplicate {name}")
    if not 500 <= cfg["target_time"] < cfg["T_end"] <= 10000:
        raise ValueError("Require 500 <= target_time < T_end <= 10000")
    if cfg["rho"] <= 0 or any(not 0 < x < cfg["rho"] for x in cfg["refinement_rho"]):
        raise ValueError("Refinement rho must be positive and smaller than base rho")
    if len(set(cfg["refinement_rho"])) != len(cfg["refinement_rho"]):
        raise ValueError("Duplicate refinement rho")
    if cfg["refinement_runs"] is not None and (not isinstance(cfg["refinement_runs"], int) or cfg["refinement_runs"] < 0):
        raise ValueError("refinement_runs must be nonnegative")
    if not isinstance(cfg["seed"], int) or not isinstance(cfg["skip_unused_spectral_descriptors"], bool):
        raise ValueError("Invalid seed or descriptor flag")
    for c in cells_for(cfg):
        if not isinstance(c["n"], int) or c["n"] < 3 or not isinstance(c["k"], int) or not 1 <= c["k"] <= c["n"]**2:
            raise ValueError(f"Invalid cell {c}")
        if c["d"] != "complete" and (not isinstance(c["d"], (int, float)) or not 2 <= c["d"] <= c["n"]-1):
            raise ValueError(f"Invalid degree {c}")


def benchmark(cfg, cell, beta, horizon, rho):
    return replace(legacy.BENCHMARKS[1], name=cell["id"], family="topology_size_matrix",
        communities=cfg["communities"], community_size=cell["n"], inter_links=cell["k"],
        macro_graph_type="chain", micro_graph_type="complete" if cell["d"] == "complete" else "random_connected",
        edge_prob=1.0 if cell["d"] == "complete" else cell["d"]/(cell["n"]-1),
        mean_internal_degree=None if cell["d"] == "complete" else cell["d"],
        beta=beta, gamma=0.0, model=2, macro_T=1.0, tau_micro=rho/beta,
        T_end=horizon, original_T_end=horizon, dt_out=horizon/(cfg["report_points"]-1),
        initial_node=0, store_trajectories=False, replications=cfg["runs"])


@contextmanager
def network_context(skip_descriptors, seconds):
    """Leave topology generation intact; omit descriptors unused by both baselines."""
    old_alpha = network._compute_micro_alphas
    old_boundary = network._compute_boundary_descriptors
    old_handler = None
    if seconds and hasattr(signal, "SIGALRM"):
        def expire(*_):
            raise TimeoutError("Network generation exceeded its wall-time budget")
        old_handler = signal.signal(signal.SIGALRM, expire)
        signal.alarm(seconds)
    if skip_descriptors:
        signature = inspect.signature(legacy.Orchestrator)
        if any(signature.parameters[x].default != "off" for x in ("alpha_mode", "correction_mode")):
            raise RuntimeError("Unused-descriptor optimization requires both correction modes off")
        network._compute_micro_alphas = lambda gs: [1.0] * len(gs)
        # Explicit placeholders, NOT measured spectral or boundary statistics.
        network._compute_boundary_descriptors = lambda gs, *_: ([0] * len(gs), [1.0] * len(gs))
    try:
        yield
    finally:
        network._compute_micro_alphas = old_alpha
        network._compute_boundary_descriptors = old_boundary
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def make_network(cfg, b, seed):
    started = time.perf_counter()
    with network_context(cfg["skip_unused_spectral_descriptors"], cfg["network_timeout_seconds"]):
        graphs, full, W = legacy._generate_network(b, seed)
    validation = legacy._validate_network(b, graphs, full, W)
    if not validation["topology_valid"] or not validation["micro_graphs_connected"]:
        raise RuntimeError(f"Network validation failed: {validation}")
    validation.update(generation_seconds=time.perf_counter()-started,
        actual_degrees=[2*g.number_of_edges()/g.number_of_nodes() for g in graphs],
        unused_descriptors_skipped=cfg["skip_unused_spectral_descriptors"])
    return graphs, full, W, validation


def full_run(b, graphs, full, event_seed):
    return legacy.MicroSimulator(full_graph=full, comm_nodes=[list(g.nodes()) for g in graphs],
        infection_rate=b.beta, recovery_rate=0, model=2).run(T_end=b.T_end, dt_out=b.dt_out,
            initial_node=0, rng=random.Random(event_seed))


def infection_events(result, full_model, horizon):
    if full_model:
        events = [(e["time"], e["dst_community"], e["dst_node"]) for e in result["transmission_events"]
                  if e["time"] <= horizon and e["dst_node"] is not None]
    else:
        events = [(e["time"], e["community"], e["node"]) for e in result[3]
                  if e.get("event_type") in ("seed", "infection", "transfer")
                  and e.get("node") is not None and e["time"] <= horizon]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    if len({e[2] for e in events}) != len(events):
        raise RuntimeError("Duplicate infection in SI event stream")
    if not events or events[0][0] != 0:
        raise RuntimeError("Missing initial infection")
    return events


def observables(events, b):
    arrivals = [None] * b.communities
    for t, community, _ in events:
        if arrivals[community] is None:
            arrivals[community] = t
    total = b.communities*b.community_size
    count99 = math.ceil(.99*total)
    count50 = math.ceil(.50*total)
    return dict(arrivals=arrivals, completion=all(t is not None for t in arrivals),
        T_last=max(arrivals) if all(t is not None for t in arrivals) else None,
        T_50=events[count50-1][0] if len(events) >= count50 else None,
        T_99=events[count99-1][0] if len(events) >= count99 else None,
        all_nodes_infected=len(events) == total, final_infected=len(events),
        T_all=events[-1][0] if len(events) == total else None)


def theory(cell, L):
    """Only a starting pilot horizon, never a fitted or guaranteed arrival time."""
    growth = cell["n"] if cell["d"] == "complete" else max(1, cell["d"]-2)
    return (L-1)*(1/cell["k"] + math.log(cell["n"])/growth) + 2*math.log(cell["n"])/growth


def calibrate(cfg, cell, out):
    directory = Path(out)/"calibration"/cell["id"]
    final = directory/"calibration.json"
    if final.exists():
        return load(final)
    refs = []
    for rep in range(cfg["calibration_runs"]):
        checkpoint = directory/f"full_{rep:03d}.json"
        if checkpoint.exists():
            refs.append(load(checkpoint))
            continue
        netseed, evseed = seeds(cfg, cell, "calibration", rep)
        H = max(32.0, 4*theory(cell, cfg["communities"]))
        b = benchmark(cfg, cell, 1.0, H, cfg["rho"])
        graphs, full, _, validation = make_network(cfg, b, netseed)
        started = time.perf_counter()
        for attempt in range(cfg["pilot_horizon_attempts"]):
            b = replace(b, T_end=H, dt_out=H/(cfg["report_points"]-1))
            result = full_run(b, graphs, full, evseed)
            obs = observables(infection_events(result, True, H), b)
            del result
            if obs["T_99"] is not None and obs["completion"]:
                break
            H *= 4
        else:
            raise RuntimeError("Full calibration remained censored; no beta was fitted")
        record = dict(replication=rep, network_seed=netseed, event_seed=evseed,
            beta_ref=1.0, reference_u=max(obs["T_last"], obs["T_99"]), horizon_u=b.T_end,
            observables=obs, network_validation=validation, runtime_seconds=time.perf_counter()-started)
        write_json(checkpoint, record)
        refs.append(record)
        del graphs, full
    C = statistics.median(v["reference_u"] for v in refs)
    beta = C/cfg["target_time"]
    record = dict(cell=cell, beta=beta, median_reference_u=C,
        target_time=cfg["target_time"], T_end=cfg["T_end"], U_end=beta*cfg["T_end"],
        rho=cfg["rho"], tau_micro=cfg["rho"]/beta, macro_T=1.0,
        calibration_runs=len(refs), reference="max(T_last, T_99) of Full SI only",
        predicted_steps=math.ceil(beta*cfg["T_end"]/cfg["rho"]))
    write_json(final, record)
    return record


def model_schedule(cfg, cell, rep):
    models = [("full", None), ("mm", cfg["rho"])]
    refinement_runs = cfg["runs"] if cfg["refinement_runs"] is None else cfg["refinement_runs"]
    if (rep < refinement_runs and cell["n"] in cfg["refinement_sizes"]
            and cell["d"] in cfg["refinement_degrees"] and cell["k"] in cfg["refinement_links"]):
        models += [("mm", rho) for rho in cfg["refinement_rho"]]
    return [("full" if kind == "full" else f"mm_rho{rho:g}", kind, rho) for kind, rho in models]


def save_trajectory(path, events, b, C, points):
    times = np.array([e[0] for e in events], dtype=float)
    communities = np.array([e[1] for e in events], dtype=np.int32)
    nodes = np.array([e[2] for e in events], dtype=np.int64)
    grid = np.linspace(0, b.T_end, points)
    I = np.stack([np.searchsorted(times[communities == c], grid, side="right") for c in range(b.communities)]).astype(np.int32)
    S = b.community_size-I
    if np.any(S < 0) or np.any(I < 0):
        raise RuntimeError("Invalid trajectory counts")
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, time=grid, time_u=b.beta*grid,
            time_relative=b.beta*grid/C, S=S, I=I, R=np.zeros_like(I),
            infection_time=times, infection_u=b.beta*times,
            infection_community=communities, infection_node=nodes)
    tmp.replace(path)


def run_pair(cfg, cell, rep, out):
    out = Path(out)
    cal = load(out/"calibration"/cell["id"]/"calibration.json")
    directory = out/"checkpoints"/cell["id"]/f"rep_{rep:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    schedule = model_schedule(cfg, cell, rep)
    if all((directory/f"{label}.json").exists() and (directory/f"{label}.npz").exists() for label, *_ in schedule):
        return dict(cell=cell["id"], replication=rep, reused=True)
    netseed, evseed = seeds(cfg, cell, "main", rep)
    b = benchmark(cfg, cell, cal["beta"], cfg["T_end"], cfg["rho"])
    graphs, full, W, validation = make_network(cfg, b, netseed)
    # Save true per-node degrees/boundary membership, not placeholder descriptors.
    nodes = np.array(sorted(full.nodes()), dtype=np.int64)
    boundary = np.zeros((len(nodes), b.communities), dtype=np.int32)
    internal_degree = np.array([graphs[int(node)//b.community_size].degree(int(node)) for node in nodes], dtype=np.int32)
    for u, v in full.edges():
        cu, cv = u//b.community_size, v//b.community_size
        if cu != cv:
            boundary[u, cv] += 1
            boundary[v, cu] += 1
    tmp = directory/"network.tmp"
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, node=nodes, community=nodes//b.community_size,
            internal_degree=internal_degree, external_degree_by_community=boundary, W=W)
    tmp.replace(directory/"network.npz")
    for label, kind, rho in schedule:
        path = directory/f"{label}.json"
        trajectory = directory/f"{label}.npz"
        if path.exists() and trajectory.exists():
            continue
        actual = b if kind == "full" else replace(b, tau_micro=rho/b.beta)
        started = time.perf_counter()
        if kind == "full":
            result = full_run(actual, graphs, full, evseed)
        else:
            result = legacy._run_experimental_micromacro(actual, graphs, full, W, evseed, independent=True)
        runtime = time.perf_counter()-started
        events = infection_events(result, kind == "full", b.T_end)
        del result
        obs = observables(events, b)
        row = dict(cell=cell["id"], n=cell["n"], d=cell["d"], k=cell["k"], communities=b.communities,
            replication=rep, simulator=label, beta=b.beta, gamma=0, macro_T=1,
            tau_micro=None if kind == "full" else actual.tau_micro, rho=rho,
            T_end=b.T_end, U_end=b.beta*b.T_end, median_reference_u=cal["median_reference_u"],
            network_seed=netseed, event_seed=evseed,
            community_rng_seeds=None if kind == "full" else [evseed+10000+1009*c for c in range(b.communities)],
            runtime_seconds=runtime, network_validation=validation,
            trajectory_path=str(trajectory.relative_to(out)), **obs)
        row["arrivals_u"] = [None if t is None else b.beta*t for t in obs["arrivals"]]
        row["T_last_u"] = None if obs["T_last"] is None else b.beta*obs["T_last"]
        save_trajectory(trajectory, events, b, cal["median_reference_u"], cfg["report_points"])
        write_json(path, row)  # Model checkpoint committed only after trajectory exists.
        print(f"  {cell['id']} r{rep} {label}: arrival={obs['T_last']} runtime={runtime:.1f}s", flush=True)
    write_json(directory/"done.json", dict(models=[s[0] for s in schedule], network_seed=netseed, event_seed=evseed))
    return dict(cell=cell["id"], replication=rep, reused=False)


def source_fingerprint():
    files = [Path(__file__), Path(legacy.__file__)] + sorted((ROOT/"two_layer_ctmc").rglob("*.py"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def gates():
    passed, reason = legacy._regression_gate()
    if not passed:
        raise RuntimeError(f"Shared RNG regression FAIL: {reason}")
    cfg = load(DEFAULT_CONFIG)
    cfg.update(communities=3, report_points=101)
    cell = dict(id="gate", n=8, d=4, k=2)
    b = benchmark(cfg, cell, .02, 400, .005)
    with network_context(False, 0):
        original = legacy._generate_network(b, 71011)
    with network_context(True, 0):
        fast = legacy._generate_network(b, 71011)
    assert list(original[1].edges(data=True)) == list(fast[1].edges(data=True))
    np.testing.assert_array_equal(original[2], fast[2])
    reference = legacy._run_experimental_micromacro(b, *original, 71012, independent=True)
    accelerated = legacy._run_experimental_micromacro(b, *fast, 71012, independent=True)
    assert reference == accelerated, "Descriptor bypass altered independent-RNG paths"
    scaled = replace(b, beta=2*b.beta, tau_micro=b.tau_micro/2, T_end=b.T_end/2, dt_out=b.dt_out/2)
    for is_full in (True, False):
        first = full_run(b, fast[0], fast[1], 71012) if is_full else accelerated
        second = full_run(scaled, fast[0], fast[1], 71012) if is_full else legacy._run_experimental_micromacro(scaled, *fast, 71012, independent=True)
        ev1, ev2 = infection_events(first, is_full, b.T_end), infection_events(second, is_full, scaled.T_end)
        assert [(x[1],x[2]) for x in ev1] == [(x[1],x[2]) for x in ev2], "Scaled event identities differ"
        np.testing.assert_allclose([x[0] for x in ev1], [2*x[0] for x in ev2], rtol=1e-9, atol=1e-9)
    with legacy._rng_ownership_patch(independent=True):
        source = legacy.MicroModel(.1, 0, model=2, rng=random.Random(51))
        source.add_node(0)
        clone = source.clone()
        assert clone.rng is not source.rng and clone.rng.getstate() == source.rng.getstate()
        source.rng.random()
        clone.refresh_from(source)
        assert clone.rng.getstate() == source.rng.getstate()
    return dict(shared_rng=reason, descriptor_bypass="pathwise PASS", time_scaling="Full and independent MM PASS", rng_clone_refresh="PASS")


def dispatch(function, jobs, cfg, phase):
    """Dense n>=250 clique jobs run alone, with no other workers resident."""
    failures = []
    done = 0
    small = [j for j in jobs if not (j[1]["d"] == "complete" and j[1]["n"] >= 250)]
    large = [j for j in jobs if j[1]["d"] == "complete" and j[1]["n"] >= 250]
    for group, workers in ((small, cfg["workers"]), (large, 1)):
        if not group:
            continue
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as pool:
            iterator = iter(group)
            pending = {}
            def submit_one():
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(function, *job)] = job
            for _ in range(workers):
                submit_one()
            while pending:
                finished, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                if not finished:
                    active = ", ".join(j[1]["id"] for j in pending.values())
                    print(f"[{phase}] {done}/{len(jobs)} done, {len(jobs)-done} remaining; active: {active}", flush=True)
                for future in finished:
                    job = pending.pop(future)
                    done += 1
                    try:
                        future.result()
                        outcome = "OK"
                    except Exception as exc:
                        outcome = f"FAILED: {type(exc).__name__}: {exc}"
                        failures.append(dict(cell=job[1]["id"], task=clean(job[2:-1]), reason=outcome))
                    print(f"[{phase}] {done}/{len(jobs)} {job[1]['id']} {outcome}", flush=True)
                    submit_one()
    return failures


def export(out, cfg, cells):
    records = [load(p) for p in sorted((out/"checkpoints").glob("*/rep_*/*.json")) if p.name != "done.json"]
    excluded = {"arrivals", "arrivals_u", "community_rng_seeds", "network_validation"}
    fields = sorted({k for row in records for k in row if k not in excluded})
    if records:
        with (out/"runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        with (out/"runtime.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cell", "replication", "simulator", "runtime_seconds"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        with (out/"community_arrivals.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cell","replication","simulator","community","arrival","arrival_u","observed","delta","delta_u","T_end","U_end"])
            writer.writeheader()
            for row in records:
                for c, t in enumerate(row["arrivals"]):
                    prev = 0 if c == 0 else row["arrivals"][c-1]
                    delta = t-prev if t is not None and prev is not None else None
                    writer.writerow(dict(cell=row["cell"], replication=row["replication"], simulator=row["simulator"], community=c,
                        arrival=t, arrival_u=row["arrivals_u"][c], observed=t is not None,
                        delta=delta, delta_u=None if delta is None else row["beta"]*delta, T_end=row["T_end"], U_end=row["U_end"]))
    summary = []
    model_summary = []
    for cell in cells:
        rs = [v for v in records if v["cell"] == cell["id"]]
        expected = sum(len(model_schedule(cfg, cell, rep)) for rep in range(cfg["runs"]))
        expected_ids = {}
        for rep in range(cfg["runs"]):
            for label, *_ in model_schedule(cfg, cell, rep):
                expected_ids.setdefault(label, set()).add(rep)
        for label, ids in expected_ids.items():
            saved = [v for v in rs if v["simulator"] == label]
            saved_ids = {v["replication"] for v in saved if (out/v["trajectory_path"]).exists()}
            model_summary.append(dict(cell=cell["id"], simulator=label, expected=len(ids), saved=len(saved_ids),
                collection_complete=saved_ids == ids and len(saved) == len(ids),
                completed_epidemics=sum(v["completion"] for v in saved)))
        summary.append(dict(cell=cell["id"], expected_simulations=expected, saved_simulations=len(rs),
            completed_simulations=sum(v["completion"] for v in rs), reached_99pct=sum(v["T_99"] is not None for v in rs)))
    write_json(out/"completion.json", summary)
    with (out/"completion_by_model.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_summary[0]))
        writer.writeheader()
        writer.writerows(model_summary)
    return all(v["collection_complete"] for v in model_summary)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT/"output/topology_size_matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--refinement-runs", type=int, help="Override control replications; config null follows --runs")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(gates(), indent=2))
        return
    cfg = load(args.config)
    if args.runs is not None:
        cfg["runs"] = args.runs
    if args.refinement_runs is not None:
        cfg["refinement_runs"] = args.refinement_runs
    if args.workers is not None:
        cfg["workers"] = args.workers
    validate_config(cfg)
    cells = cells_for(cfg)
    count = sum(len(model_schedule(cfg, cell, rep)) for cell in cells for rep in range(cfg["runs"]))
    print(f"{len(cells)} cells; {cfg['runs']} paired replications/cell; {count} main/audit simulations; {cfg['calibration_runs']} Full calibration runs/cell", flush=True)
    print(f"Target Full max(T_last,T_99) median={cfg['target_time']}; horizon={cfg['T_end']}; rho=beta*tau={cfg['rho']}", flush=True)
    if args.plan_only:
        for cell in cells:
            estimate = theory(cell, cfg["communities"])/cfg["target_time"]
            print(f"{cell['id']}: theoretical beta scale ~{estimate:.5g}; actual beta comes from Full calibration")
        return
    out = args.output.resolve()
    immutable = {k:v for k,v in cfg.items() if k not in ("workers", "runs", "refinement_runs")}
    targets = dict(runs=cfg["runs"], refinement_runs=min(cfg["runs"], cfg["runs"] if cfg["refinement_runs"] is None else cfg["refinement_runs"]))
    identity = dict(version=VERSION, config=immutable, sources=source_fingerprint(),
        python=platform.python_version(), numpy=np.__version__, networkx=nx.__version__)
    manifest_path = out/"benchmark_manifest.json"
    if manifest_path.exists():
        if not args.resume and not args.export_only:
            raise ValueError("Output already exists. Use --resume or a fresh --output")
        manifest = load(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Config/source/dependency mismatch: use a new output directory")
        previous = manifest["replication_targets"]
        if any(targets[k] < previous[k] for k in targets):
            raise ValueError(f"Cannot reduce existing replication targets {previous}; retain them on resume/export")
        if args.export_only and targets != previous:
            raise ValueError("Use --resume, not --export-only, to extend replication targets")
        if targets != previous:
            manifest.setdefault("extension_history", []).append(dict(previous=previous, target=targets,
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            manifest["replication_targets"] = targets
    else:
        if args.export_only:
            raise ValueError("No existing manifest to export")
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output is nonempty without a valid manifest")
        print("Checking shared RNG, independent clone state, time scaling and descriptor bypass...", flush=True)
        checks = gates()
        manifest = dict(identity=identity, identity_sha256=digest(identity), gates=checks, replication_targets=targets,
            status="prepared", cells=cells, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            interpretation="Total Production approximation error; refinement audit is required before attributing effects to abstraction alone")
        write_json(manifest_path, manifest)
        instructions = ROOT/"docs/topology_size_matrix.md"
        if instructions.exists():
            (out/"README.md").write_text(instructions.read_text(encoding="utf-8"), encoding="utf-8")
    if args.export_only:
        export(out, cfg, cells)
        return
    print("Regression gate: PASS (verified manifest reused on resume)" if args.resume else "Regression gate: PASS", flush=True)
    manifest["status"] = "calibrating"
    write_json(manifest_path, manifest)
    jobs = [(cfg, c, str(out)) for c in cells if not (out/"calibration"/c["id"]/"calibration.json").exists()]
    failures = dispatch(calibrate, jobs, cfg, "calibration")
    calibrated = [c for c in cells if (out/"calibration"/c["id"]/"calibration.json").exists()]
    parameters = [load(out/"calibration"/c["id"]/"calibration.json") for c in calibrated]
    write_json(out/"calibrated_parameters.json", parameters)
    if parameters:
        flattened = [dict(cell=p["cell"]["id"], n=p["cell"]["n"], d=p["cell"]["d"], k=p["cell"]["k"],
                          **{k:v for k,v in p.items() if k != "cell"}) for p in parameters]
        with (out/"calibrated_parameters.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    write_json(out/"calibration_failures.json", failures)
    if failures:
        manifest["status"] = "calibration_incomplete"
        write_json(manifest_path, manifest)
        print("Some cells could not be calibrated; collecting the successful cells. --resume retries missing cells.", flush=True)
    if args.calibrate_only:
        manifest["status"] = "calibration_incomplete" if failures else "calibrated"
        write_json(manifest_path, manifest)
        print(f"Calibration saved to {out}; continue with --resume", flush=True)
        if failures:
            raise RuntimeError("Calibration incomplete; inspect calibration_failures.json")
        return
    manifest["status"] = "collecting"
    write_json(manifest_path, manifest)
    jobs = []
    for cell in calibrated:
        for rep in range(cfg["runs"]):
            directory = out/"checkpoints"/cell["id"]/f"rep_{rep:05d}"
            if not all((directory/f"{label}.json").exists() and (directory/f"{label}.npz").exists() for label, *_ in model_schedule(cfg, cell, rep)):
                jobs.append((cfg, cell, rep, str(out)))
    failures = dispatch(run_pair, jobs, cfg, "collection")
    write_json(out/"run_failures.json", failures)
    complete = export(out, cfg, cells)
    manifest["status"] = "complete" if complete and not failures else "incomplete"
    write_json(manifest_path, manifest)
    print(f"{manifest['status']}: data saved to {out}", flush=True)
    if manifest["status"] != "complete":
        raise RuntimeError("Some jobs are incomplete; use --resume")


if __name__ == "__main__":
    main()
