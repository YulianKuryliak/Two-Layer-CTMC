#!/usr/bin/env python3
"""Extend only the two censored Full runs in the n=50, ASPL=6, k=1 chain.

Original paired checkpoints remain intact. Each extension is saved beside its
source checkpoint after verifying the complete event history through T=30000.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from extrascripts.runners import run_l10_aspl_chain_by_size_independent_rng as sizes


OLD_T_MAX = 30_000.0
NEW_T_MAX = 40_000.0
TARGETS = ((3, 0), (4, 0))


def main() -> None:
    sizes.configure_size(50)
    base = sizes.base
    scenario = base.DATA_ROOT / "scenarios" / base.scenario_name(6.0, 1)

    for network_id, replicate_id in TARGETS:
        name = f"network{network_id:03d}_replicate{replicate_id:02d}.json.gz"
        source = scenario / "checkpoints" / name
        destination = scenario / "checkpoints_full_to40000" / name
        original = read_checkpoint(source)
        task = original["task"]
        assert (task["target_ASPL"], task["k"], task["network_id"], task["replicate_id"]) == (
            6.0, 1, network_id, replicate_id
        )
        assert original["T_max"] == OLD_T_MAX
        full_old, mm_old = original["runs"]
        assert full_old["simulator"] == base.FULL and not full_old["completion"]
        assert mm_old["simulator"] == base.MICROMACRO and mm_old["completion"]
        assert full_old["arrivals"][9] is None

        seeds = base.event_seeds(6.0, network_id, replicate_id)
        assert original["event_seeds"] == seeds
        if destination.exists():
            saved = read_checkpoint(destination)
            if saved["task"] != task or saved["full_event_seed"] != seeds["full_event_seed"]:
                raise RuntimeError(f"Unexpected existing extension: {destination}")
            print(f"[reuse] {name}: T_9={saved['full_run']['T_last']:.6f}", flush=True)
            continue

        _, full_graph, _, comm_nodes = base.build_chain_network(task)
        print(f"[run] {name}: Full Micro to T={NEW_T_MAX:g}", flush=True)
        started = time.perf_counter()
        result = base.MicroSimulator(
            full_graph=full_graph,
            comm_nodes=comm_nodes,
            infection_rate=base.BETA,
            recovery_rate=base.GAMMA,
            model=base.MODEL,
        ).run(
            T_end=NEW_T_MAX,
            dt_out=NEW_T_MAX,
            initial_node=task["initial_node"],
            rng=random.Random(seeds["full_event_seed"]),
            stop_at_absorption=True,
        )
        runtime_seconds = time.perf_counter() - started
        events = base.compact_full_events(result)
        old_events = full_old["infection_events"]
        prefix = [event for event in events if event[0] <= OLD_T_MAX]
        if prefix != old_events:
            raise RuntimeError(f"Pathwise prefix differs from original checkpoint: {name}")
        if not base.validate_full_interfaces(result, full_graph):
            raise RuntimeError(f"Nonphysical Full interface event: {name}")

        old_limit = base.T_MAX
        try:
            base.T_MAX = NEW_T_MAX
            full_run = {
                "simulator": base.FULL,
                "runtime_seconds": runtime_seconds,
                "termination": result["run_metadata"],
                "physical_interface_valid": True,
                "infection_events": events,
                **base.arrival_summary(base.full_arrivals(result)),
            }
        finally:
            base.T_MAX = old_limit
        if not full_run["completion"]:
            print(f"[still censored] {name}: reached only community "
                  f"{max(i for i, t in enumerate(full_run['arrivals']) if t is not None)}", flush=True)
        else:
            print(f"[complete] {name}: T_9={full_run['T_last']:.6f}; "
                  f"prefix={len(prefix)} events verified", flush=True)

        write_checkpoint(destination, {
            "source_checkpoint": str(source.relative_to(base.ROOT)),
            "task": task,
            "original_T_max": OLD_T_MAX,
            "T_max": NEW_T_MAX,
            "full_event_seed": seeds["full_event_seed"],
            "initial_node": task["initial_node"],
            "pathwise_prefix_verified": True,
            "original_event_count": len(old_events),
            "full_run": full_run,
        })
        print(f"[saved] {destination}", flush=True)


if __name__ == "__main__":
    main()
