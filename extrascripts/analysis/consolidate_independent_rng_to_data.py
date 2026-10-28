#!/usr/bin/env python3
"""Expose existing independent-RNG results under output/data/<scenario>.

No simulation is run and large CSV/checkpoint files are not copied.  Each
scenario directory receives a canonical manifest and relative links to the
existing checkpoint directory and source CSV files.
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "data"
SOURCES = (
    ROOT / "output" / "main_clean_benchmarks_independent_rng",
    ROOT / "output" / "n1000_two_cases_independent_rng",
    ROOT / "output" / "original_benchmarks_independent_rng",
    ROOT / "output" / "simple_topology_sweep",
)


def checkpoint_dirs(source: Path) -> list[Path]:
    for dirname in (".checkpoints", "checkpoints"):
        root = source / dirname
        if root.is_dir():
            return sorted(item for item in root.iterdir() if item.is_dir())
    return []


def first_payload(directory: Path) -> dict:
    path = next(directory.glob("rep_*.json.gz"), None)
    if path is None:
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def number_tag(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p")


def scenario_name(config: dict) -> str:
    L = int(config.get("communities", 0))
    n = int(config.get("community_size", 0))
    k = int(config.get("inter_links", 0))
    beta = number_tag(config.get("beta", "unknown"))
    micro = config.get("micro_graph_type", "unknown")
    if micro == "complete":
        topology = "chain_cliques"
        degree = "dcomplete"
    elif micro == "random_connected":
        topology = "chain_random_connected"
        degree_value = config.get("mean_internal_degree")
        if degree_value is None:
            degree_value = config.get("edge_prob", "unknown")
        degree = f"d{number_tag(degree_value)}"
    elif micro == "random":
        topology = "chain_random"
        degree = f"p{number_tag(config.get('edge_prob', 'unknown'))}"
    elif micro == "clique_leaves":
        topology = "clique_leaf"
        degree = "dcomplete"
    else:
        topology = str(config.get("family", "unknown"))
        degree = f"d{micro}"
    return f"{topology}_L{L}_n{n}_{degree}_k{k}_beta{beta}_independent_rng"


def relative_link(target: Path, link_path: Path) -> Path:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return link_path
        # The same parameter identity can have multiple historical datasets
        # (for example different horizons). Keep both under one canonical
        # scenario folder and distinguish only the source link.
        suffix = target.parent.parent.name.replace(".", "_")
        link_path = link_path.with_name(f"{link_path.name}__{suffix}")
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == target.resolve():
                return link_path
            raise RuntimeError(f"conflicting link already exists: {link_path}")
    link_path.symlink_to(os.path.relpath(target, link_path.parent), target_is_directory=target.is_dir())
    return link_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {
        "format": "independent_rng_data_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "naming": "topology_L_n_d_k_beta_independent_rng; T_end and replication count are manifest fields",
        "algorithm_scope": "independent-RNG runners only; shared-RNG outputs excluded",
        "scenarios": [],
    }
    for source in SOURCES:
        for checkpoint_dir in checkpoint_dirs(source):
            payload = first_payload(checkpoint_dir)
            config = payload.get("benchmark_config", {})
            if not config:
                continue
            name = scenario_name(config)
            destination = OUT / name
            destination.mkdir(parents=True, exist_ok=True)
            checkpoint_link = relative_link(checkpoint_dir, destination / "checkpoints")
            source_manifest = next(
                (source / filename for filename in ("benchmark_manifest.json", "manifest.json", "runner_config.json")
                 if (source / filename).exists()),
                None,
            )
            variant = {
                "source_dataset": str(source.relative_to(ROOT)),
                "source_checkpoints": str(checkpoint_dir.relative_to(ROOT)),
                "checkpoint_link": str(checkpoint_link.relative_to(destination)),
                "checkpoint_count": sum(1 for _ in checkpoint_dir.glob("rep_*.json.gz")),
                "benchmark_config": config,
                "source_manifest": str(source_manifest.relative_to(ROOT)) if source_manifest else None,
            }
            manifest_path = destination / "manifest.json"
            existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
            if existing is not None:
                existing.setdefault("variants", []).append(variant)
                manifest = existing
            else:
                manifest = {
                "format": "independent_rng_scenario_v1",
                "scenario": name,
                "algorithm": "Production MicroMacro with independent RNG per MicroModel",
                "benchmark_config": config,
                "variants": [variant],
                "note": "T_end, dt_out, tau, seed policy and replication count are retained here; they are intentionally not in the folder name.",
                }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            # Keep the source aggregate tables discoverable without copying them.
            for filename in ("runs.csv", "community_arrivals.csv", "trajectories.csv", "runtime.csv"):
                source_csv = source / filename
                if source_csv.exists():
                    relative_link(source_csv, destination / filename)
            index["scenarios"].append({
                "scenario": name,
                "path": str(destination.relative_to(ROOT)),
                "source_dataset": str(source.relative_to(ROOT)),
                "checkpoint_count": variant["checkpoint_count"],
            })
    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Independent-RNG simulation data\n\n"
        "Each subdirectory is one parameter configuration. Folder names contain "
        "topology, L, community size, degree/density, k, beta and RNG variant. "
        "T_end and replication counts are stored in each manifest, not in names.\n\n"
        "Files are linked to existing outputs to avoid copying large trajectories. "
        "Only independent-RNG datasets are included.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(index['scenarios'])} scenario folders to {OUT}")


if __name__ == "__main__":
    main()
