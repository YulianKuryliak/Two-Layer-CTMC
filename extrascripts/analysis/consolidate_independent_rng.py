#!/usr/bin/env python3
"""Create one index/folder for all independent-RNG experiment outputs.

The existing data are not copied.  The unified directory contains symlinks to
the original datasets plus an index describing their manifests and checkpoint
directories.  This makes reuse discoverable without duplicating large files.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "output" / "independent_rng_unified"
LEGACY_SOURCES = (
    "main_clean_benchmarks_independent_rng",
    "n1000_two_cases_independent_rng",
    "original_benchmarks_independent_rng",
    "simple_topology_sweep",
    "topology_size_matrix",
)


def link(target: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        raise RuntimeError(f"refusing to overwrite existing path: {path}")
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def scenario_dirs(source: Path) -> list[Path]:
    for name in (".checkpoints", "checkpoints"):
        directory = source / name
        if directory.is_dir():
            return sorted(item for item in directory.iterdir() if item.is_dir())
    return []


def main() -> None:
    UNIFIED.mkdir(parents=True, exist_ok=True)
    entries = []
    source_specs = [(name, ROOT / "output" / name) for name in LEGACY_SOURCES]
    data_root = UNIFIED / "data"
    if data_root.is_dir():
        source_specs.extend((f"data/{item.name}", item) for item in sorted(data_root.iterdir()) if item.is_dir())
    for source_name, source in source_specs:
        if not source.exists():
            continue
        source_link = UNIFIED / "sources" / source_name.replace("/", "__")
        link(source, source_link)
        manifest = next(
            (source / name for name in ("benchmark_manifest.json", "manifest.json", "runner_config.json")
             if (source / name).exists()),
            None,
        )
        scenarios = []
        for directory in scenario_dirs(source):
            checkpoint_count = sum(1 for _ in directory.glob("rep_*.json.gz"))
            alias = UNIFIED / "scenarios" / f"{source_name}__{directory.name}"
            link(directory, alias)
            scenarios.append({
                "name": directory.name,
                "alias": str(alias.relative_to(UNIFIED)),
                "source_path": str(directory.relative_to(ROOT)),
                "checkpoint_count": checkpoint_count,
            })
        entries.append({
            "source": source_name,
            "source_path": str(source.relative_to(ROOT)),
            "source_link": str(source_link.relative_to(UNIFIED)),
            "manifest": str(manifest.relative_to(ROOT)) if manifest else None,
            "scenarios": scenarios,
        })

    index = {
        "format": "independent_rng_unified_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": "Only independent-RNG runners are indexed; shared-RNG datasets are excluded.",
        "reuse_rule": "Reuse only when the scenario manifest matches topology, beta, k, d, n, horizon, tau, reporting grid, seeds and simulator variant.",
        "entries": entries,
    }
    (UNIFIED / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    readme = UNIFIED / "README.md"
    readme.write_text(
        "# Unified independent-RNG data\n\n"
        "This directory indexes existing independent-RNG outputs without copying them. "
        "Use `index.json` to locate source manifests and scenario aliases.\n\n"
        "New runner outputs are written under `data/<experiment_name>/`; the canonical "
        "scenario name includes topology, L, n, d, k, beta, horizon and other fixed "
        "parameters in the runner manifest.\n\n"
        "The links under `scenarios/` point to the original checkpoint directories. "
        "Do not combine scenarios solely by a similar name: compare their manifests, "
        "especially beta, T_end, tau_micro, network generator and seed policy.\n",
        encoding="utf-8",
    )
    print(f"Unified index: {UNIFIED}")
    print(f"Datasets indexed: {len(entries)}")
    print(f"Scenario aliases: {sum(len(entry['scenarios']) for entry in entries)}")


if __name__ == "__main__":
    main()
