from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd


PRODUCTION_FILES = (
    "two_layer_ctmc/engines/__init__.py",
    "two_layer_ctmc/engines/macro_engine.py",
    "two_layer_ctmc/engines/micro_engine.py",
    "two_layer_ctmc/engines/micro_model.py",
    "two_layer_ctmc/orchestrators/__init__.py",
    "two_layer_ctmc/orchestrators/micro.py",
    "two_layer_ctmc/orchestrators/micromacro.py",
    "two_layer_ctmc/simulators/__init__.py",
    "two_layer_ctmc/simulators/micro.py",
    "two_layer_ctmc/simulators/micromacro.py",
    "two_layer_ctmc/network.py",
    "two_layer_ctmc/types.py",
    "two_layer_ctmc/simulate.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def production_hashes(repo_root: Path) -> dict[str, str]:
    return {name: sha256_file(repo_root / name) for name in PRODUCTION_FILES}


def graph_checksums(graphs: Sequence[nx.Graph], full_graph: nx.Graph) -> tuple[str, str]:
    internal = []
    for community, graph in enumerate(graphs):
        internal.extend(
            (community, min(int(u), int(v)), max(int(u), int(v)), float(d.get("weight", 1.0)))
            for u, v, d in graph.edges(data=True)
        )
    internal_payload = json.dumps(sorted(internal), separators=(",", ":")).encode("utf-8")
    node_to_community = {
        int(node): community for community, graph in enumerate(graphs) for node in graph.nodes()
    }
    inter = []
    for u, v, data in full_graph.edges(data=True):
        if node_to_community[int(u)] != node_to_community[int(v)]:
            inter.append((min(int(u), int(v)), max(int(u), int(v)), float(data.get("weight", 1.0))))
    inter_payload = json.dumps(sorted(inter), separators=(",", ":")).encode("utf-8")
    return sha256_bytes(internal_payload), sha256_bytes(inter_payload)


def canonical_checksum(rows: Iterable[Mapping]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
    ).encode("utf-8")
    return sha256_bytes(payload)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def assert_finite_records(rows: Iterable[Mapping], required: Sequence[str] = ()) -> None:
    for index, row in enumerate(rows):
        for name in required:
            if name not in row or row[name] is None:
                raise ValueError(f"row {index}: required field {name!r} is null")
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"row {index}: {name} is non-finite")


def write_parquet_atomic(rows: list[dict], destination: Path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow; install requirements-experiment.txt into the project venv"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(destination)


def write_json_atomic(payload: Mapping | list, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def write_scenario_yaml(payload: Mapping, destination: Path) -> None:
    """Write the scalar-only scenario mapping as deterministic YAML."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in payload.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_metadata() -> dict:
    versions = {}
    for name in ("numpy", "networkx", "pandas", "scipy", "matplotlib", "pyarrow"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def production_diff(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *PRODUCTION_FILES],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]
