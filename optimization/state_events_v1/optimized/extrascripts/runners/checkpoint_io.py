"""Lossless checkpoint IO shared by data runners and plotting readers.

Legacy checkpoints contain a list of trajectory row dictionaries. New custom
grid checkpoints keep metadata once and numeric columns per simulator instead.
The gzip JSON envelope/path is unchanged; floats retain Python's exact JSON
round trip. No resampling, rounding, or aggregation is done here.
"""
from __future__ import annotations

from array import array
import gzip
import io
import json
import os
from pathlib import Path
import tempfile

TRAJECTORY_FORMAT = "numeric_columns_v1"
TRAJECTORY_FIELDS = ("community", "time", "S", "I", "R")


def trajectory_block(simulator: str, rows) -> dict:
    """Consume (community, time, S, I, R) rows without allocating row dicts."""
    columns = {key: array("d" if key == "time" else "q") for key in TRAJECTORY_FIELDS}
    for row in rows:
        for key, value in zip(TRAJECTORY_FIELDS, row, strict=True):
            columns[key].append(float(value) if key == "time" else int(value))
    return {"simulator": simulator, "columns": columns}


def compact_trajectories(metadata: dict, blocks: list[dict]) -> dict:
    return {"format": TRAJECTORY_FORMAT, "metadata": metadata, "blocks": blocks}


def is_compact(trajectories) -> bool:
    return isinstance(trajectories, dict) and trajectories.get("format") == TRAJECTORY_FORMAT


def has_trajectories(payload: dict) -> bool:
    trajectories = payload.get("trajectories", [])
    if is_compact(trajectories):
        return any(len(block["columns"]["time"]) for block in trajectories["blocks"])
    return bool(trajectories)


def iter_trajectory_rows(payload: dict):
    """Reconstruct legacy rows, in their original order, for optional export."""
    trajectories = payload.get("trajectories", [])
    if not isinstance(trajectories, dict):
        yield from trajectories
        return
    if not is_compact(trajectories):
        raise ValueError(f"Unsupported trajectory format: {trajectories.get('format')}")
    for block in trajectories["blocks"]:
        columns = block["columns"]
        for values in zip(*(columns[key] for key in TRAJECTORY_FIELDS), strict=True):
            yield {**trajectories["metadata"], "simulator": block["simulator"],
                   **dict(zip(TRAJECTORY_FIELDS, values, strict=True))}


def model_trajectory_columns(payload: dict, simulator: str) -> dict:
    """Return only numeric columns; never expand compact trajectories to rows."""
    trajectories = payload.get("trajectories", [])
    if isinstance(trajectories, dict):
        if not is_compact(trajectories):
            raise ValueError(f"Unsupported trajectory format: {trajectories.get('format')}")
        blocks = [block for block in trajectories["blocks"] if block["simulator"] == simulator]
        if len(blocks) == 1:
            return blocks[0]["columns"]
        return {key: [value for block in blocks for value in block["columns"][key]]
                for key in TRAJECTORY_FIELDS}
    selected = [row for row in trajectories if row["simulator"] == simulator]
    return {key: [row[key] for row in selected] for key in TRAJECTORY_FIELDS}


def read_checkpoint(path: Path) -> dict:
    """Read either format without automatically expanding compact columns."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _json_array(value):
    if isinstance(value, array):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_checkpoint(path: Path, payload: dict) -> None:
    """Write a private temporary file, then publish one complete checkpoint.

    On worker failure only an ignored temporary file can remain. fsync completes
    before rename; the parent acknowledges completion only after this returns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                    json.dump(payload, handle, allow_nan=True, separators=(",", ":"), default=_json_array)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
