#!/usr/bin/env python3
"""Continue verified original chain results with v2; user launches simulations.

Old checkpoints retain their original bytes/version/fingerprint. A separate
import index records their provenance; only unfinished tasks use optimized code.
"""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "optimization/graph_build_v2/optimized"
spec = importlib.util.spec_from_file_location(
    "continued_chain_runner", COPY / "extrascripts/runners/rerun_chain_twoscale.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
DEFAULT_SOURCE = ROOT / "output/chain_twoscale_residual_midpoint_20261007"
DEFAULT_OUTPUT = ROOT / "output/chain_twoscale_continued_v2_20261008"


@contextmanager
def stopped_source(source):
    # Do not rewrite the source runner's lock file or any scientific results.
    with (source / ".runner.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Old runner is still active. Stop it and wait for its workers first.") from error
        yield


def verified_source(source):
    manifest = json.loads((source / "manifest.json").read_text())
    expected = json.loads((ROOT / "optimization/state_events_v1/snapshot_manifest.json").read_text())["files"]
    if manifest["version"] != "chain_twoscale_residual_midpoint_v1":
        raise RuntimeError("Only the verified original residual-midpoint version is supported")
    if manifest["environment"] != runner.environment():
        raise RuntimeError("Source Python/packages differ from the verified continuation environment")
    if runner.object_hash(manifest["code_sha256"]) != manifest["code_fingerprint"]:
        raise RuntimeError("Source code fingerprint is inconsistent")
    for relative, sha in manifest["code_sha256"].items():
        if expected.get(relative) != sha or runner.digest(source / "sources" / relative) != sha:
            raise RuntimeError(f"Source is not the validated original code: {relative}")
    report = ROOT / "output/graph_build_validation_20261008T060754957150"
    validation = json.loads((report / "summary.json").read_text())
    if validation["status"] != "complete" or not all(row["exact"] for row in validation["replays"]):
        raise RuntimeError("Required exactness validation is incomplete")
    for relative, sha in json.loads((report / "manifest.json").read_text())["code_sha256"].items():
        if runner.digest(ROOT / relative) != sha:
            raise RuntimeError(f"Validated optimization source changed: {relative}")
    tasks = runner.read_checkpoint(source / "tasks.json.gz")["tasks"]
    if runner.object_hash(tasks) != manifest["tasks_sha256"]:
        raise RuntimeError("Source task manifest changed")
    # Apply the existing rigorous checkpoint validator with original provenance.
    namespace = dict(runner.valid_existing.__globals__, VERSION=manifest["version"])
    validate = types.FunctionType(runner.valid_existing.__code__, namespace)
    return manifest, tasks, validate


def configure(source):
    source_manifest, source_tasks, old_valid = verified_source(source)
    prepare = runner.prepare
    valid = runner.valid_existing
    hashes = runner.code_hashes
    index = {}

    def code_hashes():
        return {**hashes(), str(Path(__file__).relative_to(ROOT)): runner.digest(Path(__file__))}

    runner.code_hashes = code_hashes

    def prepare_and_import(output):
        manifest, tasks = prepare(output)
        if tasks != source_tasks:
            raise RuntimeError("Source and destination task inputs/seeds/horizons differ")
        path = output / "continuation.json"
        identity = {"source": str(source), "source_manifest_sha256": runner.digest(source / "manifest.json"),
                    "source_code_fingerprint": source_manifest["code_fingerprint"],
                    "destination_code_fingerprint": manifest["code_fingerprint"],
                    "tasks_sha256": manifest["tasks_sha256"],
                    "policy": "immutable original checkpoints; mixed explicit code provenance; unfinished tasks restarted"}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise RuntimeError("Continuation provenance changed; use a new output directory")
            index.update(saved["imported"])
        runner.log(output, "continuation_import_started", simulation=False, **identity)
        try:
            for task in tasks:
                task_id = task["id"]
                relative = Path("checkpoints") / (task_id + ".json.gz")
                origin, destination = source / relative, output / relative
                if task_id in index:
                    if not destination.exists() or runner.digest(destination) != index[task_id]["sha256"]:
                        raise RuntimeError(f"Imported checkpoint changed: {task_id}")
                    continue
                if not origin.exists():
                    continue
                if not old_valid(source, task, source_manifest["code_fingerprint"]):
                    raise RuntimeError(f"Invalid old checkpoint: {task_id}")
                sha = runner.digest(origin)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    # Handles interruption after publication but before index save.
                    if runner.digest(destination) != sha:
                        if valid(output, task, manifest["code_fingerprint"]):
                            continue
                        raise RuntimeError(f"Conflicting checkpoint: {task_id}")
                    storage = "existing_identical"
                else:
                    try:
                        os.link(origin, destination)
                        storage = "hardlink_immutable"
                    except OSError:
                        temporary = destination.with_name(destination.name + ".import.tmp")
                        shutil.copyfile(origin, temporary)
                        os.replace(temporary, destination)
                        storage = "copy"
                    if runner.digest(destination) != sha:
                        raise RuntimeError(f"Checkpoint import mismatch: {task_id}")
                index[task_id] = {"checkpoint": str(relative), "sha256": sha, "storage": storage,
                                  "source_code_fingerprint": source_manifest["code_fingerprint"]}
                if len(index) % 100 == 0:
                    runner.atomic_json(path, {"identity": identity, "imported": index})
                    print(f"Verified/imported {len(index)} old trajectories", flush=True)
            runner.atomic_json(path, {"identity": identity, "imported": index})
            runner.log(output, "continuation_import_completed", simulation=False,
                       imported_tasks=len(index), index_sha256=runner.digest(path), **identity)
            print(f"Continuation ready: {len(index)} original checkpoints retained; optimized tasks in {output}", flush=True)
            return manifest, tasks
        except BaseException as error:
            runner.atomic_json(path, {"identity": identity, "imported": index})
            runner.log(output, "continuation_import_failed", simulation=False, error=repr(error), **identity)
            raise

    def valid_with_imports(output, task, fingerprint):
        if task["id"] not in index:
            return valid(output, task, fingerprint)
        record = index[task["id"]]
        if runner.digest(output / record["checkpoint"]) != record["sha256"]:
            raise RuntimeError(f"Imported checkpoint changed: {task['id']}")
        return old_valid(output, task, source_manifest["code_fingerprint"])

    runner.prepare = prepare_and_import
    runner.valid_existing = valid_with_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--resume-from", type=Path, default=DEFAULT_SOURCE)
    known, remainder = parser.parse_known_args()
    source = known.resume_from.resolve()
    # Runner keeps its existing --run / --prepare-only / workers CLI.
    if "--output" not in remainder and not any(arg.startswith("--output=") for arg in remainder):
        remainder += ["--output", str(DEFAULT_OUTPUT)]
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path)
    output, _ = output_parser.parse_known_args(remainder)
    if output.output.resolve() == source:
        raise RuntimeError("Continuation requires a new output directory")
    sys.argv = [sys.argv[0]] + remainder
    with stopped_source(source):
        configure(source)
        runner.main()


if __name__ == "__main__":
    main()
