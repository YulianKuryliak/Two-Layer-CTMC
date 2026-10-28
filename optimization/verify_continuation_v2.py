"""Non-simulation filesystem/provenance checks for the continuation adapter."""
import importlib.util
import json
import pickle
from pathlib import Path
import tempfile
import types
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


class ContinuationChecks(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("continuation_check", ROOT / "optimization/continue_chain_v2.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.runner = self.module.runner
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "destination"
        self.source.mkdir()
        self.output.mkdir()
        (self.source / "manifest.json").write_text('{}')
        self.task = {"id": "case/task", "input_sha256": "input", "full_extension": None,
                     "source_checkpoint": {"source": "full.json.gz", "sha256": "fullsha"}}
        self.original_valid = types.FunctionType(self.runner.valid_existing.__code__,
                                                 dict(self.runner.valid_existing.__globals__, VERSION="old"))
        self.source_manifest = {"version": "old", "code_fingerprint": "old-code"}
        self.module.verified_source = lambda source: (self.source_manifest, [self.task], self.original_valid)
        self.runner.prepare = lambda output: ({"code_fingerprint": "new-code", "tasks_sha256": "tasks"}, [self.task])
        self.runner.log = lambda *args, **kwargs: None
        full = {"simulator": self.runner.FULL, "value": 7}
        self.payload = {"version": "old", "status": "complete", "input_sha256": "input",
                        "code_fingerprint": "old-code", "runs": [full, {"simulator": self.runner.MM,
                        "typed_events": [], "change_point_trajectories": [], "termination": {}, "repeat_imports": 0}],
                        "full_baseline_provenance": {"reused": True, "source": "full.json.gz",
                        "sha256": "fullsha", "row_sha256": self.runner.object_hash(full)}}
        self.relative = Path("checkpoints/case/task.json.gz")
        self.runner.write_checkpoint(self.source / self.relative, self.payload)

    def test_import_resume_exact_bytes_and_tampering(self):
        self.module.configure(self.source)
        source_bytes = (self.source / self.relative).read_bytes()
        self.runner.prepare(self.output)
        self.assertEqual((self.output / self.relative).read_bytes(), source_bytes)
        self.assertTrue(self.runner.valid_existing(self.output, self.task, "new-code"))
        self.assertEqual(self.runner.read_checkpoint(self.output / self.relative)["code_fingerprint"], "old-code")
        self.runner.prepare(self.output)
        self.assertEqual(len(json.loads((self.output / "continuation.json").read_text())["imported"]), 1)
        # Replace destination inode; never mutate even the temporary source link.
        replacement = self.output / "replacement"
        replacement.write_bytes(b'corrupt')
        replacement.replace(self.output / self.relative)
        self.assertEqual((self.source / self.relative).read_bytes(), source_bytes)
        with self.assertRaises(RuntimeError):
            self.runner.valid_existing(self.output, self.task, "new-code")

    def test_wrong_old_input_rejected(self):
        self.payload["input_sha256"] = "different"
        self.runner.write_checkpoint(self.source / self.relative, self.payload)
        self.module.configure(self.source)
        with self.assertRaises(RuntimeError):
            self.runner.prepare(self.output)

    def test_unfinished_task_uses_new_validator(self):
        self.task["id"] = "case/unfinished"
        self.module.configure(self.source)
        self.runner.prepare(self.output)
        self.assertFalse(self.runner.valid_existing(self.output, self.task, "new-code"))
        self.payload.update(version=self.runner.VERSION, code_fingerprint="new-code")
        self.runner.write_checkpoint(self.output / "checkpoints/case/unfinished.json.gz", self.payload)
        self.assertTrue(self.runner.valid_existing(self.output, self.task, "new-code"))

    def test_changed_task_manifest_rejected(self):
        self.module.configure(self.source)
        # Captured source task stays unchanged; destination prepare differs.
        self.task = dict(self.task, input_sha256="different")
        with self.assertRaises(RuntimeError):
            self.runner.prepare(self.output)

    def test_active_source_lock_rejected(self):
        lock = self.source / ".runner.lock"
        lock.write_text("original lock metadata")
        with lock.open("r") as handle:
            self.module.fcntl.flock(handle, self.module.fcntl.LOCK_EX | self.module.fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                with self.module.stopped_source(self.source):
                    self.fail("Active source lock must not be accepted")
        with self.module.stopped_source(self.source):
            self.assertEqual(lock.read_text(), "original lock metadata")

    def test_parallel_worker_callable_serializable(self):
        self.assertIs(pickle.loads(pickle.dumps(self.runner.simulate_task)), self.runner.simulate_task)


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("continuation_logging", ROOT / "optimization/continue_chain_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = ROOT / "output" / ("continuation_adapter_checks_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    output.mkdir()
    fingerprint = {str(path.relative_to(ROOT)): module.runner.digest(path) for path in
                   (Path(__file__), ROOT / "optimization/continue_chain_v2.py")}
    module.runner.log(output, "verification_started", simulation=False, command=[str(Path(__file__))],
                      code_sha256=fingerprint, parameters="temporary deterministic checkpoint fixtures",
                      seeds=None, trajectory_policy="no simulation; preserve existing checkpoint bytes", resume="disabled")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContinuationChecks))
    module.runner.log(output, "verification_completed", simulation=False, tests=result.testsRun,
                      status="complete" if result.wasSuccessful() else "failed", code_sha256=fingerprint)
    raise SystemExit(0 if result.wasSuccessful() else 1)
