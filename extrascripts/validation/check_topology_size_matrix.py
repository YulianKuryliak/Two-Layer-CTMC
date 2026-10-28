"""Targeted checks, including a tiny end-to-end collection and interrupted resume.

Run: python extrascripts/validation/check_topology_size_matrix.py
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from extrascripts.runners import run_topology_size_matrix as runner
import numpy as np


class MatrixTests(unittest.TestCase):
    def test_censoring_and_event_horizon(self):
        cfg = runner.load(runner.DEFAULT_CONFIG)
        cfg["communities"] = 3
        b = runner.benchmark(cfg, dict(id="synthetic", n=8, d=4, k=2), .01, 1000, .003)
        events = runner.infection_events(dict(transmission_events=[
            dict(time=0, dst_community=0, dst_node=0),
            dict(time=1001, dst_community=1, dst_node=8)]), True, 1000)
        obs = runner.observables(events, b)
        self.assertEqual(obs["arrivals"], [0, None, None])
        self.assertIsNone(obs["T_last"])
        self.assertIsNone(obs["T_99"])
        self.assertFalse(obs["completion"])

    def test_config_and_seeds(self):
        cfg = runner.load(runner.DEFAULT_CONFIG)
        runner.validate_config(cfg)
        cells = runner.cells_for(cfg)
        self.assertEqual(len(cells), 64)
        self.assertEqual(sum(len(runner.model_schedule(cfg, c, r)) for c in cells for r in range(cfg["runs"])), 25600)
        cfg["runs"] = 150
        self.assertTrue(all(len(runner.model_schedule(cfg, c, 149)) == 4 for c in cells))
        self.assertNotEqual(runner.seeds(cfg, cells[0], "main", 0), runner.seeds(cfg, cells[0], "calibration", 0))
        cfg["degrees"] = [1001]
        with self.assertRaises(ValueError):
            runner.validate_config(cfg)

    def test_collect_resume_and_source_guard(self):
        with tempfile.TemporaryDirectory(prefix="ctmc_matrix_test_") as folder:
            root = Path(folder)
            cfg = runner.load(runner.DEFAULT_CONFIG)
            cfg.update(communities=3, sizes=[8], degrees=[4], inter_links=[2], runs=1,
                       calibration_runs=1, target_time=500, T_end=2000, workers=1,
                       rho=.02, refinement_rho=[.01], refinement_runs=1,
                       refinement_sizes=[8], refinement_degrees=[4], refinement_links=[2], report_points=33)
            config = root/"config.json"
            config.write_text(json.dumps(cfg))
            out = root/"data"
            cmd = [sys.executable, str(Path(runner.__file__).resolve()), "--config", str(config), "--output", str(out)]
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            manifest = runner.load(out/"benchmark_manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(runner.load(out/"completion.json")[0]["saved_simulations"], 3)
            directory = out/"checkpoints/L3_n8_d4_k2/rep_00000"
            original = runner.load(directory/"mm_rho0.02.json")
            calibration = out/"calibration/L3_n8_d4_k2/calibration.json"
            cal = runner.load(calibration)
            self.assertAlmostEqual(cal["median_reference_u"]/cal["beta"], 500)
            stamp = calibration.stat().st_mtime_ns
            for label in ("full", "mm_rho0.02", "mm_rho0.01"):
                with np.load(directory/f"{label}.npz") as z:
                    np.testing.assert_array_equal(z["S"]+z["I"]+z["R"], np.full((3,33),8))
                    np.testing.assert_allclose(z["time_u"], cal["beta"]*z["time"])
                    self.assertEqual(z["I"][:,0].sum(), 1)
                    self.assertTrue(np.all(np.diff(z["I"], axis=1) >= 0))
                    self.assertLessEqual(z["infection_time"].max(), 2000)
            # Delete only our temporary generated partner checkpoint to emulate interruption.
            (directory/"mm_rho0.02.json").unlink()
            full_stamp = (directory/"full.json").stat().st_mtime_ns
            result = subprocess.run(cmd+["--resume"], text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(calibration.stat().st_mtime_ns, stamp)
            self.assertEqual((directory/"full.json").stat().st_mtime_ns, full_stamp)
            resumed = runner.load(directory/"mm_rho0.02.json")
            for key in ("arrivals", "T_last", "T_99", "network_seed", "event_seed", "community_rng_seeds"):
                self.assertEqual(original[key], resumed[key])
            result = subprocess.run(cmd+["--resume", "--runs", "2", "--refinement-runs", "2"],
                                    text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(calibration.stat().st_mtime_ns, stamp)
            self.assertEqual((directory/"full.json").stat().st_mtime_ns, full_stamp)
            self.assertEqual(runner.load(out/"completion.json")[0]["saved_simulations"], 6)
            self.assertEqual(runner.load(out/"benchmark_manifest.json")["replication_targets"], dict(runs=2, refinement_runs=2))
            with (out/"completion_by_model.csv").open() as handle:
                counts = list(csv.DictReader(handle))
            self.assertEqual(len(counts), 3)
            self.assertTrue(all(v["saved"] == "2" and v["collection_complete"] == "True" for v in counts))
            # A changed scientific setting must not reuse the old data.
            cfg["rho"] = .03
            config.write_text(json.dumps(cfg))
            result = subprocess.run(cmd+["--resume"], text=True, capture_output=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
