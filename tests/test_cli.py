"""CLI contracts, including two small end-to-end CSV checks."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from two_layer_ctmc.__main__ import main


class CLITests(unittest.TestCase):
    def test_help_does_not_start_a_simulation(self):
        with patch("two_layer_ctmc.__main__.simulate_micro") as simulate:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--help"])
        self.assertEqual(error.exception.code, 0)
        simulate.assert_not_called()

    def test_invalid_parameters_report_usage_errors(self):
        invalid = [
            ["--tau", "0"], ["--dt", "nan"], ["--horizon", "inf"],
            ["--beta", "-1"], ["--gamma", "nan"], ["--communities", "0"],
            ["--community-size", "-1"], ["--inter-links", "-1"],
        ]
        with patch("two_layer_ctmc.__main__.build_two_layer_network") as build:
            for args in invalid:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(args)
                    self.assertEqual(error.exception.code, 2)
        build.assert_not_called()

    def test_existing_output_is_protected_before_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.csv"
            path.write_text("keep this", encoding="utf-8")
            with patch("two_layer_ctmc.__main__.simulate_micro") as simulate:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(["--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep this")
            simulate.assert_not_called()

    def test_explicit_overwrite_and_csv_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.csv"
            path.write_text("old data", encoding="utf-8")
            with patch("two_layer_ctmc.__main__.build_two_layer_network", return_value=object()):
                with patch("two_layer_ctmc.__main__.simulate_micro",
                           return_value={"rows": [(0, 0.0, 2, 1, 0)]}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["--output", str(path), "--overwrite"]), 0)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [["community", "time", "S", "I", "R"],
                                    ["0", "0.0", "2", "1", "0"]])

    def test_model_errors_are_reported_without_a_traceback_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.csv"
            with patch("two_layer_ctmc.__main__.build_two_layer_network",
                       side_effect=ValueError("invalid graph")):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        main(["--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid graph", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse(path.exists())

    def test_small_full_and_twoscale_runs_save_valid_csv(self):
        # Fixed tiny cases: six nodes, horizon 2, seed 7; no historical inputs.
        with tempfile.TemporaryDirectory() as folder:
            for method, model in [("full", "SIR"), ("twoscale", "SIS")]:
                with self.subTest(method=method, model=model):
                    path = Path(folder) / method / "result.csv"
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = main([
                            "--method", method, "--model", model,
                            "--communities", "2", "--community-size", "3",
                            "--inter-links", "1", "--beta", "0.4", "--gamma", "0.1",
                            "--horizon", "2", "--dt", "0.5", "--tau", "0.5",
                            "--seed", "7", "--output", str(path),
                        ])
                    self.assertEqual(status, 0)
                    with path.open(newline="", encoding="utf-8") as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(reader.fieldnames, ["community", "time", "S", "I", "R"])
                        rows = list(reader)
                    self.assertTrue(rows)
                    self.assertEqual({int(row["community"]) for row in rows}, {0, 1})
                    for row in rows:
                        counts = [int(row[key]) for key in ("S", "I", "R")]
                        self.assertEqual(sum(counts), 3)
                        self.assertTrue(all(value >= 0 for value in counts))
                        self.assertTrue(0 <= float(row["time"]) <= 2)
                        if model == "SIS":
                            self.assertEqual(int(row["R"]), 0)


if __name__ == "__main__":
    unittest.main()
