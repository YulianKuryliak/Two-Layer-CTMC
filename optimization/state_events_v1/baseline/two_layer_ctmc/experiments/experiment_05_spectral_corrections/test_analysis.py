import json
import unittest

import pandas as pd

from .analysis import analyze_replications, normalize_replication_table


class ArrivalObservationTests(unittest.TestCase):
    def test_explicit_observation_flags_control_completed_layer_metrics(self):
        frame = pd.DataFrame([
            {
                "scenario_id": "synthetic", "variant_id": "V0", "replication_id": 1,
                "arrival_times": json.dumps([0.0, 10.0]),
                "arrival_observed": json.dumps([True, False]),
            },
            {
                "scenario_id": "synthetic", "variant_id": "V0", "replication_id": 2,
                "arrival_times": json.dumps([0.0, 100.0]),
                # An event exactly at the horizon remains observed when the
                # simulation records it explicitly.
                "arrival_observed": json.dumps([True, True]),
            },
            {
                "scenario_id": "synthetic", "variant_id": "V1", "replication_id": 1,
                "arrival_times": json.dumps([0.0, 12.0]),
                "arrival_observed": json.dumps([True, True]),
            },
            {
                "scenario_id": "synthetic", "variant_id": "V1", "replication_id": 2,
                "arrival_times": json.dumps([0.0, 20.0]),
                "arrival_observed": json.dumps([True, False]),
            },
        ])

        long, outputs = analyze_replications(
            frame, horizon=100.0, bootstrap_reps=20, bootstrap_seed=7
        )
        terminal = long[long.layer_index == 1].sort_values(
            ["variant_id", "replication_id"]
        )
        self.assertEqual(terminal.observed.tolist(), [False, True, True, False])

        layer = outputs.layer_metrics.query(
            "scenario_id == 'synthetic' and variant_id == 'V1' and layer_index == 1"
        ).iloc[0]
        self.assertEqual(layer.n_reference_completed, 1)
        self.assertEqual(layer.n_variant_completed, 1)

    def test_time_based_censoring_remains_backward_compatible_without_flags(self):
        frame = pd.DataFrame([
            {
                "scenario_id": "synthetic", "variant_id": "V0", "replication_id": 1,
                "arrival_times": json.dumps([0.0, 99.0]),
            },
            {
                "scenario_id": "synthetic", "variant_id": "V1", "replication_id": 1,
                "arrival_times": json.dumps([0.0, 100.0]),
            },
        ])
        long = normalize_replication_table(frame, horizon=100.0)
        terminal = long[long.layer_index == 1].sort_values("variant_id")
        self.assertEqual(terminal.observed.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
