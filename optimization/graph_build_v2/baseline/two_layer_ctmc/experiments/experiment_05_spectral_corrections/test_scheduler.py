from __future__ import annotations

from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np

from .config import ModelVariant, ScenarioConfig
from .scheduler import ExperimentalMicroMacroScheduler


def _seeds() -> dict[str, int]:
    return {
        "micro_event_rng": 11,
        "macro_budget_rng": 1,
        "macro_pair_rng": 13,
        "placement_rng": 17,
    }


class HazardValidationTests(unittest.TestCase):
    def _scheduler_shell(self) -> ExperimentalMicroMacroScheduler:
        scheduler = ExperimentalMicroMacroScheduler.__new__(ExperimentalMicroMacroScheduler)
        scheduler.config = ScenarioConfig("test", L=2, n=1)
        scheduler.W = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        scheduler.sizes = np.ones(2)
        return scheduler

    def test_nonfinite_hazard_raises_instead_of_silent_zero(self):
        scheduler = self._scheduler_shell()
        with self.assertRaises(FloatingPointError):
            scheduler._hazards(np.asarray([np.nan, 0.0]), np.ones(2))

    def test_negative_hazard_raises(self):
        scheduler = self._scheduler_shell()
        scheduler.W[0, 1] = -1.0
        with self.assertRaises(ValueError):
            scheduler._hazards(np.ones(2), np.ones(2))


class DiagnosticStateTests(unittest.TestCase):
    def test_hazard_diagnostic_records_supplied_boundary_state(self):
        scheduler = ExperimentalMicroMacroScheduler.__new__(ExperimentalMicroMacroScheduler)
        scheduler.W = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        scheduler.degenerate = np.asarray([False, False])
        records: list[dict] = []
        scheduler._append_hazard_diagnostics(
            records,
            0.0,
            0.5,
            np.asarray([[0.0, 0.2], [0.3, 0.0]]),
            np.asarray([0.4, 0.5]),
            np.asarray([0.6, 0.7]),
            np.asarray([0.2, 0.3]),
            np.asarray([0.11, 0.22]),
            0.0,
            1.0,
        )
        self.assertEqual([row["i_boundary"] for row in records], [0.11, 0.22])


class MacroCounterTests(unittest.TestCase):
    def test_processed_clock_events_are_distinct_from_successful_imports(self):
        graph = nx.complete_graph(2)
        nx.set_edge_attributes(graph, 1.0, "weight")
        config = ScenarioConfig(
            "counter_test",
            L=1,
            n=2,
            beta_in=0.0,
            beta_out=100.0,
            tau_micro=1.0,
            T_end=0.1,
        )
        descriptor = SimpleNamespace(
            tau_find=0.0,
            kappa=float("inf"),
            degenerate_descriptors=False,
        )
        scheduler = ExperimentalMicroMacroScheduler(
            np.asarray([[1.0]]),
            [graph],
            config,
            ModelVariant.V1_BASELINE,
            [descriptor],
            _seeds(),
            collect_diagnostics=False,
            collect_trajectories=False,
        )
        result = scheduler.run(0, 0)
        self.assertGreater(result.macro_events, 0)
        self.assertEqual(result.macro_events, result.null_macro_events)
        self.assertEqual(result.successful_macro_imports, 0)


class SchedulerLimitTests(unittest.TestCase):
    def test_zero_lag_infinite_kappa_variants_match_baseline_pathwise(self):
        first = nx.complete_graph(range(4))
        second = nx.complete_graph(range(4, 8))
        nx.set_edge_attributes(first, 1.0, "weight")
        nx.set_edge_attributes(second, 1.0, "weight")
        config = ScenarioConfig(
            "limit_test",
            L=2,
            n=4,
            k=2,
            beta_in=0.3,
            beta_out=5.0,
            tau_micro=0.5,
            T_end=5.0,
        )
        descriptor = SimpleNamespace(
            tau_find=0.0,
            kappa=float("inf"),
            degenerate_descriptors=False,
        )
        results = {}
        for variant in (
            ModelVariant.V1_BASELINE,
            ModelVariant.V2_LAG,
            ModelVariant.V3_LATENT,
            ModelVariant.V4_COMBINED,
        ):
            scheduler = ExperimentalMicroMacroScheduler(
                np.asarray([[0.0, 2.0], [2.0, 0.0]]),
                [first, second],
                config,
                variant,
                [descriptor, descriptor],
                _seeds(),
                collect_diagnostics=False,
                collect_trajectories=True,
            )
            results[variant] = scheduler.run(0, 0)

        baseline = results[ModelVariant.V1_BASELINE]

        def event_signature(result):
            fields = (
                "event_mode", "time", "community", "event_type", "source_i",
                "dest_j", "target_node", "node", "src_node", "null_event",
            )
            return [
                tuple(row.get(field) for field in fields)
                for row in result.raw_records
                if row.get("record_type") == "event"
            ]

        for variant in (
            ModelVariant.V2_LAG,
            ModelVariant.V3_LATENT,
            ModelVariant.V4_COMBINED,
        ):
            with self.subTest(variant=variant.value):
                observed = results[variant]
                self.assertEqual(observed.arrivals, baseline.arrivals)
                self.assertEqual(observed.trajectory_records, baseline.trajectory_records)
                self.assertEqual(event_signature(observed), event_signature(baseline))
                self.assertGreater(observed.micro_events, 0)


if __name__ == "__main__":
    unittest.main()
