"""Run small pathwise regressions; no benchmark datasets are generated.

Usage: python extrascripts/validation/check_result_preserving_optimizations.py
"""
import concurrent.futures
import math
import random
import sys
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extrascripts.runners import run_original_benchmarks_independent_rng as runner
from two_layer_ctmc.engines.micro_engine import MicroEngine
from two_layer_ctmc.engines.micro_model import MicroModel
from two_layer_ctmc.orchestrators.micro import MicroOrchestrator


class RecountingOrchestrator(MicroOrchestrator):
    """Reference: recount after an event, preserving pre-event sample-and-hold."""

    @staticmethod
    def _counts_SIR_per_community(sim, comm_nodes):
        def recount():
            return MicroOrchestrator._counts_SIR_per_community(sim, comm_nodes)

        class Counts(list):
            def __setitem__(self, index, value):
                # Ignore incremental arithmetic and independently scan nodes.
                super().__setitem__(slice(None), recount())

        return Counts(recount())


def square_task(value):
    """Picklable, non-simulation task for the process-pool check."""
    return value * value


class CountTests(unittest.TestCase):
    def test_exact_full_results_and_rng_against_recount(self):
        for topology in (nx.path_graph, nx.complete_graph):
            graph = topology(12)
            for u, v in graph.edges():
                graph[u][v]["weight"] = 0.5 + (u + v) % 3
            for model, gamma in ((2, 0.0), (2, 0.3), (1, 0.3)):
                for seed in (1, 7, 19):
                    for communities in (
                        [list(range(6)), list(range(6, 12))],
                        [[0, 0, 1, 2], [2, 3, 4], []],
                    ):
                        with self.subTest(topology=topology.__name__, model=model, gamma=gamma,
                                          seed=seed, communities=communities):
                            args = (graph, communities, 0.3, gamma, model)
                            actual = MicroOrchestrator(*args, rng=random.Random(seed))
                            reference = RecountingOrchestrator(*args, rng=random.Random(seed))
                            self.assertEqual(actual.run(12.0, 0.25, 0), reference.run(12.0, 0.25, 0))
                            self.assertEqual(actual.rng.getstate(), reference.rng.getstate())
                            self.assertEqual(list(actual.sim.G.nodes(data=True)),
                                             list(reference.sim.G.nodes(data=True)))

    def test_one_initial_recount_only(self):
        sim = MicroOrchestrator(nx.complete_graph(8), [list(range(8))], 1.0, 0.0,
                                rng=random.Random(3))
        with patch.object(sim, "_counts_SIR_per_community", wraps=sim._counts_SIR_per_community) as recount:
            result = sim.run(10.0, 0.5, 0)
        self.assertEqual(recount.call_count, 1)
        self.assertEqual(result["rows"][-1][2:], (0, 8, 0))

    def test_noop_infection_does_not_change_counts(self):
        sim = MicroOrchestrator(nx.path_graph(2), [[0, 1]], 0.0, 0.0, rng=random.Random(3))
        with patch.object(sim.sim, "simulate_step", side_effect=[
            (0.5, ("infection", 0, 0)), (math.inf, None)
        ]):
            result = sim.run(1.0, 0.25, 0)
        self.assertTrue(all(row[2:] == (1, 1, 0) for row in result["rows"]))


class CacheTests(unittest.TestCase):
    def test_skip_empty_cache_but_invalidate_pending_event(self):
        sim = MicroModel(0.2, 0.0, rng=random.Random(5))
        for node in range(3):
            sim.add_node(node)
        state = sim.rng.getstate()
        with patch.object(sim, "_invalidate_cache", wraps=sim._invalidate_cache) as invalidate:
            sim.add_edge(0, 1, 1.0)
            invalidate.assert_not_called()
            self.assertEqual(sim.rng.getstate(), state)
            for wait in (0.5, math.inf):
                sim._cached_wait = wait
                sim._cached_event = ("infection", 0, 1)
                sim._cached_rng_before = state
                sim._cached_rng_after = state
                sim.add_edge(1, 2, 1.0)
                self.assertIsNone(sim._cached_wait)
                self.assertIsNone(sim._cached_event)
                self.assertIsNone(sim._cached_rng_before)
                self.assertIsNone(sim._cached_rng_after)
            self.assertEqual(invalidate.call_count, 2)

    def test_shared_rng_regression_gate(self):
        passed, reason = runner._regression_gate()
        self.assertTrue(passed, reason)


class TopologyStorageTests(unittest.TestCase):
    def test_node_only_storage_preserves_neighbors_and_validation(self):
        compact = MicroModel(0.2, 0.0, rng=random.Random(5))
        reference = MicroModel(0.2, 0.0, rng=random.Random(5))
        for node in range(4):
            compact.add_node(node)
            reference.add_node(node)
        # Include self-loops, repeated edges and zero weights: preserve even
        # the legacy neighbor-cache behavior rather than silently deduplicate.
        for u, v, weight in ((0, 2, 0.5), (0, 1, 1.0), (0, 2, 2.0), (1, 1, 0.0)):
            compact.add_edge(u, v, weight)
            MicroEngine.add_edge(reference, u, v, weight)
        self.assertEqual(compact.G.number_of_edges(), 0)
        self.assertGreater(reference.G.number_of_edges(), 0)
        self.assertEqual(compact._neighbors, reference._neighbors)
        self.assertEqual(list(compact.G.nodes(data=True)), list(reference.G.nodes(data=True)))
        self.assertEqual(compact.rng.getstate(), reference.rng.getstate())
        for invalid in (-1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                compact.add_edge(0, 3, invalid)
        self.assertEqual(compact._neighbors, reference._neighbors)
        full = MicroEngine(0.2, 0.0)
        full.add_node(0)
        full.add_node(1)
        full.add_edge(0, 1, 1.0)
        self.assertEqual(full.G.number_of_edges(), 1)

    def test_exact_micromacro_against_legacy_edge_storage(self):
        original_run = runner.Orchestrator.run
        captured = []

        def capture_run(orchestrator):
            result = original_run(orchestrator)
            captured.append([
                (model.rng.getstate(), list(model.G.nodes(data=True)),
                 model.current_time, model.total_infection_rate, model.total_recovery_rate,
                 model._cached_wait, model._cached_event,
                 model._cached_rng_before, model._cached_rng_after)
                for model in orchestrator.micro_models
            ])
            return result

        for topology in ("complete", "random_connected"):
            for model, gamma in ((2, 0.0), (2, 0.3), (1, 0.3)):
                benchmark = replace(runner.BENCHMARKS[1], communities=3, community_size=8,
                                    inter_links=2, micro_graph_type=topology, edge_prob=0.5,
                                    beta=0.3, gamma=gamma, model=model, T_end=15.0)
                for independent in (False, True):
                    for seed in (1, 7, 19):
                        with self.subTest(topology=topology, model=model, gamma=gamma,
                                          independent=independent, seed=seed):
                            graphs, full, W = runner._generate_network(benchmark, seed + 100)
                            captured.clear()
                            with patch.object(runner.Orchestrator, "run", capture_run):
                                with patch.object(MicroModel, "add_edge", MicroEngine.add_edge):
                                    expected = runner._run_experimental_micromacro(
                                        benchmark, graphs, full, W, seed, independent=independent)
                                expected_rng = random.getstate()
                                actual = runner._run_experimental_micromacro(
                                    benchmark, graphs, full, W, seed, independent=independent)
                            self.assertEqual(actual, expected)
                            self.assertEqual(random.getstate(), expected_rng)
                            self.assertEqual(captured[0], captured[1])


class PostprocessingTests(unittest.TestCase):
    @staticmethod
    def old_arrivals(events, communities, horizon):
        arrivals = [math.nan] * communities
        for event in sorted(events, key=lambda item: float(item.get("time", math.inf))):
            if str(event.get("event_type", "")).lower() not in {"seed", "infection", "transfer"}:
                continue
            community = event.get("community")
            event_time = float(event.get("time", math.inf))
            if community is None or event_time > horizon:
                continue
            community = int(community)
            if 0 <= community < communities and not math.isfinite(arrivals[community]):
                arrivals[community] = event_time
        arrivals[0] = 0.0
        return arrivals

    def test_arrivals_match_sort_including_nonfinite_and_censoring(self):
        rng = random.Random(91)
        for special in ([], [math.nan], [math.inf], [-math.inf], [math.nan, -math.inf, math.inf]):
            events = [{"time": value, "community": ci, "event_type": event_type}
                      for value in [0.0, 1.0, 1.0, 5.0, 11.0] + special
                      for ci in (None, -1, 0, 1, 2, 9)
                      for event_type in ("seed", "INFECTION", "transfer", "recovery")]
            for _ in range(5):
                rng.shuffle(events)
                for horizon in (5.0, math.inf):
                    actual = runner._arrivals_from_events(events, 4, horizon)
                    expected = self.old_arrivals(events, 4, horizon)
                    np.testing.assert_equal(actual, expected)

    def test_vectorized_sampling_matches_scalar(self):
        benchmark = replace(runner.BENCHMARKS[1], communities=2, T_end=3.0, dt_out=0.3)
        logs = {ci: {"times": [0.0, 0.3, 0.3, 0.9, 1.71],
                     "S": [9, 8, 7, 6, 5], "I": [1, 2, 3, 4, 5], "R": [0] * 5}
                for ci in range(2)}
        expected = []
        for ci in range(2):
            log = logs[ci]
            for time in np.arange(0.0, benchmark.T_end + 0.5 * benchmark.dt_out, benchmark.dt_out):
                index = max(0, int(np.searchsorted(np.asarray(log["times"]), time, side="right") - 1))
                expected.append(dict(family=benchmark.family, benchmark=benchmark.name,
                                     replication_id=3, simulator=runner.MICROMACRO, time=float(time),
                                     community=ci, S=log["S"][index], I=log["I"][index], R=log["R"][index]))
        self.assertEqual(runner._micromacro_trajectory(benchmark, 3, logs), expected)


class ExecutorTests(unittest.TestCase):
    def test_bounded_futures_and_release(self):
        refs = []
        submitted = []

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, task):
                future = concurrent.futures.Future()
                future.set_result(square_task(task))
                refs.append(weakref.ref(future))
                submitted.append(task)
                if sum(ref() is not None for ref in refs) > 2 * self.max_workers:
                    raise AssertionError("Completed Future/results are being retained")
                return future

        with patch.object(runner.concurrent.futures, "ProcessPoolExecutor", ImmediateExecutor):
            results = runner._execute(list(range(100)), 3)
            first = next(results)
            self.assertEqual(len(submitted), 6)
            all_results = [first, *results]
            self.assertEqual(sorted(all_results), [i * i for i in range(100)])
        self.assertTrue(all(ref() is None for ref in refs))

    def test_real_process_pool_matches_serial_without_simulations(self):
        with patch.object(runner, "_run_task", square_task):
            self.assertEqual(sorted(runner._execute(list(range(7)), 2)),
                             list(runner._execute(list(range(7)), 1)))
            self.assertEqual(list(runner._execute([], 2)), [])


if __name__ == "__main__":
    unittest.main()
