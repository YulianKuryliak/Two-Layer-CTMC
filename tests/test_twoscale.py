"""Short deterministic-seed checks for the basic TwoScale public API."""
import copy
import random
import unittest

import networkx as nx
import numpy as np

from two_layer_ctmc import build_two_layer_network, simulate_micromacro
from two_layer_ctmc.engines.macro_engine import MacroEngine
from two_layer_ctmc.orchestrators.micromacro import MicroMacroOrchestrator


class TwoScaleTests(unittest.TestCase):
    def setUp(self):
        self.network = build_two_layer_network(
            n_communities=2, community_size=3, inter_links=2, seed=7,
            macro_graph_type="chain",
        )

    def run_model(self, **changes):
        kwargs = dict(network=self.network, beta_micro=0.4, beta_macro=0.7,
                      gamma=0.2, model="SIR", tau_micro=0.25,
                      T_end=3.0, initial_node=0, seed=11)
        kwargs.update(changes)
        return simulate_micromacro(**kwargs)

    def test_seeded_sir_and_sis_repeatability(self):
        for model in ("SIR", "SIS"):
            with self.subTest(model=model):
                self.assertEqual(self.run_model(model=model), self.run_model(model=model))

    def test_population_counts_and_time_bounds(self):
        for model in ("SIR", "SIS"):
            times, infected, logs, events = self.run_model(model=model)
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 3.0)
            self.assertEqual(times, sorted(times))
            self.assertEqual(infected[0], 1)
            for community in logs.values():
                self.assertEqual(community["times"], times)
                for S, I, R in zip(community["S"], community["I"], community["R"]):
                    self.assertEqual(S + I + R, 3)
                    self.assertGreaterEqual(min(S, I, R), 0)
                    if model == "SIS":
                        self.assertEqual(R, 0)
            self.assertEqual([event["time"] for event in events], sorted(event["time"] for event in events))
            self.assertTrue(all(0 <= event["time"] <= 3 for event in events))
            self.assertTrue(all(event["wait_time"] >= 0 for event in events))

    def test_input_network_is_not_mutated(self):
        original = copy.deepcopy(self.network)
        self.run_model()
        self.assertTrue(nx.utils.graphs_equal(original.full_graph, self.network.full_graph))
        for before, after in zip(original.micro_graphs, self.network.micro_graphs):
            self.assertTrue(nx.utils.graphs_equal(before, after))
        np.testing.assert_array_equal(original.inter_matrix, self.network.inter_matrix)

    def test_global_random_generators_are_unchanged(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        self.run_model(seed=123)
        self.assertEqual(python_state, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])

    def test_zero_rates_hold_state_to_horizon(self):
        times, infected, logs, events = self.run_model(beta_micro=0, beta_macro=0, gamma=0)
        self.assertEqual(times[-1], 3)
        self.assertEqual(set(infected), {1})
        self.assertEqual(len(events), 1)
        self.assertEqual(logs[0]["I"], [1] * len(times))
        self.assertEqual(logs[1]["I"], [0] * len(times))

    def test_invalid_time_steps_and_rates_are_rejected(self):
        for name in ("tau_micro", "T_end", "macro_T"):
            for invalid in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.run_model(**{name: invalid})
        for name in ("beta_micro", "beta_macro", "gamma"):
            for invalid in (-1, float("nan"), float("inf")):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.run_model(**{name: invalid})

    def test_invalid_initial_node_and_community_are_rejected(self):
        for changes in ({"initial_community": -1}, {"initial_community": 2},
                        {"initial_node": 3}, {"initial_node": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_model(**changes)

    def test_macro_hazard_formula(self):
        engine = MacroEngine(np.array([[0, 2], [2, 0]]), 0.7,
                             community_sizes=[3, 3], rng=random.Random(7))
        engine.update_hazards([1, 0], [2, 3])
        self.assertAlmostEqual(engine.hazards[0, 1], 0.7 * 2 / 3)
        self.assertEqual(engine.hazards[1, 0], 0)
        self.assertEqual(engine.sample_transfer(), (0, 1))
        engine.update_hazards([0, 0], [3, 3])
        self.assertEqual(engine.sample_transfer(), (-1, -1))

    def test_invalid_matrix_and_overlapping_communities_are_rejected(self):
        for W in (np.zeros((1, 1)), np.array([[0, -1], [-1, 0]]),
                  np.array([[0, 1], [0, 0]]), np.array([[1, 1], [1, 0]]),
                  np.array([[0, np.nan], [np.nan, 0]])):
            with self.subTest(W=W), self.assertRaises(ValueError):
                MicroMacroOrchestrator(W, self.network.micro_graphs, 0.4, 0.2, 0.7, 0.25, 3, rng=random.Random(7))
        graph = nx.path_graph(3)
        with self.assertRaises(ValueError):
            MicroMacroOrchestrator(np.array([[0, 1], [1, 0]]), [graph, graph],
                                0.4, 0.2, 0.7, 0.25, 3, rng=random.Random(7))


if __name__ == "__main__":
    unittest.main()
