"""Small deterministic checks of the public network and Full CTMC contracts."""
import copy
import random
import unittest

import networkx as nx
import numpy as np

from two_layer_ctmc import TwoLayerNetwork, build_two_layer_network, simulate_micro
from two_layer_ctmc.engines.micro_engine import MicroEngine


class NetworkTests(unittest.TestCase):
    def test_seeded_topologies_and_interface_weights(self):
        for topology in ("complete", "chain", "random", "random_connected"):
            with self.subTest(topology=topology):
                kwargs = dict(n_communities=2, community_size=3, inter_links=2,
                              micro_graph_type=topology, edge_prob=1.0, seed=7)
                first, second = build_two_layer_network(**kwargs), build_two_layer_network(**kwargs)
                self.assertEqual(list(first.full_graph.edges(data=True)), list(second.full_graph.edges(data=True)))
                self.assertEqual(first.community_nodes, [[0, 1, 2], [3, 4, 5]])
                np.testing.assert_array_equal(first.inter_matrix, [[0, 2], [2, 0]])
                self.assertEqual(nx.number_of_selfloops(first.full_graph), 0)

    def test_network_generator_does_not_change_global_rng(self):
        state = random.getstate()
        build_two_layer_network(n_communities=2, community_size=3, seed=7)
        self.assertEqual(random.getstate(), state)

    def test_invalid_network_parameters_are_rejected(self):
        for kwargs in ({"community_size": 0}, {"n_communities": 1.5},
                       {"n_communities": True}, {"inter_links": -1},
                       {"inter_links": 10, "community_size": 3}, {"edge_prob": float("nan")},
                       {"edge_prob": 2}, {"micro_graph_type": "unknown"},
                       {"macro_graph_type": "unknown"}, {"seed": 1.5},
                       {"micro_graph_type": "random_connected", "edge_prob": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_two_layer_network(**kwargs)

    def test_custom_network_checks_physical_edges_and_partition(self):
        net = build_two_layer_network(n_communities=2, community_size=2, inter_links=1, seed=7)
        for bad_matrix in (np.zeros((2, 2)), [[0, -1], [-1, 0]],
                           [[0, 1], [0, 0]], [[0, float("inf")], [float("inf"), 0]],
                           np.eye(2), [[0]]):
            with self.subTest(W=bad_matrix), self.assertRaises(ValueError):
                TwoLayerNetwork(net.micro_graphs, net.full_graph, bad_matrix, net.community_nodes)
        with self.assertRaises(ValueError):
            TwoLayerNetwork(net.micro_graphs, net.full_graph, net.inter_matrix, [[0, 1], [1, 2, 3]])
        local = copy.deepcopy(net.micro_graphs)
        local[0].remove_edge(0, 1)
        with self.assertRaises(ValueError):
            TwoLayerNetwork(local, net.full_graph, net.inter_matrix, net.community_nodes)

    def test_loop_policy_and_duplicate_edge_guard(self):
        engine = MicroEngine(1.0, 1.0, model=1, cache_neighbors=True)
        engine.add_node(0)
        engine.add_node(1)
        with self.assertRaises(ValueError):
            engine.add_edge(0, 0, 1.0)
        engine.add_edge(0, 1, 1.0)
        with self.assertRaises(ValueError):
            engine.add_edge(0, 1, 1.0)
        loop = nx.Graph(); loop.add_edge(0, 0)
        with self.assertRaises(ValueError):
            MicroEngine().bind_static_graph(loop)

    def test_sis_recovery_restores_only_neighbor_transmission(self):
        engine = MicroEngine(0.5, 0.2, model=1, cache_neighbors=True)
        for node in range(3):
            engine.add_node(node)
        engine.add_edge(0, 1, 2.0)
        engine.add_edge(1, 2, 3.0)
        engine._infect_node(0)
        engine._infect_node(1)
        self.assertAlmostEqual(engine.total_infection_rate, 1.5)
        engine._recover_node(1)
        self.assertAlmostEqual(engine.total_infection_rate, 1.0)
        self.assertAlmostEqual(engine.total_recovery_rate, 0.2)
        self.assertEqual(engine.count_states(), (2, 1, 0))


class FullTests(unittest.TestCase):
    def network(self):
        return build_two_layer_network(n_communities=2, community_size=3, inter_links=1, seed=7)

    def test_repeatability_conservation_horizon_and_input_immutability(self):
        network = self.network()
        before = copy.deepcopy(network.full_graph)
        global_state = random.getstate()
        for model in ("SIR", "SIS"):
            kwargs = dict(network=network, beta=0.4, gamma=0.2, T_end=3.0,
                          dt_out=0.25, model=model, initial_node=0, seed=11)
            first, second = simulate_micro(**kwargs), simulate_micro(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(first["rows"][0][1], 0.0)
            self.assertEqual(first["rows"][-1][1], 3.0)
            for _, time, S, I, R in first["rows"]:
                self.assertTrue(0 <= time <= 3)
                self.assertEqual(S + I + R, 3)
                self.assertGreaterEqual(min(S, I, R), 0)
                if model == "SIS":
                    self.assertEqual(R, 0)
            event_times = [event["time"] for event in first["events"]]
            self.assertEqual(event_times, sorted(event_times))
            self.assertTrue(all(0 <= time <= 3 for time in event_times))
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(list(network.full_graph.nodes(data=True)), list(before.nodes(data=True)))
        self.assertEqual(list(network.full_graph.edges(data=True)), list(before.edges(data=True)))

    def test_zero_rates_hold_state_and_include_non_grid_horizon(self):
        result = simulate_micro(network=self.network(), beta=0, gamma=0,
                                T_end=2.5, dt_out=1, initial_node=0, seed=11)
        rows = [row for row in result["rows"] if row[0] == 0]
        self.assertEqual([row[1] for row in rows], [0.0, 1.0, 2.0, 2.5])
        self.assertTrue(all(row[2:] == (2, 1, 0) for row in rows))
        self.assertEqual(len(result["events"]), 1)

    def test_one_node_recovery_matches_exponential_and_model_semantics(self):
        expected = random.Random(123).expovariate(1.0)
        for model, final in (("SIR", (0, 0, 1)), ("SIS", (1, 0, 0))):
            result = simulate_micro(n_communities=1, community_size=1, inter_links=0,
                                    beta=0, gamma=1, T_end=1, dt_out=0.25,
                                    initial_node=0, model=model, seed=123)
            self.assertEqual(len(result["events"]), 2)
            self.assertAlmostEqual(result["events"][1]["time"], expected)
            self.assertEqual(result["rows"][-1][2:], final)

    def test_events_beyond_horizon_are_not_reported_or_applied(self):
        result = simulate_micro(n_communities=1, community_size=1, inter_links=0,
                                beta=0, gamma=0.1, T_end=0.1, dt_out=0.1,
                                initial_node=0, seed=123)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["rows"][-1][2:], (0, 1, 0))

    def test_invalid_simulation_parameters_fail_before_running(self):
        defaults = dict(network=self.network(), beta=0.4, gamma=0.2,
                        T_end=1, dt_out=0.5, initial_node=0, seed=11)
        for values in ({"beta": -1}, {"gamma": float("inf")}, {"gamma": True},
                       {"T_end": 0}, {"T_end": float("nan")}, {"dt_out": 0},
                       {"dt_out": float("inf")}, {"model": "unknown"}, {"model": True},
                       {"initial_node": 100}, {"initial_node": True}, {"seed": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                simulate_micro(**(defaults | values))


class WeightedGraphTests(unittest.TestCase):
    def test_single_weighted_transmission_has_expected_wait(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=2.0)
        network = TwoLayerNetwork([graph.copy()], graph, np.zeros((1, 1)), [[0, 1]])
        result = simulate_micro(network=network, beta=0.5, gamma=0, T_end=1,
                                dt_out=0.5, initial_node=0, seed=123)
        self.assertEqual(len(result["events"]), 2)
        self.assertAlmostEqual(result["events"][1]["time"], random.Random(123).expovariate(1.0))
        self.assertEqual(result["rows"][-1][2:], (0, 2, 0))

    def test_overflowing_rates_fail_cleanly(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1e308)
        network = TwoLayerNetwork([graph.copy()], graph, np.zeros((1, 1)), [[0, 1]])
        with self.assertRaises(ValueError):
            simulate_micro(network=network, beta=1e308, gamma=0, T_end=1,
                           dt_out=0.5, initial_node=0, seed=123)


class SmallRateTests(unittest.TestCase):
    def test_small_positive_rates_remain_enabled(self):
        from two_layer_ctmc.engines.micro_model import MicroModel
        for engine_type in (MicroEngine, MicroModel):
            with self.subTest(engine=engine_type.__name__):
                engine = engine_type(0.0, 1e-15, model=2, rng=random.Random(123))
                engine.add_node(0)
                engine._infect_node(0)
                wait, event = engine._sample_next_event()
                self.assertEqual(event, ("recovery", 0))
                self.assertAlmostEqual(wait / 1e15, random.Random(123).expovariate(1.0))
                if engine_type is MicroModel:
                    cached_wait, cached_event, _, _ = engine._ensure_next_event()
                    self.assertGreater(cached_wait, 0)
                    self.assertEqual(cached_event, ("recovery", 0))
