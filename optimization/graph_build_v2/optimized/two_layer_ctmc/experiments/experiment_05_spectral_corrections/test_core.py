import math
import unittest

import networkx as nx
import numpy as np

from .corrections import (
    CombinedLagLatentPolicy,
    FixedGridHistory,
    LagPolicy,
    LatentBoundaryPolicy,
    aggregate_hazard,
)
from .graph_descriptors import (
    SpectralConvergenceError,
    compute_graph_descriptors,
    spectral_radius_power_iteration,
)


class GraphDescriptorTests(unittest.TestCase):
    def test_shifted_iteration_handles_bipartite_graph(self):
        result = spectral_radius_power_iteration(nx.path_graph(5))
        expected = 2.0 * math.cos(math.pi / 6.0)
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertLessEqual(result.residual_norm, 1.0e-6 * max(1.0, result.value))

    def test_clique_descriptors_match_closed_form(self):
        descriptors = compute_graph_descriptors(nx.complete_graph(5), beta_in=0.2)
        self.assertAlmostEqual(descriptors.spectral_radius, 4.0, places=10)
        self.assertEqual(descriptors.average_shortest_path_length, 1.0)
        self.assertAlmostEqual(descriptors.mean_degree, 4.0)
        self.assertAlmostEqual(descriptors.tau_find, math.log(5) / 0.6)
        self.assertAlmostEqual(descriptors.kappa, 0.6)
        self.assertFalse(descriptors.degenerate_descriptors)

    def test_disconnected_graph_has_no_replacement_path_length(self):
        graph = nx.Graph([(0, 1), (2, 3)])
        descriptors = compute_graph_descriptors(graph, beta_in=0.2)
        self.assertIsNone(descriptors.average_shortest_path_length)
        self.assertTrue(descriptors.degenerate_descriptors)
        self.assertTrue(math.isinf(descriptors.tau_find))
        self.assertEqual(descriptors.kappa, 0.0)

    def test_nonconvergence_is_explicit(self):
        with self.assertRaises(SpectralConvergenceError):
            spectral_radius_power_iteration(
                nx.path_graph(7), tolerance=1.0e-15, max_iterations=1
            )

    def test_weighted_radius_matches_exact_eigenvalue(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 0.5), (0, 2, 1.0)])
        adjacency = nx.to_numpy_array(graph, weight="weight")
        expected = float(np.linalg.eigvalsh(adjacency)[-1])
        observed = spectral_radius_power_iteration(graph).value
        self.assertAlmostEqual(observed, expected, places=6)


class CorrectionPolicyTests(unittest.TestCase):
    def test_fixed_grid_history_interpolates(self):
        history = FixedGridHistory(1.0)
        history.append(0.0, 0.0)
        history.append(1.0, 0.4)
        history.append(2.0, 1.0)
        self.assertAlmostEqual(history.value_at(1.5), 0.7)
        self.assertEqual(history.value_at(3.0), 1.0)

    def test_lag_uses_max_zero_time(self):
        lag = LagPolicy(delay=1.5, sync_step=1.0)
        lag.record(0.0, 0.2)
        lag.record(1.0, 0.4)
        lag.record(2.0, 0.8)
        self.assertEqual(lag.contagiousness(1.0), 0.2)
        self.assertAlmostEqual(lag.contagiousness(2.0), 0.3)

    def test_lambda_one_is_degenerate_for_lag_and_latent(self):
        descriptors = compute_graph_descriptors(nx.path_graph(2), beta_in=0.2)
        self.assertAlmostEqual(descriptors.spectral_radius, 1.0)
        lag = LagPolicy.from_descriptors(descriptors, sync_step=1.0)
        lag.record(0.0, 1.0)
        latent = LatentBoundaryPolicy.from_descriptors(descriptors)
        self.assertEqual(lag.contagiousness(100.0), 0.0)
        self.assertEqual(latent.advance(1.0, 100.0), 0.0)
        self.assertEqual(
            aggregate_hazard(0.3, 5.0, 1.0, 1.0, degenerate=True), 0.0
        )

    def test_zero_lag_limit_matches_baseline(self):
        lag = LagPolicy(delay=0.0, sync_step=1.0)
        lag.record(0.0, 0.1)
        lag.record(1.0, 0.5)
        lag.record(2.0, 0.9)
        self.assertAlmostEqual(lag.contagiousness(1.25), 0.6)

    def test_infinite_kappa_limit_matches_driving_state(self):
        latent = LatentBoundaryPolicy(math.inf)
        self.assertEqual(latent.advance(0.75, 1.0), 0.75)
        self.assertEqual(latent.advance(0.25, 0.5), 0.25)

    def test_segment_exact_update_has_semigroup_property(self):
        one_step = LatentBoundaryPolicy(0.7)
        split_step = LatentBoundaryPolicy(0.7)
        expected = one_step.advance(0.8, 2.0)
        split_step.advance(0.8, 0.75)
        observed = split_step.advance(0.8, 1.25)
        self.assertAlmostEqual(observed, expected, places=14)

    def test_combined_policy_uses_left_segment_lagged_input(self):
        combined = CombinedLagLatentPolicy(
            LagPolicy(delay=0.5, sync_step=1.0), LatentBoundaryPolicy(kappa=1.0)
        )
        combined.record(0.0, 0.0)
        combined.record(1.0, 1.0)
        combined.record(2.0, 1.0)
        observed = combined.advance(segment_start=1.0, duration=1.0)
        expected = (1.0 - math.exp(-1.0)) * 0.5
        self.assertAlmostEqual(observed, expected)

    def test_combined_zero_lag_infinite_kappa_limit_matches_baseline(self):
        combined = CombinedLagLatentPolicy(
            LagPolicy(delay=0.0, sync_step=1.0),
            LatentBoundaryPolicy(kappa=math.inf),
        )
        combined.record(0.0, 0.2)
        combined.record(1.0, 0.8)
        combined.record(2.0, 1.0)
        self.assertAlmostEqual(combined.value_after(1.5, 0.25), 0.9)
        self.assertAlmostEqual(combined.advance(1.5, 0.25), 0.9)


if __name__ == "__main__":
    unittest.main()
