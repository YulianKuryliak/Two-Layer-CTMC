"""Regression checks for the community SSA clock interrupted by imports."""
import math
import random
import unittest

from two_layer_ctmc.engines.micro_model import MicroModel


class CountingRandom(random.Random):
    def __init__(self, seed, target=2):
        super().__init__(seed)
        self.target = target
        self.exponential_draws = 0

    def expovariate(self, rate):
        self.exponential_draws += 1
        return super().expovariate(rate)

    def choice(self, seq):
        return self.target if self.target in seq else seq[0]


def path_model(rng, n=4):
    model = MicroModel(1.0, 0.0, rng=rng)
    for node in range(n):
        model.add_node(node)
    for node in range(n - 1):
        model.add_edge(node, node + 1, 1.0)
    model._infect_node(0)
    return model


class ResidualClockTests(unittest.TestCase):
    def test_import_preserves_budget_and_rescales_time(self):
        rng = CountingRandom(19)
        model = path_model(rng)
        old_wait, _, _, after = model._ensure_next_event()
        when = old_wait / 4
        model.simulate_until(when)
        self.assertEqual(model.rng.getstate(), after)  # no rollback
        self.assertEqual(model.import_infection(at_time=when), 2)
        self.assertEqual(model.total_infection_rate, 3.0)
        wait, event, _, _ = model._ensure_next_event()
        self.assertAlmostEqual(wait, (old_wait - when) / 3)
        self.assertEqual(rng.exponential_draws, 1)  # no new threshold
        self.assertTrue(event[2] in (1, 3))  # fresh valid mark

    def test_multiple_imports_spend_only_elapsed_intensity(self):
        model = path_model(CountingRandom(31), n=6)
        old_wait = model._ensure_next_event()[0]
        when = old_wait / 4
        model.simulate_until(when)
        model.import_infection(at_time=when)
        wait = model._ensure_next_event()[0]
        next_import = when + wait / 2
        model.simulate_until(next_import)
        model.rng.target = 4
        model.import_infection(at_time=next_import)
        new_wait = model._ensure_next_event()[0]
        self.assertAlmostEqual(new_wait * model.total_infection_rate,
                               (old_wait - when) / 2)
        self.assertEqual(model.rng.exponential_draws, 1)

    def test_clone_and_refresh_preserve_clock_without_rng_leak(self):
        model = path_model(random.Random(7), n=6)
        wait = model._ensure_next_event()[0]
        state = model.rng.getstate()
        clone = model.clone()
        self.assertIsNot(clone.rng, model.rng)
        self.assertEqual(clone._ensure_next_event(), model._ensure_next_event())
        expected = clone.simulate_until(wait + 1)
        self.assertEqual(state, model.rng.getstate())
        actual = model.simulate_until(wait + 1)
        self.assertEqual(actual, expected)
        clone.refresh_from(model)
        self.assertEqual(clone.rng.getstate(), model.rng.getstate())
        self.assertEqual(clone._ensure_next_event(), model._ensure_next_event())

    def test_consecutive_imports_without_advance_spend_updated_intensity(self):
        model = path_model(CountingRandom(31), n=6)
        old_wait = model._ensure_next_event()[0]
        when = old_wait / 4
        model.import_infection(at_time=when)
        wait = model._cached_wait
        next_import = when + wait / 2
        model.rng.target = 4
        model.import_infection(at_time=next_import)
        self.assertAlmostEqual(model._cached_wait * model.total_infection_rate,
                               (old_wait - when) / 2)
        self.assertEqual(model.rng.exponential_draws, 1)

    def test_import_activates_fresh_clock_in_uninfected_community(self):
        model = MicroModel(1, 0, rng=CountingRandom(5, target=0))
        for node in range(2):
            model.add_node(node)
        model.add_edge(0, 1, 1)
        model.simulate_until(10)
        model.import_infection(at_time=10)
        self.assertTrue(math.isfinite(model._ensure_next_event()[0]))
        self.assertEqual(model.rng.exponential_draws, 1)

    def test_import_after_unprocessed_local_event_is_rejected(self):
        model = path_model(random.Random(9))
        wait = model._ensure_next_event()[0]
        with self.assertRaises(ValueError):
            model.import_infection(at_time=wait + 1)

    def test_decreased_rate_delays_pending_event(self):
        model = MicroModel(1, 0, rng=CountingRandom(12, target=2))
        for node in range(4):
            model.add_node(node)
        for u in range(4):
            for v in range(u + 1, 4):
                model.add_edge(u, v, 1)
        model._infect_node(0)
        model._infect_node(1)
        self.assertEqual(model.total_infection_rate, 4)
        wait = model._ensure_next_event()[0]
        when = wait / 4
        model.import_infection(at_time=when)
        self.assertEqual(model.total_infection_rate, 3)
        self.assertAlmostEqual(model._cached_wait, (wait-when)*4/3)

    def test_clone_after_import_carries_rescaled_proposal(self):
        model = path_model(random.Random(39), n=6)
        when = model._ensure_next_event()[0] / 4
        model.import_infection(at_time=when)
        clone = model.clone()
        self.assertEqual(clone._pending_origin, when)
        self.assertEqual(clone._pending_rate, model.total_infection_rate)
        horizon = when + model._cached_wait + 1
        self.assertEqual(clone.simulate_until(horizon), model.simulate_until(horizon))

    def test_import_can_pause_clock_at_zero_rate(self):
        model = path_model(CountingRandom(5, target=1), n=2)
        wait = model._ensure_next_event()[0]
        when = wait / 2
        model.simulate_until(when)
        model.import_infection(at_time=when)
        budget = model._remaining_intensity_budget
        self.assertAlmostEqual(budget, wait / 2)
        model.simulate_until(when + 100)
        self.assertEqual(model._remaining_intensity_budget, budget)
        self.assertEqual(model.count_states(), (0, 2, 0))


if __name__ == "__main__":
    unittest.main()
