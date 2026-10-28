from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ...engines.micro_model import MicroModel
from .config import ModelVariant, ScenarioConfig


@dataclass
class SchedulerResult:
    raw_records: list[dict]
    arrivals: list[float]
    completion: bool
    micro_events: int
    macro_events: int
    null_macro_events: int
    trajectory_records: list[dict]
    successful_macro_imports: int = 0
    arrival_observed: list[bool] | None = None


class LagHistory:
    """Fixed-grid history with a provisional current observation for tau -> 0."""

    def __init__(self, initial: np.ndarray):
        self.times = [0.0]
        self.values = [np.asarray(initial, dtype=float).copy()]

    def append_grid(self, time: float, value: np.ndarray) -> None:
        if time <= self.times[-1] + 1e-12:
            if abs(time - self.times[-1]) <= 1e-12:
                self.values[-1] = np.asarray(value, dtype=float).copy()
                return
            raise ValueError("lag history times must increase")
        self.times.append(float(time))
        self.values.append(np.asarray(value, dtype=float).copy())

    def evaluate(
        self,
        time: float,
        delays: np.ndarray,
        provisional_time: float,
        provisional_value: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros_like(self.values[0])
        provisional_value = np.asarray(provisional_value, dtype=float)
        for i, delay in enumerate(delays):
            if not math.isfinite(float(delay)):
                result[i] = 0.0
                continue
            query = max(0.0, float(time) - float(delay))
            if query >= self.times[-1] and provisional_time > self.times[-1] + 1e-12:
                fraction = min(1.0, max(0.0, (query - self.times[-1]) / (provisional_time - self.times[-1])))
                result[i] = self.values[-1][i] + fraction * (provisional_value[i] - self.values[-1][i])
                continue
            right = int(np.searchsorted(self.times, query, side="right"))
            if right == 0:
                result[i] = self.values[0][i]
            elif right >= len(self.times):
                result[i] = self.values[-1][i]
            else:
                left = right - 1
                width = self.times[right] - self.times[left]
                fraction = 0.0 if width <= 0 else (query - self.times[left]) / width
                result[i] = self.values[left][i] + fraction * (self.values[right][i] - self.values[left][i])
        return result


class ExperimentalMicroMacroScheduler:
    """Production-aligned midpoint scheduler with isolated corrected hazards."""

    def __init__(
        self,
        W: np.ndarray,
        micro_graphs: Sequence[nx.Graph],
        config: ScenarioConfig,
        variant: ModelVariant,
        descriptors: Sequence[object],
        seeds: Mapping[str, int],
        collect_diagnostics: bool = True,
        collect_trajectories: bool = True,
    ):
        if variant == ModelVariant.V0_FULL_MICRO:
            raise ValueError("V0 uses the production full-network Micro simulator")
        self.W = np.asarray(W, dtype=float)
        self.graphs = list(micro_graphs)
        self.config = config
        self.variant = variant
        self.descriptors = list(descriptors)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.collect_trajectories = bool(collect_trajectories)
        self.sizes = np.asarray([g.number_of_nodes() for g in self.graphs], dtype=float)
        self.micro_rng = random.Random(int(seeds["micro_event_rng"]))
        self.budget_rng = random.Random(int(seeds["macro_budget_rng"]))
        self.pair_rng = random.Random(int(seeds["macro_pair_rng"]))
        self.placement_rng = random.Random(int(seeds["placement_rng"]))
        self.models: list[MicroModel] = []
        for graph in self.graphs:
            model = MicroModel(config.beta_in, config.gamma, model=2, rng=self.micro_rng)
            for node in graph.nodes():
                model.add_node(int(node))
            for u, v, data in graph.edges(data=True):
                model.add_edge(int(u), int(v), float(data.get("weight", 1.0)))
            self.models.append(model)

        self.tau_find = np.asarray([float(getattr(d, "tau_find")) for d in descriptors])
        self.kappa = np.asarray([float(getattr(d, "kappa")) for d in descriptors])
        self.degenerate = np.asarray([bool(getattr(d, "degenerate_descriptors")) for d in descriptors])

    def _counts(self, models: Optional[Sequence[MicroModel]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        use = self.models if models is None else models
        states = [model.count_states() for model in use]
        return tuple(np.asarray([state[k] for state in states], dtype=float) for k in range(3))  # type: ignore[return-value]

    def _hazards(self, contagiousness: np.ndarray, susceptible: np.ndarray) -> np.ndarray:
        fraction = contagiousness.reshape(-1, 1) * (susceptible / self.sizes).reshape(1, -1)
        hazards = self.config.beta_out * self.config.macro_T * self.W * fraction
        if not np.isfinite(hazards).all():
            raise FloatingPointError("macro hazard matrix contains NaN or infinity")
        if np.any(hazards < 0.0):
            raise ValueError("macro hazard matrix contains a negative intensity")
        return hazards

    def _sample_pair(self, hazards: np.ndarray) -> tuple[int, int]:
        total = float(hazards.sum())
        if total <= 0.0:
            return -1, -1
        threshold = self.pair_rng.random() * total
        cumulative = 0.0
        for index, value in enumerate(hazards.ravel()):
            cumulative += float(value)
            if cumulative > threshold:
                return divmod(index, hazards.shape[1])
        return divmod(hazards.size - 1, hazards.shape[1])

    def _place_import(self, destination: int, time: float) -> Optional[int]:
        model = self.models[destination]
        susceptible = [
            int(node) for node, data in model.G.nodes(data=True)
            if not data["infected"] and not data["recovered"]
        ]
        if not susceptible:
            return None
        node = self.placement_rng.choice(susceptible)
        model.current_time = float(time)
        model._infect_node(node)
        return node

    def _advance(self, target: float) -> list[tuple[int, float, str, int, Optional[int]]]:
        events = []
        for community, model in enumerate(self.models):
            _, _, _, observed = model.simulate_until(target)
            events.extend((community, float(t), kind, int(node), None if src is None else int(src)) for t, kind, node, src in observed)
        events.sort(key=lambda row: (row[1], row[0], row[3]))
        return events

    def _midpoint_counts(self, target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self.micro_rng.getstate()
        clones = [model.clone() for model in self.models]
        try:
            for clone, source in zip(clones, self.models):
                clone.refresh_from(source)
                clone.simulate_until(target)
            return self._counts(clones)
        finally:
            self.micro_rng.setstate(state)

    def run(self, initial_community: int, initial_node: int) -> SchedulerResult:
        self.models[initial_community]._infect_node(int(initial_node))
        S, I, _ = self._counts()
        initial_fraction = I / self.sizes
        lag_history = LagHistory(initial_fraction)
        latent = np.zeros_like(initial_fraction)
        combined = np.zeros_like(initial_fraction)
        raw: list[dict] = [{
            "record_type": "event", "event_mode": "seed", "time": 0.0,
            "community": initial_community, "node": int(initial_node),
        }]
        trajectories: list[dict] = []
        if self.collect_trajectories:
            self._append_trajectory(raw, trajectories, 0.0)
        arrivals: list[Optional[float]] = [None] * len(self.models)
        arrivals[initial_community] = 0.0
        budget = -math.log(max(self.budget_rng.random(), np.finfo(float).tiny))
        accumulated = 0.0
        t = 0.0
        next_grid = min(self.config.tau_micro, self.config.T_end)
        micro_count = 0
        macro_clock_count = 0
        import_count = 0
        null_count = 0
        logged_micro = 0

        while t < self.config.T_end - 1e-12:
            segment_end = next_grid
            dt = segment_end - t
            if dt <= 1e-12:
                next_grid = min(next_grid + self.config.tau_micro, self.config.T_end)
                continue
            S_start, I_start, _ = self._counts()
            frac_start = I_start / self.sizes
            lag_start = lag_history.evaluate(t, self.tau_find, t, frac_start)
            midpoint = t + 0.5 * dt
            S_mid, I_mid, _ = self._midpoint_counts(midpoint)
            frac_mid = I_mid / self.sizes
            lag_mid = lag_history.evaluate(midpoint, self.tau_find, midpoint, frac_mid)
            # Use the same prospective midpoint summaries as the production
            # timing hazard.  Besides being a midpoint discretization of the
            # relaxation driver, this preserves the required hard limit:
            # kappa=inf is pathwise identical to the corresponding unfiltered
            # source even when micro events occur before the midpoint.
            latent_mid = self._relax(latent, frac_mid, 0.5 * dt)
            combined_mid = self._relax(combined, lag_mid, 0.5 * dt)
            source_mid = self._source_for_variant(frac_mid, lag_mid, latent_mid, combined_mid)
            timing_hazards = self._hazards(source_mid, S_mid)
            total_mid = float(timing_hazards.sum())
            if self.collect_diagnostics:
                boundary_mid = (
                    combined_mid
                    if self.variant == ModelVariant.V4_COMBINED
                    else latent_mid
                )
                self._append_hazard_diagnostics(
                    raw, t, midpoint, timing_hazards, source_mid, frac_mid, lag_mid,
                    boundary_mid, accumulated, budget,
                )

            if total_mid > 0.0 and accumulated + total_mid * dt >= budget:
                event_time = t + (budget - accumulated) / total_mid
                events = self._advance(event_time)
                micro_count += len(events)
                logged_micro = self._append_micro_events(raw, events, logged_micro)
                S_event, I_event, _ = self._counts()
                frac_event = I_event / self.sizes
                lag_event = lag_history.evaluate(event_time, self.tau_find, event_time, frac_event)
                latent_event = self._relax(latent, frac_event, event_time - t)
                combined_event = self._relax(combined, lag_event, event_time - t)
                source_event = self._source_for_variant(frac_event, lag_event, latent_event, combined_event)
                mark_hazards = self._hazards(source_event, S_event)
                source, destination = self._sample_pair(mark_hazards)
                macro_clock_count += 1
                node = None
                if source >= 0 and destination >= 0 and source != destination:
                    node = self._place_import(destination, event_time)
                if node is None:
                    null_count += 1
                else:
                    import_count += 1
                    if arrivals[destination] is None:
                        arrivals[destination] = event_time
                raw.append({
                    "record_type": "event", "event_mode": "macro", "time": event_time,
                    "source_i": source, "dest_j": destination, "target_node": node,
                    "i_bar": None if source < 0 else float(frac_event[source]),
                    "s_bar": None if destination < 0 else float(S_event[destination] / self.sizes[destination]),
                    "i_lag": None if source < 0 or self.variant not in (ModelVariant.V2_LAG, ModelVariant.V4_COMBINED) else float(lag_event[source]),
                    "i_boundary": None if source < 0 or self.variant not in (ModelVariant.V3_LATENT, ModelVariant.V4_COMBINED) else float(source_event[source]),
                    "lambda_ij": None if source < 0 else float(mark_hazards[source, destination]),
                    "null_event": node is None,
                })
                latent = latent_event
                combined = combined_event
                t = event_time
                budget = -math.log(max(self.budget_rng.random(), np.finfo(float).tiny))
                accumulated = 0.0
                continue

            events = self._advance(segment_end)
            micro_count += len(events)
            logged_micro = self._append_micro_events(raw, events, logged_micro)
            accumulated += total_mid * dt
            latent = self._relax(latent, frac_mid, dt)
            combined = self._relax(combined, lag_mid, dt)
            t = segment_end
            S_grid, I_grid, _ = self._counts()
            fraction_grid = I_grid / self.sizes
            lag_history.append_grid(t, fraction_grid)
            if self.collect_trajectories:
                self._append_trajectory(raw, trajectories, t)
            for community, value in enumerate(I_grid):
                if value > 0 and arrivals[community] is None:
                    # Imports are event-logged exactly; this fallback covers only a
                    # logically impossible missed first import and is diagnostic.
                    arrivals[community] = t
            if t >= self.config.T_end - 1e-12:
                break
            next_grid = min(next_grid + self.config.tau_micro, self.config.T_end)

        censored = [self.config.T_end if value is None else float(value) for value in arrivals]
        return SchedulerResult(
            raw_records=raw,
            arrivals=censored,
            completion=arrivals[-1] is not None,
            micro_events=micro_count,
            macro_events=macro_clock_count,
            null_macro_events=null_count,
            trajectory_records=trajectories,
            successful_macro_imports=import_count,
            arrival_observed=[value is not None for value in arrivals],
        )

    def _relax(self, state: np.ndarray, source: np.ndarray, dt: float) -> np.ndarray:
        factor = np.zeros_like(self.kappa)
        valid = (~self.degenerate) & (self.kappa > 0.0)
        factor[valid] = -np.expm1(-self.kappa[valid] * max(0.0, dt))
        result = state + factor * (source - state)
        result[self.degenerate] = 0.0
        return result

    def _source_for_variant(self, baseline, lagged, latent, combined):
        if self.variant == ModelVariant.V1_BASELINE:
            return baseline
        if self.variant == ModelVariant.V2_LAG:
            return np.where(self.degenerate, 0.0, lagged)
        if self.variant == ModelVariant.V3_LATENT:
            return latent
        if self.variant == ModelVariant.V4_COMBINED:
            return combined
        raise ValueError(self.variant)

    def _append_micro_events(self, raw, events, already_logged: int) -> int:
        for community, time, kind, node, source in events:
            if already_logged < 100:
                raw.append({
                    "record_type": "event", "event_mode": "micro", "time": time,
                    "community": community, "event_type": kind, "node": node, "src_node": source,
                })
                already_logged += 1
        return already_logged

    def _append_trajectory(self, raw, trajectories, time: float) -> None:
        for community, model in enumerate(self.models):
            S, I, R = model.count_states()
            row = {
                "record_type": "trajectory", "time": float(time), "community": community,
                "S": int(S), "I": int(I), "R": int(R),
            }
            raw.append(row)
            trajectories.append(row.copy())

    def _append_hazard_diagnostics(
        self, raw, segment_start, evaluation_time, hazards, source_effective,
        source_actual, lagged, latent, accumulated, budget,
    ) -> None:
        total = float(hazards.sum())
        for source in range(hazards.shape[0]):
            for destination in range(hazards.shape[1]):
                if source == destination or self.W[source, destination] <= 0.0:
                    continue
                raw.append({
                    "record_type": "hazard", "time": float(segment_start),
                    "evaluation_time": float(evaluation_time), "source_i": source,
                    "dest_j": destination, "i_bar": float(source_actual[source]),
                    "i_lag": float(lagged[source]), "i_boundary": float(latent[source]),
                    "effective_i": float(source_effective[source]),
                    "lambda_ij": float(hazards[source, destination]), "Theta": total,
                    "accumulated_hazard": float(accumulated), "macro_budget": float(budget),
                    "degenerate_descriptors": bool(self.degenerate[source]),
                })
