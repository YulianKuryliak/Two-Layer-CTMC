from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from typing import Dict


class ModelVariant(str, Enum):
    V0_FULL_MICRO = "V0"
    V1_BASELINE = "V1"
    V2_LAG = "V2"
    V3_LATENT = "V3"
    V4_COMBINED = "V4"


ALL_VARIANTS = tuple(ModelVariant)


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    L: int = 10
    n: int = 100
    k: int = 5
    mean_internal_degree: float = 6.0
    beta_in: float = 0.003
    beta_out: float = 0.003
    gamma: float = 0.0
    tau_micro: float = 1.0
    T_end: float = 4000.0
    macro_T: float = 1.0

    @property
    def edge_prob(self) -> float:
        return self.mean_internal_degree / (self.n - 1)

    @property
    def target_edges(self) -> int:
        return round(self.edge_prob * self.n * (self.n - 1) / 2)

    def to_dict(self) -> dict:
        return {**asdict(self), "edge_prob": self.edge_prob, "target_edges": self.target_edges}


SCENARIOS: Dict[str, ScenarioConfig] = {
    "baseline": ScenarioConfig("baseline"),
    "k1": ScenarioConfig("k1", k=1),
    "k10": ScenarioConfig("k10", k=10),
    "d4": ScenarioConfig("d4", mean_internal_degree=4.0),
    "d10": ScenarioConfig("d10", mean_internal_degree=10.0),
}

STREAM_NAMES = (
    "network_rng",
    "initialization_rng",
    "micro_event_rng",
    "macro_budget_rng",
    "macro_pair_rng",
    "placement_rng",
    "bootstrap_rng",
)


def derive_seed(master_seed: int, replication_id: int, stream_name: str, variant: str) -> int:
    payload = f"{master_seed}|{replication_id}|{stream_name}|{variant}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFFFFFF


def shared_seed(master_seed: int, replication_id: int, stream_name: str) -> int:
    if stream_name not in {"network_rng", "initialization_rng"}:
        raise ValueError(f"{stream_name} is not a shared stream")
    return derive_seed(master_seed, replication_id, stream_name, "SHARED")


def variant_seeds(master_seed: int, replication_id: int, variant: ModelVariant) -> Dict[str, int]:
    seeds = {
        name: derive_seed(master_seed, replication_id, name, variant.value)
        for name in STREAM_NAMES
    }
    seeds["network_rng"] = shared_seed(master_seed, replication_id, "network_rng")
    seeds["initialization_rng"] = shared_seed(master_seed, replication_id, "initialization_rng")
    return seeds


def stage_scenarios(stage: str, requested: str) -> tuple[str, ...]:
    allowed = {
        "smoke": ("baseline",),
        "validation": ("baseline", "k1"),
        "final": tuple(SCENARIOS),
    }
    if stage not in allowed:
        raise ValueError(f"unknown stage: {stage}")
    if requested == "all":
        return allowed[stage]
    if requested not in allowed[stage]:
        raise ValueError(f"scenario {requested!r} is not part of stage {stage!r}")
    return (requested,)
