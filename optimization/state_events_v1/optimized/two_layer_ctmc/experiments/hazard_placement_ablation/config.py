from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from typing import Dict, Optional


class HazardMode(str, Enum):
    EXACT_ACTIVE_EDGE = "EXACT_ACTIVE_EDGE"
    AGGREGATE_PREVALENCE = "AGGREGATE_PREVALENCE"


class PlacementMode(str, Enum):
    EXACT_EDGE_CONDITIONED = "EXACT_EDGE_CONDITIONED"
    BOUNDARY_WEIGHTED = "BOUNDARY_WEIGHTED"
    UNIFORM_SUSCEPTIBLE = "UNIFORM_SUSCEPTIBLE"


class ModelVariant(str, Enum):
    FULL_MICRO_REFERENCE = "FULL_MICRO_REFERENCE"
    ORACLE_EXACT_HAZARD_EXACT_PLACEMENT = "ORACLE_EXACT_HAZARD_EXACT_PLACEMENT"
    ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT = "ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT"
    AGGREGATE_HAZARD_BOUNDARY_PLACEMENT = "AGGREGATE_HAZARD_BOUNDARY_PLACEMENT"
    AGGREGATE_HAZARD_UNIFORM_PLACEMENT = "AGGREGATE_HAZARD_UNIFORM_PLACEMENT"


VARIANT_MODES = {
    ModelVariant.ORACLE_EXACT_HAZARD_EXACT_PLACEMENT: (
        HazardMode.EXACT_ACTIVE_EDGE,
        PlacementMode.EXACT_EDGE_CONDITIONED,
    ),
    ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT: (
        HazardMode.EXACT_ACTIVE_EDGE,
        PlacementMode.UNIFORM_SUSCEPTIBLE,
    ),
    ModelVariant.AGGREGATE_HAZARD_BOUNDARY_PLACEMENT: (
        HazardMode.AGGREGATE_PREVALENCE,
        PlacementMode.BOUNDARY_WEIGHTED,
    ),
    ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT: (
        HazardMode.AGGREGATE_PREVALENCE,
        PlacementMode.UNIFORM_SUSCEPTIBLE,
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "ablation_hazard_placement"
    L: int = 10
    n: int = 100
    mean_internal_degree: float = 6.0
    k: int = 5
    beta_in: float = 0.003
    beta_out: float = 0.003
    gamma: float = 0.0
    tau_micro: float = 1.0
    T_end: float = 4000.0
    base_seed: int = 270000
    initial_community: int = 0
    initial_node: Optional[int] = 0
    bootstrap_reps: int = 1000

    @property
    def edge_prob(self) -> float:
        return self.mean_internal_degree / (self.n - 1)

    def to_dict(self) -> dict:
        return {**asdict(self), "edge_prob": self.edge_prob}


STREAM_NAMES = (
    "network_rng",
    "initialization_rng",
    "micro_event_rng",
    "macro_budget_rng",
    "macro_pair_rng",
    "placement_rng",
    "bootstrap_rng",
)


def derive_seed(base_seed: int, replication_id: int, stream_name: str, model_variant: str) -> int:
    payload = f"{base_seed}|{replication_id}|{stream_name}|{model_variant}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFFFFFF


def stream_seeds(config: ExperimentConfig, replication_id: int, variant: ModelVariant) -> Dict[str, int]:
    return {
        name: derive_seed(config.base_seed, replication_id, name, variant.value)
        for name in STREAM_NAMES
    }


def shared_network_seed(config: ExperimentConfig, replication_id: int) -> int:
    return derive_seed(config.base_seed, replication_id, "network_rng", "SHARED_NETWORK")


def shared_initialization_seed(config: ExperimentConfig, replication_id: int) -> int:
    return derive_seed(config.base_seed, replication_id, "initialization_rng", "SHARED_INITIALIZATION")
