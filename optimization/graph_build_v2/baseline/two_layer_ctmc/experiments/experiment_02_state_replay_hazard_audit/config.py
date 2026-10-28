from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditConfig:
    experiment_name: str = "experiment_02_state_replay_hazard_audit"
    L: int = 10
    n: int = 100
    mean_internal_degree: float = 6.0
    k: int = 5
    beta_in: float = 0.003
    beta_out: float = 0.003
    beta_macro: float = 0.003
    macro_T: float = 1.0
    gamma: float = 0.0
    T_end: float = 4000.0
    base_seed: int = 270000
    initial_community: int = 0
    initial_node: int = 0
    match_abs_tolerance: float = 1e-12
    match_rel_tolerance: float = 1e-9
    phase_early_upper: float = 0.10
    phase_middle_upper: float = 0.50
    bootstrap_reps: int = 2000
    bootstrap_seed: int = 270002
    source_experiment_dir: str = "output/ablation_hazard_placement/final_100"

    @property
    def edge_prob(self) -> float:
        return self.mean_internal_degree / (self.n - 1)

    def to_dict(self) -> dict:
        return {**asdict(self), "edge_prob": self.edge_prob}

    @property
    def source_dir(self) -> Path:
        return Path(self.source_experiment_dir)


STAGE_RUNS = {"smoke": 5, "validation": 20, "final": 100}
