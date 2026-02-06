from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeAlias


class EventLogEntry(TypedDict, total=False):
    time: float
    wait_time: float
    event_type: str
    mode: str
    community: int
    node: Optional[int]
    src: Optional[int]
    hazard_matrix: Any
    total_hazard: float
    states: Any


LogsPerCommunity: TypeAlias = Dict[int, Dict[str, List[int]]]
OrchestratorResult: TypeAlias = Tuple[List[float], List[int], LogsPerCommunity, List[EventLogEntry]]

MicroRunRow: TypeAlias = Tuple[int, float, int, int, int]
InfectionEvent: TypeAlias = Tuple[float, int, str, int]


class MicroSimulationResult(TypedDict):
    rows: List[MicroRunRow]
    infection_events: List[InfectionEvent]
    initial_node: Optional[int]


MicroMacroSimulationResult: TypeAlias = OrchestratorResult
