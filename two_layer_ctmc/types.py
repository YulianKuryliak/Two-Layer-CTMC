"""In-memory result types; no storage or analysis dependencies."""
from typing import Dict, List, Optional, Tuple, TypeAlias, TypedDict


class EventLogEntry(TypedDict, total=False):
    time: float
    wait_time: float
    event_type: str
    mode: str
    community: int
    node: Optional[int]
    src: Optional[int]
    total_hazard: float


MicroRunRow: TypeAlias = Tuple[int, float, int, int, int]
LogsPerCommunity: TypeAlias = Dict[int, Dict[str, list]]
MicroMacroSimulationResult: TypeAlias = Tuple[
    List[float], List[int], LogsPerCommunity, List[EventLogEntry]
]


class MicroSimulationResult(TypedDict):
    rows: List[MicroRunRow]
    events: List[EventLogEntry]
    initial_node: int
