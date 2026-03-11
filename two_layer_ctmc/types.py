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


class TransmissionEvent(TypedDict, total=False):
    time: float
    kind: str
    src_node: Optional[int]
    dst_node: Optional[int]
    src_community: Optional[int]
    dst_community: int


class CommunityTiming(TypedDict, total=False):
    community: int
    first_infection_time: Optional[float]
    first_bridge_infection_time: Optional[float]
    first_export_time: Optional[float]
    first_import_time: Optional[float]
    first_import_source_community: Optional[int]
    time_to_bridge: Optional[float]
    time_to_export: Optional[float]
    time_to_import: Optional[float]


class MicroSimulationResult(TypedDict, total=False):
    rows: List[MicroRunRow]
    infection_events: List[InfectionEvent]
    initial_node: Optional[int]
    transmission_events: List[TransmissionEvent]
    community_timings: List[CommunityTiming]
    bridge_nodes: Dict[int, List[int]]


MicroMacroSimulationResult: TypeAlias = OrchestratorResult
