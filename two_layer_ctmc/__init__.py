"""SIR/SIS simulation on networks of connected communities."""
from .simulate import (
    TwoLayerNetwork,
    build_two_layer_network,
    simulate_micro,
    simulate_micromacro,
)

__all__ = [
    "TwoLayerNetwork",
    "build_two_layer_network",
    "simulate_micro",
    "simulate_micromacro",
]
