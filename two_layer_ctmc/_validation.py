"""Shared input checks for the supported public simulation API."""
import math
from numbers import Integral

import networkx as nx
import numpy as np


def nonnegative(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def positive(value, name):
    result = nonnegative(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def seed_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be an integer or None")
    return int(value)


def model_id(value):
    if isinstance(value, str):
        value = {"sir": 2, "sis": 1}.get(value.strip().lower())
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (1, 2):
        raise ValueError("model must be 'SIR', 'SIS', 1 (SIS), or 2 (SIR)")
    return int(value)


def graph(value, name="graph"):
    if not isinstance(value, nx.Graph) or value.is_directed() or value.is_multigraph():
        raise ValueError(f"{name} must be a simple undirected NetworkX graph")
    if not value.number_of_nodes():
        raise ValueError(f"{name} must contain at least one node")
    if nx.number_of_selfloops(value):
        raise ValueError(f"{name} must not contain self-loops")
    for node in value:
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral):
            raise ValueError(f"{name} node IDs must be integers")
    for _, _, data in value.edges(data=True):
        nonnegative(data.get("weight", 1.0), f"{name} edge weight")


def matrix(value, n):
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inter_matrix must be a numeric square matrix") from exc
    if result.shape != (n, n):
        raise ValueError("inter_matrix shape must match the number of communities")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("inter_matrix must be finite and nonnegative")
    if np.any(np.diag(result) != 0) or not np.allclose(result, result.T, rtol=1e-12, atol=1e-12):
        raise ValueError("inter_matrix must be symmetric with a zero diagonal")
    return result
