"""Compact ordered topology storage; no stochastic operations or reductions."""
from array import array
from collections.abc import Sequence


class NeighborList(Sequence):
    """List-compatible iteration over (node, weight), using int64/float64 arrays.

    Iteration returns Python ints/floats, preserving the old scalar arithmetic.
    Non-int or out-of-int64 node IDs fall back to a list without relabeling.
    Duplicate entries and self-loops are deliberately not deduplicated.
    """

    __slots__ = ("_nodes", "_weights", "_fallback")

    def __init__(self):
        self._nodes = array("q")
        self._weights = array("d")
        self._fallback = None

    def append(self, item):
        node, weight = item
        if self._fallback is not None:
            self._fallback.append(item)
        elif type(node) is int and -(1 << 63) <= node < (1 << 63):
            self._nodes.append(node)
            self._weights.append(weight)
        else:
            self._fallback = list(zip(self._nodes, self._weights))
            self._fallback.append(item)
            self._nodes = array("q")
            self._weights = array("d")

    def __len__(self):
        return len(self._fallback) if self._fallback is not None else len(self._nodes)

    def __iter__(self):
        if self._fallback is not None:
            return iter(self._fallback)
        return zip(self._nodes, self._weights)

    def __getitem__(self, index):
        if self._fallback is not None:
            return self._fallback[index]
        if isinstance(index, slice):
            return list(zip(self._nodes[index], self._weights[index]))
        return self._nodes[index], self._weights[index]

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))
