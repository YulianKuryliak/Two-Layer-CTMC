"""Tiny deterministic reference for contact groups and home-city block sums.

This is an algebraic specification, not a country generator or simulator.
"""
from collections import defaultdict
from itertools import combinations
import math


def merge_group_contacts(groups):
    """groups: iterable of (distinct person IDs, contacts/person/unit-time)."""
    weights = defaultdict(float)
    for people, kappa in groups:
        people = tuple(people)
        if len(set(people)) != len(people):
            raise ValueError("A group cannot contain one person twice")
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if len(people) < 2:
            continue
        weight = kappa / (len(people) - 1)
        for a, b in combinations(sorted(people), 2):
            if weight:
                weights[a, b] += weight
    return dict(weights)


def split_by_home(edges, home):
    """Return local edges and oriented W entries; W diagonal is absent/zero."""
    local = defaultdict(dict)
    macro = defaultdict(float)
    for (a, b), weight in edges.items():
        i, j = home[a], home[b]
        if i == j:
            local[i][a, b] = weight
        else:
            macro[i, j] += weight
            macro[j, i] += weight
    return dict(local), dict(macro)
