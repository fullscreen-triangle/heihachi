"""
core.py -- shared primitives for the validation suite of
"Searching Continuous Audio as a Whole Item".

Every structural object in the paper is a finite weighted graph or a
construction on one. This module provides:

  * exact minimum cuts against a medium vertex (Ford-Fulkerson via networkx),
  * random constituent contact-graph construction with a positive floor,
  * signature / content-set objects and handle re-encodings,
  * edit-distance sequence alignment (Needleman-Wunsch global,
    Smith-Waterman-style local) with match/substitution/gap costs,
  * least-sufficient-identifier resolution and a name/acoustic cost model.

No claim is assumed; each experiment script checks a theorem by direct
computation on these primitives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def rng(seed: int) -> np.random.Generator:
    """A fresh, seeded PRNG so every experiment is reproducible."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Contact graphs and exact minimum cuts
# ---------------------------------------------------------------------------

MEDIUM = "m"  # the catalogue / medium vertex


def random_contact_graph(
    n_items: int,
    floor: float,
    edge_prob: float,
    weight_spread: float,
    gen: np.random.Generator,
) -> nx.Graph:
    """
    Construct a random constituent contact graph (Def. 3.3 / 3.4).

    Vertices: a medium vertex ``MEDIUM`` adjacent to every item, plus
    ``n_items`` item vertices ``0..n_items-1``. Every edge weight is >= floor
    (the derived positive floor, Thm. 4.3), so the graph satisfies the paper's
    premise by construction.
    """
    g = nx.Graph()
    items = list(range(n_items))
    g.add_node(MEDIUM)
    g.add_nodes_from(items)

    # Medium is adjacent to every item (Def. 3.4): each item is individuated
    # against the whole at cost >= floor.
    for v in items:
        w = floor + weight_spread * float(gen.random())
        g.add_edge(MEDIUM, v, weight=w)

    # Random item-item contacts, also floored.
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if gen.random() < edge_prob:
                w = floor + weight_spread * float(gen.random())
                g.add_edge(i, j, weight=w)

    return g


def two_cluster_graph(
    k: int, floor: float, dense_weight: float, gen: np.random.Generator
) -> nx.Graph:
    """
    Two dense item-clusters of size ``k`` joined to a common medium and to each
    other by a single floor-weight bridge (the construction used in the paper to
    show identity is region-valued: the cheapest nontrivial cut separates a whole
    cluster, so the minimising side is not a singleton).
    """
    g = nx.Graph()
    g.add_node(MEDIUM)
    A = [("A", i) for i in range(k)]
    B = [("B", i) for i in range(k)]
    g.add_nodes_from(A + B)

    # Dense intra-cluster contacts (expensive to cut apart).
    for cluster in (A, B):
        for a in range(k):
            for b in range(a + 1, k):
                g.add_edge(cluster[a], cluster[b], weight=dense_weight)

    # Each item lightly individuated against the medium (floor).
    for v in A + B:
        g.add_edge(MEDIUM, v, weight=floor + 0.01 * float(gen.random()))

    # A single floor-weight bridge between the two clusters.
    g.add_edge(A[0], B[0], weight=floor)
    return g


def min_cut_against_medium(g: nx.Graph, v) -> tuple[float, set]:
    """
    Exact v--medium minimum cut (Def. 3.5), computed by max-flow
    (Ford-Fulkerson). Returns (cut_value, source_side_vertex_set) where the
    source side contains ``v`` and excludes the medium.
    """
    cut_value, (side_v, _side_m) = nx.minimum_cut(
        g, v, MEDIUM, capacity="weight"
    )
    return float(cut_value), set(side_v)


def realised_floor(g: nx.Graph) -> float:
    """The least item--medium separation cost over all items (the realised floor)."""
    items = [x for x in g.nodes if x != MEDIUM]
    return min(min_cut_against_medium(g, v)[0] for v in items)


# ---------------------------------------------------------------------------
# Handles, signatures, content sets (Sec. 5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signature:
    """An item signature: an ordered tuple of handles (Def. 5.1)."""

    handles: tuple

    def content_multiset(self) -> dict:
        """The content set: handle -> multiplicity, order forgotten."""
        out: dict = {}
        for h in self.handles:
            out[h] = out.get(h, 0) + 1
        return out

    def pattern(self) -> tuple:
        """
        The induced sequence pattern (Thm. 5.4(i)): each position replaced by
        the index of first appearance of its handle. Invariant under any
        equality-preserving re-encoding of handle labels.
        """
        first: dict = {}
        pat = []
        for h in self.handles:
            if h not in first:
                first[h] = len(first)
            pat.append(first[h])
        return tuple(pat)

    def __len__(self) -> int:
        return len(self.handles)


def reencode(sig: Signature, mapping: dict) -> Signature:
    """
    Apply a handle re-encoding (Def. 5.2): a bijection on labels that preserves
    which handles are equal. ``mapping`` must be injective on the handles that
    appear.
    """
    return Signature(tuple(mapping[h] for h in sig.handles))


def random_bijection(labels, gen: np.random.Generator) -> dict:
    """A random bijection (relabelling) over the given label set."""
    labels = list(labels)
    perm = list(gen.permutation(len(labels)))
    # Map to a disjoint fresh label space to make it a genuine re-encoding.
    return {labels[i]: f"r{perm[i]}" for i in range(len(labels))}


# ---------------------------------------------------------------------------
# Sequence alignment (Sec. 6)
# ---------------------------------------------------------------------------

def align_cost(
    a: Signature | tuple,
    b: Signature | tuple,
    sub: float = 1.0,
    gap: float = 1.0,
    mode: str = "global",
) -> float:
    """
    Minimum-cost alignment between two handle sequences (Def. 6.1).

    match cost = 0, substitution cost = ``sub``, indel (gap) cost = ``gap``.
    ``mode='global'`` = Needleman-Wunsch; ``mode='local'`` = best-scoring
    contiguous sub-alignment (Smith-Waterman style; returns the minimum-cost
    embedding of the shorter local region -- reported as 0 when a perfect
    contiguous submatch exists).
    """
    x = a.handles if isinstance(a, Signature) else tuple(a)
    y = b.handles if isinstance(b, Signature) else tuple(b)
    m, n = len(x), len(y)

    if mode == "global":
        d = np.zeros((m + 1, n + 1))
        d[:, 0] = np.arange(m + 1) * gap
        d[0, :] = np.arange(n + 1) * gap
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c = 0.0 if x[i - 1] == y[j - 1] else sub
                d[i, j] = min(
                    d[i - 1, j - 1] + c,
                    d[i - 1, j] + gap,
                    d[i, j - 1] + gap,
                )
        return float(d[m, n])

    if mode == "local":
        # Cost-minimising local alignment: we score matches as -1 and
        # substitutions/gaps as positive, then report the best (lowest) window
        # cost, floored at 0. Used only for containment-style queries.
        best = 0.0
        d = np.zeros((m + 1, n + 1))
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c = -1.0 if x[i - 1] == y[j - 1] else sub
                d[i, j] = min(
                    0.0,
                    d[i - 1, j - 1] + c,
                    d[i - 1, j] + gap,
                    d[i, j - 1] + gap,
                )
                best = min(best, d[i, j])
        return float(best)

    raise ValueError(f"unknown mode {mode!r}")


def align_score(a: Signature, b: Signature, sub=1.0, gap=1.0, mode="global") -> float:
    """Normalised alignment score in [0,1] (Def. 6.2): cost / (gap*max(m,n))."""
    x = a.handles if isinstance(a, Signature) else tuple(a)
    y = b.handles if isinstance(b, Signature) else tuple(b)
    omega = gap * max(len(x), len(y), 1)
    return align_cost(a, b, sub=sub, gap=gap, mode=mode) / omega


def align_traceback(a: Signature, b: Signature, sub=1.0, gap=1.0):
    """
    Global alignment with traceback, returning (cost, matched_positions,
    residual_positions) where matched = positions aligned at cost 0 and residual
    = substituted/indel positions (Def. 6.9). Positions are indices into ``a``.
    """
    x, y = a.handles, b.handles
    m, n = len(x), len(y)
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1) * gap
    d[0, :] = np.arange(n + 1) * gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0.0 if x[i - 1] == y[j - 1] else sub
            d[i, j] = min(
                d[i - 1, j - 1] + c, d[i - 1, j] + gap, d[i, j - 1] + gap
            )
    # Traceback
    i, j = m, n
    matched, residual = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0.0 if x[i - 1] == y[j - 1] else sub
            if np.isclose(d[i, j], d[i - 1, j - 1] + c):
                if c == 0.0:
                    matched.append(i - 1)
                else:
                    residual.append(i - 1)  # substitution
                i, j = i - 1, j - 1
                continue
        if i > 0 and np.isclose(d[i, j], d[i - 1, j] + gap):
            residual.append(i - 1)  # deletion of a query position
            i -= 1
            continue
        # insertion of a target position (no query index consumed)
        j -= 1
    matched.sort()
    residual.sort()
    return float(d[m, n]), matched, residual


# ---------------------------------------------------------------------------
# Least-sufficient identifier and the name/acoustic cost model (Sec. 4.3)
# ---------------------------------------------------------------------------

@dataclass
class Constituent:
    """
    A stream constituent with a (possibly missing) symbolic name handle and an
    always-available acoustic handle. Two constituents may share a name yet
    differ acoustically (a false-friend / version collision).
    """

    name: str | None      # symbolic handle, or None if unlabelled ("ID")
    acoustic: str         # acoustic handle (always resolvable, but expensive)


def least_sufficient_identifier(
    c: Constituent,
    ambiguity: list[Constituent],
    c_sym: float,
    c_ac: float,
) -> tuple[str, str, float]:
    """
    Return (handle, kind, cost) for the constituent's least sufficient
    identifier against the given ambiguity set (Def. 4.6, Thm. 4.7, Cor. 4.8).

    Name is chosen when it is available AND separates c from every alternative
    in the ambiguity set (no alternative shares the name). Otherwise the
    acoustic handle is acquired. kind is 'name' or 'acoustic'.
    """
    name_sufficient = c.name is not None and all(
        other.name != c.name for other in ambiguity
    )
    if name_sufficient and c_sym <= c_ac:
        return c.name, "name", c_sym
    return c.acoustic, "acoustic", c_ac


# ---------------------------------------------------------------------------
# JSON result output
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_results(name: str, payload: dict) -> str:
    """Write one experiment's results to results/<name>.json and return the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
    return path
