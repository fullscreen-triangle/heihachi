"""
hkcore -- reference implementation for the heihachi micro-kernel papers.

Three subjects, one module:

  * the runtime graph   (Paper I)   nodes, chunks, values, execution, record
  * mishima             (Paper II)  probing, ladders, closure, decline
  * sangoma             (Paper III) separators, targets, reachability

Standard library only. Every routine here is used by at least one
experiment in exp_kernel.py, exp_mishima.py or exp_sangoma.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ─────────────────────────────────────────────────────────────────────
# Contact graphs
# ─────────────────────────────────────────────────────────────────────

MEDIUM = "@medium"


class ContactGraph:
    """A finite weighted graph with a distinguished medium vertex.

    Every item is adjacent to the medium, so every cut separating an
    item from the medium is non-empty and its weight is bounded below
    by the smallest edge weight in the graph.
    """

    def __init__(self, floor: float = 0.02) -> None:
        if floor <= 0.0:
            raise ValueError("floor must be strictly positive")
        self.declared_floor = float(floor)
        self.w: dict[tuple[str, str], float] = {}
        self.items: set[str] = set()

    # -- construction --------------------------------------------------

    @staticmethod
    def _key(u: str, v: str) -> tuple[str, str]:
        return (u, v) if u <= v else (v, u)

    def link(self, u: str, v: str, weight: float) -> None:
        if weight < self.declared_floor:
            raise ValueError(
                f"weight {weight} below declared floor {self.declared_floor}"
            )
        self.w[self._key(u, v)] = float(weight)
        for x in (u, v):
            if x != MEDIUM:
                self.items.add(x)

    def attach(self, v: str, weight: float) -> None:
        """Join an item to the medium."""
        self.link(v, MEDIUM, weight)

    def ensure_medium(self, weight: float | None = None) -> None:
        """Give every item a medium edge if it lacks one."""
        w = self.declared_floor if weight is None else weight
        for v in sorted(self.items):
            if self._key(v, MEDIUM) not in self.w:
                self.link(v, MEDIUM, w)

    # -- observables ---------------------------------------------------

    def floor(self) -> float:
        """The realised floor: the smallest weight actually present."""
        return min(self.w.values()) if self.w else self.declared_floor

    def omega(self) -> float:
        return sum(self.w.values())

    def vertices(self) -> list[str]:
        out: set[str] = set()
        for u, v in self.w:
            out.add(u)
            out.add(v)
        return sorted(out)

    def adjacency(self) -> dict[str, dict[str, float]]:
        adj: dict[str, dict[str, float]] = {x: {} for x in self.vertices()}
        for (u, v), wt in self.w.items():
            adj[u][v] = adj[u].get(v, 0.0) + wt
            adj[v][u] = adj[v].get(u, 0.0) + wt
        return adj


# ─────────────────────────────────────────────────────────────────────
# Maximum flow / minimum cut
# ─────────────────────────────────────────────────────────────────────

def max_flow(graph: ContactGraph, source: str, sink: str) -> tuple[float, set[str]]:
    """Edmonds-Karp on the undirected contact graph.

    Returns the flow value and the source side of a minimum cut. On an
    undirected graph each edge is modelled as two opposed arcs of equal
    capacity, which is the standard reduction.
    """
    if source == sink:
        raise ValueError("source and sink coincide")

    cap: dict[str, dict[str, float]] = {}
    for x in graph.vertices():
        cap.setdefault(x, {})
    for (u, v), wt in graph.w.items():
        cap.setdefault(u, {})
        cap.setdefault(v, {})
        cap[u][v] = cap[u].get(v, 0.0) + wt
        cap[v][u] = cap[v].get(u, 0.0) + wt

    if source not in cap or sink not in cap:
        return 0.0, {source}

    flow = 0.0
    while True:
        # breadth-first search for an augmenting path
        parent: dict[str, str] = {source: source}
        queue = [source]
        while queue and sink not in parent:
            nxt: list[str] = []
            for u in queue:
                for v, c in cap[u].items():
                    if c > 1e-12 and v not in parent:
                        parent[v] = u
                        nxt.append(v)
            queue = nxt
        if sink not in parent:
            break

        # bottleneck
        bottleneck = math.inf
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, cap[u][v])
            v = u
        # augment
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= bottleneck
            cap[v][u] = cap[v].get(u, 0.0) + bottleneck
            v = u
        flow += bottleneck

    reachable = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v, c in cap[u].items():
            if c > 1e-12 and v not in reachable:
                reachable.add(v)
                stack.append(v)
    return flow, reachable


def separation(graph: ContactGraph, u: str, v: str) -> float:
    """Minimum weight of a cut separating u from v."""
    value, _ = max_flow(graph, u, v)
    return value


def strength(graph: ContactGraph, v: str) -> float:
    """Separation cost of an item against the medium."""
    return separation(graph, v, MEDIUM)


def realised_floor(graph: ContactGraph) -> float:
    """The smallest separation cost over all items."""
    items = [x for x in graph.vertices() if x != MEDIUM]
    if not items:
        return graph.declared_floor
    return min(strength(graph, v) for v in items)


def alignment(graph: ContactGraph, x: str, target: str) -> float:
    """Normalised separation of x from the target, in (0, 1]."""
    om = graph.omega()
    if om <= 0.0:
        return 1.0
    if x == target:
        return graph.floor() / om
    return separation(graph, x, target) / om


def brute_force_min_cut(graph: ContactGraph, source: str, sink: str) -> float:
    """Minimum over every bipartition. Exponential; small graphs only.

    Used to check max_flow against an independent computation rather
    than against itself.
    """
    verts = [x for x in graph.vertices() if x not in (source, sink)]
    n = len(verts)
    if n > 16:
        raise ValueError("brute force refused above 16 free vertices")
    best = math.inf
    for mask in range(1 << n):
        side = {source}
        for i, x in enumerate(verts):
            if mask & (1 << i):
                side.add(x)
        total = 0.0
        for (u, v), wt in graph.w.items():
            if (u in side) != (v in side):
                total += wt
        best = min(best, total)
    return best


# ─────────────────────────────────────────────────────────────────────
# The runtime graph  (Paper I)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Value:
    """A datum attached to a node.

    `floor` is never optional and never zero: a value with no stated
    resolution would be a claim of exact measurement.
    """
    channel: str
    magnitude: Any
    floor: float
    unit: str = ""
    kind: str = "reading"          # reading | anomaly | finding | decline
    record: int = 0
    origin: str = ""
    reads: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.floor <= 0.0:
            raise ValueError("a value with a non-positive floor is not writable")


@dataclass
class Node:
    """A subtask identity, a bag of chunks, and the values that accreted."""
    tau: str
    chunks: list["Chunk"] = field(default_factory=list)
    values: list[Value] = field(default_factory=list)


class Chunk:
    """An executable realisation of a subtask.

    The kernel requires only that it can be executed and that it
    returns a finite list of emissions. It is never parsed, compared or
    selected among.
    """

    def __init__(self, name: str, body: Callable[["Runtime", Node], list[Value]],
                 dsl: str = "") -> None:
        self.name = name
        self.body = body
        self.dsl = dsl

    def execute(self, rt: "Runtime", node: Node) -> list[Value]:
        return self.body(rt, node)


class Runtime:
    """The kernel.

    Its whole semantic content is `run`: execute every chunk in a
    node's bag and increment the record once per emission. It does not
    select among chunks, does not order the graph, and does not inspect
    the values that result.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.record = 0
        self.log: list[Value] = []
        self.edges: set[tuple[str, str]] = set()
        self.executed: list[str] = []

    # -- graph vocabulary: identify, read, transform, emit -------------

    def identify(self, tau: str) -> Node:
        """Locate the node bearing a subtask identity, raising it if new.

        Convergence rather than creation: raising a subtask that exists
        returns the existing node, so two agents that decompose their
        problems and arrive at the same subtask meet on one node.
        """
        if tau not in self.nodes:
            self.nodes[tau] = Node(tau)
        return self.nodes[tau]

    def read(self, tau: str) -> list[Value]:
        return list(self.identify(tau).values)

    def emit(self, tau: str, value: Value, origin_tau: str | None = None) -> None:
        self.record += 1
        value.record = self.record
        node = self.identify(tau)
        node.values.append(value)
        self.log.append(value)
        if origin_tau is not None and origin_tau != tau:
            self.edges.add((origin_tau, tau))

    # -- execution -----------------------------------------------------

    def run(self, tau: str) -> None:
        """Execute every chunk in the node's bag.

        No chunk is skipped on the strength of what another produced,
        and a chunk that raises contributes an anomaly value rather
        than halting the run.
        """
        node = self.identify(tau)
        self.executed.append(tau)
        for chunk in node.chunks:
            try:
                emissions = chunk.execute(self, node)
            except Exception as exc:                      # noqa: BLE001
                emissions = [Value(
                    channel=f"{tau}.anomaly",
                    magnitude=str(exc),
                    floor=max(1e-9, 1e-6),
                    kind="anomaly",
                    origin=chunk.name,
                )]
            for value in emissions:
                self.emit(tau, value, origin_tau=tau)

    def run_all(self, order: Iterable[str]) -> None:
        for tau in order:
            self.run(tau)

    # -- provenance ----------------------------------------------------

    def trajectory(self) -> list[str]:
        """The nodes that carried propagated information."""
        seen: set[str] = set()
        for u, v in self.edges:
            seen.add(u)
            seen.add(v)
        return sorted(seen)

    def anomalies(self) -> list[Value]:
        return [v for v in self.log if v.kind == "anomaly"]

    def report(self) -> dict[str, Any]:
        """The runtime's output. Not an exit code."""
        return {
            "record": self.record,
            "nodes_executed": len(self.executed),
            "emissions": len(self.log),
            "anomalies": len(self.anomalies()),
            "induced_edges": len(self.edges),
        }


class HaltingRuntime(Runtime):
    """A conventional runtime, for contrast only.

    It inspects emission content and halts on an anomaly. Used by the
    run-to-completion experiment to measure the work such a runtime
    abandons.
    """

    def __init__(self) -> None:
        super().__init__()
        self.halted = False

    def run(self, tau: str) -> None:
        if self.halted:
            return
        node = self.identify(tau)
        self.executed.append(tau)
        for chunk in node.chunks:
            try:
                emissions = chunk.execute(self, node)
            except Exception:                              # noqa: BLE001
                self.halted = True
                return
            for value in emissions:
                self.emit(tau, value, origin_tau=tau)


# ─────────────────────────────────────────────────────────────────────
# Probes, ladders, closure  (Paper II: mishima)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    """A move that closes some of the distance to a target.

    Nothing below inspects a probe beyond its power and its cost, so a
    spectral measurement, a stored annotation and a model inference are
    the same kind of object here.
    """
    name: str
    power: float                      # kappa in [0, 1]
    cost: float = 1.0
    lands: str = ""                   # class reached, for closure tests

    def __post_init__(self) -> None:
        if not 0.0 <= self.power <= 1.0:
            raise ValueError("probing power must lie in [0, 1]")


def composite_power(powers: Iterable[float]) -> float:
    """Power of probes composed in series: 1 - prod(1 - k_i)."""
    remaining = 1.0
    for k in powers:
        remaining *= (1.0 - k)
    return 1.0 - remaining


def rungs_required(target: float, best: float) -> int:
    """Depth needed to reach a target composite with the strongest rung."""
    if not 0.0 < best < 1.0:
        raise ValueError("per-rung power must lie strictly in (0, 1)")
    if target >= 1.0:
        return -1                     # unreachable at any finite depth
    if target <= 0.0:
        return 0
    return math.ceil(math.log(1.0 - target) / math.log(1.0 - best))


def refinement_cost(thickness: float, gain: float = 1.0) -> float:
    """Cost of realising a separator of a given thickness.

    Diverges as thickness tends to zero, which is what forces nesting.
    """
    if thickness <= 0.0:
        return math.inf
    return gain / thickness


def marginal_return(thickness: float, extent: float = 1.0,
                    gain: float = 1.0, step: float = 1e-3) -> float:
    """Return per unit cost from thinning a separator by one step."""
    if thickness - step <= 0.0:
        return 0.0
    d_return = extent * (1.0 / (thickness - step) - 1.0 / thickness) * 0.0
    # the certified content is bounded by the extent of the space, so
    # thinning buys at most `extent` in total, with diminishing steps
    d_return = extent * (step / max(thickness, step))
    d_cost = refinement_cost(thickness - step, gain) - refinement_cost(thickness, gain)
    return d_return / d_cost if d_cost > 0.0 else math.inf


def water_fill(gains: list[Callable[[float], float]], budget: float,
               iterations: int = 200) -> list[float]:
    """Allocate a budget by equalising marginal gain at a single price.

    Bisection on the scalar price. `gains` are derivatives of concave
    yield functions, each non-increasing.
    """
    if budget <= 0.0:
        return [0.0] * len(gains)

    def spend(price: float) -> list[float]:
        out = []
        for g in gains:
            lo, hi = 0.0, budget
            if g(0.0) <= price:
                out.append(0.0)
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if g(mid) > price:
                    lo = mid
                else:
                    hi = mid
            out.append(0.5 * (lo + hi))
        return out

    lo, hi = 0.0, max(g(0.0) for g in gains) if gains else 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if sum(spend(mid)) > budget:
            lo = mid
        else:
            hi = mid
    return spend(0.5 * (lo + hi))


def knapsack_priority(floor_i: float, cost_i: float, omega: float) -> float:
    """Admission priority for a probe that is all-or-nothing."""
    if cost_i <= 0.0 or omega <= floor_i:
        return math.inf
    return math.log(omega / (omega - floor_i)) / cost_i


@dataclass
class Determination:
    """The outcome of a seek: convergent closure, or honest decline."""
    status: str                        # "converged" | "declined"
    classes: list[str]
    probes_invoked: list[str]
    record_before: int = 0
    record_after: int = 0

    @property
    def declined(self) -> bool:
        return self.status == "declined"


def seek_to_closure(seed: str, registry: list[Probe],
                    threshold: float = 0.99) -> tuple[Determination, int]:
    """Invoke probes until no remaining probe reaches a new class.

    Returns the determination and the number of probes that had been
    invoked at the moment a fixed confidence threshold was first met,
    which is what the closure-versus-threshold experiment compares
    against.
    """
    reached: list[str] = []
    invoked: list[str] = []
    threshold_at = -1
    running = 0.0

    for probe in registry:
        invoked.append(probe.name)
        running = composite_power([running, probe.power])
        if probe.lands and probe.lands not in reached:
            reached.append(probe.lands)
        if threshold_at < 0 and running >= threshold:
            threshold_at = len(invoked)
        # closure: does any uninvoked probe still reach a new class?
        remaining = registry[len(invoked):]
        if all((p.lands in reached) or not p.lands for p in remaining):
            break

    status = "converged" if len(reached) <= 1 else "declined"
    return Determination(status, reached, invoked), threshold_at


def supports_triangle(support: dict[str, set[str]]) -> bool:
    """Whether a support relation contains a cycle of length >= 3."""
    colour: dict[str, int] = {}

    def visit(u: str, stack: list[str]) -> bool:
        colour[u] = 1
        for v in support.get(u, ()):
            if colour.get(v, 0) == 0:
                if visit(v, stack + [u]):
                    return True
            elif colour.get(v) == 1:
                cycle_len = len(stack) + 1 - (stack.index(v) if v in stack else 0)
                if cycle_len >= 3:
                    return True
        colour[u] = 2
        return False

    return any(visit(u, []) for u in support if colour.get(u, 0) == 0)


# ─────────────────────────────────────────────────────────────────────
# Separators and construction  (Paper III: sangoma)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    """One processing stage declared as a separator.

    `thickness` is the resolution at which the stage's effect can be
    told from its absence; `power` is how much of the distance to a
    declared target it closes.
    """
    name: str
    thickness: float
    power: float
    latency: int = 0                   # samples


def detectability(stage_thickness: float, assembly_extent: float) -> float:
    """tau = separator thickness / extent. Below 1 the assembly is audible."""
    if assembly_extent <= 0.0:
        return math.inf
    return stage_thickness / assembly_extent


def chain_thickness(stages: Iterable[Stage]) -> float:
    return sum(s.thickness for s in stages)


def reachable(target_power: float, stages: list[Stage]) -> tuple[bool, int]:
    """Whether a target is reachable, and the depth required if not."""
    attainable = composite_power([s.power for s in stages])
    if attainable >= target_power:
        return True, len(stages)
    best = max((s.power for s in stages), default=0.0)
    if best <= 0.0:
        return False, -1
    return False, rungs_required(target_power, best)


def headroom_after(stages: Iterable[Stage], gains_db: Iterable[float],
                   ceiling_db: float = -1.0) -> tuple[float, bool]:
    """Peak level after a chain of declared gains, and whether it clears.

    A physical rule rather than a matter of taste: a chain whose summed
    gain drives the signal past the declared ceiling is refused.
    """
    peak = sum(float(g) for g in gains_db)
    return peak, peak <= ceiling_db


def latency_total(stages: Iterable[Stage]) -> int:
    return sum(s.latency for s in stages)


def floor_negotiation(program_floor: float, backend_resolution: float,
                      reference_resolution: float = 0.0) -> tuple[bool, str]:
    """Whether a program may run on a back end at all.

    If the machinery's error exceeds the program's declared floor, a
    reported distinction may be an artefact, and the program is refused
    with both quantities named.
    """
    total = backend_resolution + reference_resolution
    if total <= program_floor:
        return True, ""
    return False, (
        f"floor {program_floor:g} finer than back end can resolve "
        f"({total:g}); refused before execution"
    )


# ─────────────────────────────────────────────────────────────────────
# Shared lexer
# ─────────────────────────────────────────────────────────────────────

class LexError(Exception):
    pass


@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int
    value: float | None = None
    floor: float | None = None


_NUM = re.compile(r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPS = [">>", ":=", "<=", ">=", "==", "!=", "||", "{", "}", "(", ")",
        "[", "]", ",", ":", ".", "<", ">", "#"]


def lex(src: str, keywords: set[str]) -> list[Token]:
    """One lexer for both languages; they differ only in their keywords.

    A numeric literal may carry a floor by the suffix `value#floor`.
    A non-positive floor suffix is rejected here, not later: there is
    no way to write a literal of zero floor.
    """
    toks: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(src)
    while i < n:
        ch = src[i]
        if ch == "\n":
            line, col, i = line + 1, 1, i + 1
            continue
        if ch in " \t\r":
            i, col = i + 1, col + 1
            continue
        if src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        m = _NUM.match(src, i)
        if m:
            text = m.group(0)
            value = float(text)
            floor = None
            j = m.end()
            if j < n and src[j] == "#":
                m2 = _NUM.match(src, j + 1)
                if not m2:
                    raise LexError(f"line {line}: '#' must be followed by a floor")
                floor = float(m2.group(0))
                if floor <= 0.0:
                    raise LexError(
                        f"line {line}: floor suffix must be strictly positive; "
                        f"a claim of exact measurement is not writable"
                    )
                text += "#" + m2.group(0)
                j = m2.end()
            toks.append(Token("num", text, line, col, value, floor))
            col += j - i
            i = j
            continue
        m = _IDENT.match(src, i)
        if m:
            text = m.group(0)
            kind = "kw" if text in keywords else "ident"
            toks.append(Token(kind, text, line, col))
            col += len(text)
            i = m.end()
            continue
        if ch == '"':
            j = src.find('"', i + 1)
            if j < 0:
                raise LexError(f"line {line}: unterminated string")
            toks.append(Token("str", src[i + 1:j], line, col))
            col += j - i + 1
            i = j + 1
            continue
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Token("op", op, line, col))
                i += len(op)
                col += len(op)
                break
        else:
            raise LexError(f"line {line}: unexpected character {ch!r}")
    toks.append(Token("eof", "", line, col))
    return toks


# ─────────────────────────────────────────────────────────────────────
# mishima front end  (.mma)
# ─────────────────────────────────────────────────────────────────────

MISHIMA_KEYWORDS = {
    "floor", "module", "probe", "commit", "seek", "not", "toward", "via",
    "until", "closure", "converge", "otherwise", "decline", "yield",
    "ladder", "rung", "at", "observe", "assert", "emit", "let", "as", "when",
}


class CheckError(Exception):
    def __init__(self, rule: str, message: str, remedy: str, line: int = 0) -> None:
        super().__init__(message)
        self.rule = rule
        self.message = message
        self.remedy = remedy
        self.line = line

    def __str__(self) -> str:
        return f"[{self.rule}] line {self.line}: {self.message} -- {self.remedy}"


@dataclass
class Seek:
    subject: str
    exclusions: list[str]
    toward: str
    ladder: list[tuple[str, float]]
    admit: str
    otherwise_decline: bool
    result: str
    line: int


@dataclass
class MishimaProgram:
    ambient_floor: float | None
    seeks: list[Seek]
    asserts: list[tuple[str, int]]


class Parser:
    def __init__(self, toks: list[Token]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def accept(self, text: str) -> Token | None:
        if self.peek().text == text:
            return self.next()
        return None

    def expect(self, text: str, rule: str = "syntax") -> Token:
        t = self.peek()
        if t.text != text:
            raise CheckError(rule, f"expected {text!r}, found {t.text or 'end of input'!r}",
                             f"insert {text!r}", t.line)
        return self.next()


def parse_mishima(src: str) -> MishimaProgram:
    toks = lex(src, MISHIMA_KEYWORDS)
    p = Parser(toks)
    ambient: float | None = None
    seeks: list[Seek] = []
    asserts: list[tuple[str, int]] = []

    while p.peek().kind != "eof":
        t = p.peek()
        if t.text == "floor":
            p.next()
            num = p.next()
            if num.kind != "num":
                raise CheckError("syntax", "floor needs a number", "write e.g. floor 0.02", t.line)
            ambient = num.value
            if ambient is not None and ambient <= 0.0:
                raise CheckError("rule:floor-positivity",
                                 "the ambient floor must be strictly positive",
                                 "declare the resolution your instruments deliver", t.line)
        elif t.text == "module":
            p.next(); p.next(); p.expect("{")
            depth = 1
            while depth and p.peek().kind != "eof":
                tk = p.next()
                depth += (tk.text == "{") - (tk.text == "}")
        elif t.text == "seek":
            seeks.append(_parse_seek(p))
        elif t.text == "assert":
            p.next()
            line = t.line
            buf = []
            while p.peek().text not in ("emit",) and p.peek().kind != "eof" \
                    and p.peek().text not in ("seek", "assert", "floor"):
                buf.append(p.next().text)
            if p.accept("emit"):
                p.next()
            asserts.append((" ".join(buf), line))
        else:
            p.next()
    return MishimaProgram(ambient, seeks, asserts)


def _parse_seek(p: Parser) -> Seek:
    line = p.peek().line
    p.expect("seek")
    subject = p.next().text

    # the `not` clause is mandatory, and its absence is a parse error
    if p.peek().text != "not":
        raise CheckError(
            "rule:mandatory-not",
            "a seek without a `not` clause does not specify a region",
            "state what the search excludes, e.g. not { thin, undistorted }",
            p.peek().line,
        )
    p.next()
    p.expect("{")
    exclusions: list[str] = []
    while not p.accept("}"):
        tok = p.next()
        if tok.text != ",":
            exclusions.append(tok.text)
    if not exclusions:
        raise CheckError("rule:mandatory-not",
                         "the exclusion list is empty",
                         "name at least one region the target is not", line)

    p.expect("toward")
    p.expect("{")
    toward_parts: list[str] = []
    while not p.accept("}"):
        toward_parts.append(p.next().text)
    toward = " ".join(toward_parts)

    ladder: list[tuple[str, float]] = []
    if p.accept("via"):
        p.expect("{")
        while not p.accept("}"):
            if p.accept("rung"):
                name = p.next().text
                power = 0.0
                if p.accept("at"):
                    num = p.next()
                    power = float(num.value or 0.0)
                ladder.append((name, power))
            elif p.peek().text == ">>":
                p.next()
            else:
                p.next()

    p.expect("until")
    admit = p.next().text
    otherwise = False
    if p.accept("otherwise"):
        p.expect("decline")
        otherwise = True
    p.expect("yield")
    result = p.next().text
    return Seek(subject, exclusions, toward, ladder, admit, otherwise, result, line)


def check_mishima(prog: MishimaProgram,
                  backend_resolution: float = 0.0) -> list[CheckError]:
    """Static rules. Returns diagnostics; every one carries a remedy."""
    errs: list[str] = []
    out: list[CheckError] = []

    if prog.ambient_floor is None:
        out.append(CheckError("rule:floor-declared",
                              "no ambient floor was declared",
                              "add a `floor` declaration naming your resolution", 1))
    else:
        ok, why = floor_negotiation(prog.ambient_floor, backend_resolution)
        if not ok:
            out.append(CheckError("rule:floor-negotiation", why,
                                  "coarsen the declared floor or improve the back end", 1))

    for s in prog.seeks:
        if s.admit == "closure" and 0 < len(s.ladder) < 3:
            out.append(CheckError(
                "rule:coherence",
                f"ladder of {len(s.ladder)} rung(s) cannot close: "
                f"a support structure of fewer than three is not robust "
                f"to the loss of one",
                "add rungs of differing kind until at least three support the result",
                s.line,
            ))
        if s.ladder:
            attainable = composite_power([k for _, k in s.ladder])
            if attainable < 0.5:
                out.append(CheckError(
                    "rule:saturation",
                    f"ladder attains composite power {attainable:.4f}",
                    f"add rungs; {rungs_required(0.5, max(k for _, k in s.ladder))} "
                    f"of the strongest would reach 0.5",
                    s.line,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────
# sangoma front end  (.sgn)
# ─────────────────────────────────────────────────────────────────────

SANGOMA_KEYWORDS = {
    "floor", "medium", "species", "source", "motion", "construct", "stage",
    "chain", "assemble", "from", "target", "via", "rung", "at", "observe",
    "assert", "emit", "let", "as", "ceiling", "expresses", "couples", "by",
}


@dataclass
class TargetItem:
    name: str
    relation: str
    magnitude: float
    floor: float
    line: int


@dataclass
class Construct:
    name: str
    stages: list[str]
    targets: list[TargetItem]
    ladder: list[tuple[str, float]]
    line: int


@dataclass
class SangomaProgram:
    ambient_floor: float | None
    species: dict[str, dict[str, float]]
    constructs: list[Construct]


def parse_sangoma(src: str) -> SangomaProgram:
    toks = lex(src, SANGOMA_KEYWORDS)
    p = Parser(toks)
    ambient: float | None = None
    species: dict[str, dict[str, float]] = {}
    constructs: list[Construct] = []

    while p.peek().kind != "eof":
        t = p.peek()
        if t.text == "floor":
            p.next()
            num = p.next()
            ambient = num.value
            if ambient is not None and ambient <= 0.0:
                raise CheckError("rule:floor-positivity",
                                 "the ambient floor must be strictly positive",
                                 "declare the resolution your render path delivers", t.line)
        elif t.text in ("species", "medium"):
            p.next()
            name = p.next().text
            fields: dict[str, float] = {}
            p.expect("{")
            depth = 1
            while depth and p.peek().kind != "eof":
                tk = p.next()
                if tk.text == "{":
                    depth += 1
                elif tk.text == "}":
                    depth -= 1
                elif tk.kind == "ident" and p.peek().text == ":":
                    p.next()
                    val = p.next()
                    if val.kind == "num":
                        fields[tk.text] = float(val.value or 0.0)
            species[name] = fields
        elif t.text == "construct":
            constructs.append(_parse_construct(p))
        else:
            p.next()
    return SangomaProgram(ambient, species, constructs)


def _parse_construct(p: Parser) -> Construct:
    line = p.peek().line
    p.expect("construct")
    name = p.next().text
    p.expect("{")
    stages: list[str] = []
    targets: list[TargetItem] = []
    ladder: list[tuple[str, float]] = []

    while not p.accept("}"):
        t = p.peek()
        if t.text == "stage":
            p.next()
            stages.append(p.next().text)
            if p.accept(":="):
                while p.peek().text not in ("stage", "target", "via", "}") \
                        and p.peek().kind != "eof":
                    p.next()
        elif t.text == "chain" or t.text == "assemble":
            p.next()
            while p.peek().text not in ("stage", "target", "via", "}") \
                    and p.peek().kind != "eof":
                p.next()
        elif t.text == "target":
            p.next()
            p.expect("{")
            while not p.accept("}"):
                item = p.next()
                rel = p.next().text
                num = p.next()
                if num.kind != "num":
                    raise CheckError("syntax", "target needs a magnitude",
                                     "write e.g. crest >= 6.0#0.5", item.line)
                if num.floor is None:
                    raise CheckError(
                        "rule:floor-on-target",
                        f"target `{item.text}` states no resolution",
                        "annotate the magnitude, e.g. 6.0#0.5, so the claim is checkable",
                        item.line,
                    )
                targets.append(TargetItem(item.text, rel, float(num.value or 0.0),
                                          float(num.floor), item.line))
        elif t.text == "via":
            p.next()
            p.expect("{")
            while not p.accept("}"):
                if p.accept("rung"):
                    rname = p.next().text
                    power = 0.0
                    if p.accept("at"):
                        power = float(p.next().value or 0.0)
                    ladder.append((rname, power))
                else:
                    p.next()
        else:
            p.next()
    return Construct(name, stages, targets, ladder, line)


def check_sangoma(prog: SangomaProgram, required_power: float = 0.8,
                  backend_resolution: float = 0.0) -> list[CheckError]:
    out: list[CheckError] = []

    if prog.ambient_floor is None:
        out.append(CheckError("rule:floor-declared",
                              "no ambient floor was declared",
                              "add a `floor` declaration", 1))
    else:
        ok, why = floor_negotiation(prog.ambient_floor, backend_resolution)
        if not ok:
            out.append(CheckError("rule:floor-negotiation", why,
                                  "coarsen the floor or render at higher resolution", 1))

    for c in prog.constructs:
        if not c.targets:
            out.append(CheckError("rule:target-required",
                                  f"construct `{c.name}` declares no target",
                                  "state what the finished sound must satisfy", c.line))
        for ti in c.targets:
            if prog.ambient_floor is not None and ti.floor < prog.ambient_floor:
                out.append(CheckError(
                    "rule:over-claim",
                    f"target `{ti.name}` claims resolution {ti.floor:g}, "
                    f"finer than the ambient floor {prog.ambient_floor:g}",
                    "coarsen the target or declare a finer floor your tools support",
                    ti.line,
                ))
        if c.ladder:
            ok_reach, depth = reachable(
                required_power, [Stage(n, 0.01, k) for n, k in c.ladder])
            if not ok_reach:
                attain = composite_power([k for _, k in c.ladder])
                out.append(CheckError(
                    "rule:reachability",
                    f"construct `{c.name}` attains {attain:.4f} of a required "
                    f"{required_power:g}",
                    f"{depth} rungs at the strongest available power would reach it",
                    c.line,
                ))
    return out
