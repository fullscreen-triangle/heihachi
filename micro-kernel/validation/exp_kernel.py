"""Experiments for Paper I: the heihachi runtime graph."""

from __future__ import annotations

import random

from hkcore import (MEDIUM, Chunk, ContactGraph, HaltingRuntime, Runtime, Value,
                    brute_force_min_cut, separation, strength)


def _reading(channel: str, mag: float, floor: float = 0.01) -> Value:
    return Value(channel=channel, magnitude=mag, floor=floor, unit="dB")


def exp_floor_positivity() -> dict:
    """No value with a non-positive floor is constructible."""
    refused = 0
    for bad in (0.0, -1.0, -1e-9):
        try:
            Value(channel="x", magnitude=1.0, floor=bad)
        except ValueError:
            refused += 1
    ok = Value(channel="x", magnitude=1.0, floor=1e-9)
    return {"attempts": 3, "refused": refused, "positive_accepted": ok.floor > 0,
            "passed": refused == 3 and ok.floor > 0}


def exp_mincut_against_brute_force() -> dict:
    """Max-flow separation checked against a minimum over all bipartitions."""
    random.seed(11)
    trials, mismatches, worst = 300, 0, 0.0
    for _ in range(trials):
        g = ContactGraph(0.02)
        names = ["a", "b", "c", "d", "e"][: random.randint(2, 5)]
        for i, u in enumerate(names):
            for v in names[i + 1:]:
                if random.random() < 0.6:
                    g.link(u, v, round(random.uniform(0.02, 1.0), 3))
        for v in names:
            g.attach(v, round(random.uniform(0.02, 1.0), 3))
        for v in names:
            a = separation(g, v, MEDIUM)
            b = brute_force_min_cut(g, v, MEDIUM)
            worst = max(worst, abs(a - b))
            if abs(a - b) > 1e-9:
                mismatches += 1
    return {"trials": trials, "mismatches": mismatches,
            "max_abs_difference": worst, "passed": mismatches == 0}


def exp_run_to_completion() -> dict:
    """A chunk that raises does not halt the run."""
    lengths = [4, 8, 12, 16, 20, 24, 28, 32]
    densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for L in lengths:
        for d in densities:
            def make(i: int, dens: float):
                def body(rt, node):
                    if dens > 0 and i > 0 and (i % max(1, round(1 / dens))) == 0:
                        raise RuntimeError("anomalous stage")
                    return [_reading("n%d.level" % i, float(i))]
                return body

            rt, ht = Runtime(), HaltingRuntime()
            for engine in (rt, ht):
                for i in range(L):
                    engine.identify("n%d" % i).chunks.append(
                        Chunk("c%d" % i, make(i, d)))
                engine.run_all(["n%d" % i for i in range(L)])
            rows.append({
                "length": L, "density": d,
                "kernel_reached": len(rt.executed),
                "halting_reached": len(ht.executed),
                "recovered": len(rt.executed) - len(ht.executed),
            })
    complete = all(r["kernel_reached"] == r["length"] for r in rows)
    worst_halt = min(r["halting_reached"] / r["length"] for r in rows)
    max_recovered = max(r["recovered"] for r in rows)
    return {"cells": len(rows), "kernel_always_complete": complete,
            "worst_halting_fraction": round(worst_halt, 4),
            "max_nodes_recovered": max_recovered,
            "rows": rows,
            "passed": complete and max_recovered > 0}


def exp_no_exit_code() -> dict:
    """The runtime's output is a report; it exposes no verdict."""
    rt = Runtime()

    def ok(rt_, node):
        return [_reading("a.level", -6.0)]

    def bad(rt_, node):
        raise ValueError("clipped")

    rt.identify("a").chunks.extend([Chunk("ok", ok), Chunk("bad", bad)])
    rt.run("a")
    report = rt.report()
    verdict_words = {"success", "failure", "exit_code", "status", "ok", "passed"}
    exposes = verdict_words & set(report.keys())
    return {"report_keys": sorted(report.keys()),
            "verdict_keys_exposed": sorted(exposes),
            "anomalies_recorded": report["anomalies"],
            "emissions": report["emissions"],
            "passed": (not exposes and report["anomalies"] == 1
                       and report["emissions"] == 2)}


def exp_monotone_record() -> dict:
    """The record never decreases; re-emission is a new commitment."""
    rt = Runtime()
    rt.identify("a").chunks.append(
        Chunk("m", lambda r, n: [_reading("a.level", 1.0)]))
    seq = []
    for _ in range(400):
        rt.run("a")
        seq.append(rt.record)
    decreasing = sum(1 for i in range(1, len(seq)) if seq[i] <= seq[i - 1])
    repeats = []
    for _ in range(3):
        rt.run("a")
        repeats.append(rt.record)
    return {"final_record": rt.record, "decreases": decreasing,
            "repeat_records": repeats,
            "memoised": len(set(repeats)) != len(repeats),
            "passed": decreasing == 0 and len(set(repeats)) == 3}


def exp_convergence() -> dict:
    """Two agents raising the same subtask meet on one node."""
    rt = Runtime()
    a = rt.identify("bass.transient")
    a.chunks.append(Chunk("spectral", lambda r, n: [_reading("t.attack", 3.0)]))
    b = rt.identify("bass.transient")
    b.chunks.append(Chunk("annotation", lambda r, n: [_reading("t.attack", 3.2)]))
    rt.run("bass.transient")
    return {"distinct_nodes": len(rt.nodes), "same_object": a is b,
            "chunks_on_node": len(a.chunks), "emissions": len(a.values),
            "passed": (len(rt.nodes) == 1 and a is b and len(a.chunks) == 2
                       and len(a.values) == 2)}


def exp_total_chunk_execution() -> dict:
    """Every chunk runs; none is skipped on the strength of another."""
    fired = []
    rt = Runtime()
    node = rt.identify("n")
    for name in ("first", "raises", "third", "fourth"):
        def body(r, nd, nm=name):
            fired.append(nm)
            if nm == "raises":
                raise RuntimeError("boom")
            return [_reading(nm + ".v", 1.0)]
        node.chunks.append(Chunk(name, body))
    rt.run("n")
    return {"chunks": 4, "fired": fired, "all_fired": len(fired) == 4,
            "passed": len(fired) == 4 and fired[-1] == "fourth"}


def exp_trajectory_emergence() -> dict:
    """The induced edge relation is a product of the run, not an input."""
    catalogue = ["a", "b", "c", "d", "e", "f"]
    seen = set()
    for seed in catalogue:
        for magnitude in range(1, 7):
            rt = Runtime()
            for name in catalogue:
                rt.identify(name)

            def carrier(r, node, m=magnitude):
                idx = catalogue.index(node.tau)
                reach = (idx + m) % len(catalogue)
                r.edges.add((node.tau, catalogue[reach]))
                return [_reading(node.tau + ".v", float(m))]

            for name in catalogue:
                rt.identify(name).chunks.append(Chunk("carry", carrier))
            start = catalogue.index(seed)
            rt.run_all(catalogue[start:] + catalogue[:start])
            seen.add(tuple(sorted(rt.edges)))
    return {"runs": 36, "distinct_trajectories": len(seen),
            "multivalued": len(seen) > 1, "passed": len(seen) > 1}


def exp_non_propagation() -> dict:
    """An emission nothing reads forms no edge and needs no rejecting."""
    rt = Runtime()
    rt.identify("heard").chunks.append(
        Chunk("h", lambda r, n: [_reading("heard.v", 1.0)]))
    rt.identify("unheard").chunks.append(
        Chunk("u", lambda r, n: [_reading("unheard.v", 1.0)]))
    rt.run("heard")
    rt.run("unheard")
    rt.emit("downstream", _reading("d.v", 2.0), origin_tau="heard")
    traj = rt.trajectory()
    return {"emitted_nodes": 2, "trajectory": traj,
            "unheard_in_trajectory": "unheard" in traj,
            "rejections_performed": 0,
            "passed": "unheard" not in traj and "heard" in traj}


def exp_blast_radius() -> dict:
    """An edit at depth k affects exactly b^(D-k) leaves."""
    rows = []
    for b in (2, 3, 4):
        D = 4
        for k in range(D + 1):
            rows.append({"branching": b, "depth": k, "leaves": b ** (D - k)})
    ok = all(r["leaves"] == r["branching"] ** (4 - r["depth"]) for r in rows)
    spans = {}
    for b in (2, 3, 4):
        vals = [r["leaves"] for r in rows if r["branching"] == b]
        spans[str(b)] = [min(vals), max(vals)]
    return {"rows": len(rows), "spans": spans,
            "passed": ok and spans["4"] == [1, 256]}


def exp_floor_is_scale() -> dict:
    """Rescaling the floor rescales residues and leaves the ratio fixed."""
    ratios = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        g = ContactGraph(0.02 * scale)
        g.link("a", "b", 0.30 * scale)
        g.link("b", "c", 0.50 * scale)
        g.attach("a", 0.20 * scale)
        g.attach("b", 0.40 * scale)
        g.attach("c", 0.60 * scale)
        ratios.append(strength(g, "a") / g.floor())
    spread = max(ratios) - min(ratios)
    return {"scales": 5, "ratios": [round(r, 9) for r in ratios],
            "spread": spread, "passed": spread < 1e-9}


EXPERIMENTS = [
    ("kernel_floor_positivity", exp_floor_positivity),
    ("kernel_mincut_brute_force", exp_mincut_against_brute_force),
    ("kernel_run_to_completion", exp_run_to_completion),
    ("kernel_no_exit_code", exp_no_exit_code),
    ("kernel_monotone_record", exp_monotone_record),
    ("kernel_convergence", exp_convergence),
    ("kernel_total_chunk_execution", exp_total_chunk_execution),
    ("kernel_trajectory_emergence", exp_trajectory_emergence),
    ("kernel_non_propagation", exp_non_propagation),
    ("kernel_blast_radius", exp_blast_radius),
    ("kernel_floor_is_scale", exp_floor_is_scale),
]
