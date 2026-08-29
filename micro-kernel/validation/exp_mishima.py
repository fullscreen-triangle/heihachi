"""Experiments for Paper II: mishima, the propagation language."""

from __future__ import annotations

import math

from hkcore import (CheckError, LexError, Probe, check_mishima, composite_power,
                    knapsack_priority, lex, parse_mishima, rungs_required,
                    seek_to_closure, supports_triangle, water_fill,
                    MISHIMA_KEYWORDS)

GOOD = """-- recall.mma
floor 0.02

seek reese_growl
  not    { thin, undistorted, mono }
  toward { region(that_2019_growl) }
  via    { rung spectral   at 0.45
        >> rung annotation at 0.30
        >> rung model      at 0.55 }
  until  closure
  otherwise decline
  yield  found
"""


def exp_lexer_floor_suffix() -> dict:
    """A literal of zero floor is not writable."""
    refused, accepted = 0, 0
    for src in ("floor 1.0#0", "floor 1.0#0.0", "floor 2.0#-1"):
        try:
            lex(src, MISHIMA_KEYWORDS)
        except LexError:
            refused += 1
    for src in ("floor 0.02", "floor 1.0#0.001"):
        lex(src, MISHIMA_KEYWORDS)
        accepted += 1
    return {"zero_floor_attempts": 3, "refused": refused,
            "positive_accepted": accepted, "passed": refused == 3 and accepted == 2}


def exp_parses_worked_example() -> dict:
    """The worked program parses and its ladder is recovered."""
    prog = parse_mishima(GOOD)
    s = prog.seeks[0]
    power = composite_power([k for _, k in s.ladder])
    return {"floor": prog.ambient_floor, "seeks": len(prog.seeks),
            "exclusions": s.exclusions, "rungs": len(s.ladder),
            "composite_power": round(power, 4),
            "passed": (prog.ambient_floor == 0.02 and len(s.ladder) == 3
                       and len(s.exclusions) == 3
                       and abs(power - 0.8267500) < 1e-6)}


def exp_mandatory_not() -> dict:
    """A seek without a `not` clause is refused at parse time."""
    bad = "floor 0.02\nseek x toward { y } until closure yield f"
    rule = ""
    try:
        parse_mishima(bad)
        refused = False
    except CheckError as exc:
        refused, rule = True, exc.rule
    empty = "floor 0.02\nseek x not { } toward { y } until closure yield f"
    try:
        parse_mishima(empty)
        empty_refused = False
    except CheckError:
        empty_refused = True
    return {"refused": refused, "rule": rule, "empty_list_refused": empty_refused,
            "passed": refused and rule == "rule:mandatory-not" and empty_refused}


def exp_composition_law() -> dict:
    """Composite power equals 1 - prod(1 - k), checked against a direct product."""
    cases = [[0.45, 0.30, 0.55], [0.4, 0.35, 0.55], [0.2] * 5, [0.9, 0.1]]
    worst = 0.0
    rows = []
    for ks in cases:
        got = composite_power(ks)
        want = 1.0
        for k in ks:
            want *= (1.0 - k)
        want = 1.0 - want
        worst = max(worst, abs(got - want))
        rows.append({"rungs": ks, "composite": round(got, 6)})
    return {"cases": rows, "max_abs_difference": worst, "passed": worst < 1e-12}


def exp_diversify() -> dict:
    """A ladder beats repetition of any rung weaker than its mean.

    The naive claim -- that distinct rungs beat every repetition -- is
    false, and the experiment records the counterexample rather than
    suppressing it: repeating the strongest rung of a set beats the
    mixed ladder. What holds is the comparison against the mean, which
    is the form the paper states.
    """
    distinct = [0.45, 0.30, 0.55]
    mixed = composite_power(distinct)
    mean = sum(distinct) / len(distinct)
    repeated = {str(k): composite_power([k] * len(distinct)) for k in distinct}
    beats_weaker = all(mixed > v for k, v in repeated.items()
                       if float(k) < mean)
    strongest = max(distinct)
    counterexample = composite_power([strongest] * len(distinct)) > mixed
    return {"distinct_composite": round(mixed, 4),
            "mean_rung_power": round(mean, 4),
            "repeated": {k: round(v, 4) for k, v in repeated.items()},
            "beats_every_below_mean_repetition": beats_weaker,
            "strongest_repetition_wins": counterexample,
            "passed": beats_weaker and counterexample}


def exp_saturation_dichotomy() -> dict:
    """Residual gap vanishes iff the sum of powers diverges."""
    n = 4000
    harmonic = [1.0 / (i + 2) for i in range(n)]
    geometric = [0.5 ** (i + 1) for i in range(n)]
    gap_h = 1.0 - composite_power(harmonic)
    gap_g = 1.0 - composite_power(geometric)
    # the divergent series drives the gap toward zero and the
    # convergent one does not; at this truncation the separation is
    # three orders of magnitude
    return {"rungs": n,
            "harmonic_residual_gap": gap_h,
            "geometric_residual_gap": gap_g,
            "separation_orders": round(math.log10(gap_g / gap_h), 2),
            "passed": gap_h < 1e-3 and gap_g > 1e-1 and gap_g / gap_h > 100.0}


def exp_depth_closed_form() -> dict:
    """Closed-form depth agrees with incrementing until the target clears."""
    mismatches, rows = 0, 0
    for target in (0.5, 0.7, 0.9, 0.95, 0.99):
        for power in (0.15, 0.25, 0.35, 0.5):
            closed = rungs_required(target, power)
            n, comp = 0, 0.0
            while comp < target and n < 10000:
                comp = composite_power([comp, power])
                n += 1
            rows += 1
            if closed != n:
                mismatches += 1
    return {"combinations": rows, "mismatches": mismatches,
            "passed": mismatches == 0}


def exp_refinement_diverges() -> dict:
    """Thinning one boundary costs without bound; nesting does not."""
    from hkcore import refinement_cost
    thicknesses = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    costs = [refinement_cost(t) for t in thicknesses]
    monotone = all(costs[i] < costs[i + 1] for i in range(len(costs) - 1))
    # a ladder of ordinary boundaries reaching comparable resolution
    ladder = [0.35] * 8
    diverges = monotone and costs[-1] > 0.99e5
    return {"thicknesses": thicknesses, "costs": costs,
            "cost_diverges": diverges,
            "cost_ratio_first_to_last": round(costs[-1] / costs[0], 1),
            "ladder_rungs": len(ladder),
            "ladder_composite": round(composite_power(ladder), 6),
            "ladder_cost": len(ladder),
            "passed": diverges and composite_power(ladder) > 0.9}


def exp_triangle() -> dict:
    """Fewer than three mutually supporting rungs is not robust."""
    chain = {"a": {"b"}, "b": {"c"}, "c": set()}
    pair = {"a": {"b"}, "b": {"a"}}
    triangle = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    return {"chain_robust": supports_triangle(chain),
            "pair_robust": supports_triangle(pair),
            "triangle_robust": supports_triangle(triangle),
            "passed": (not supports_triangle(chain)
                       and not supports_triangle(pair)
                       and supports_triangle(triangle))}


def exp_coherence_diagnostic() -> dict:
    """A ladder of one or two rungs declaring closure is refused."""
    def prog(n: int) -> str:
        rungs = " >> ".join("rung r%d at 0.4" % i for i in range(n))
        return ("floor 0.02\nseek x not { y } toward { z } via { %s } "
                "until closure yield f" % rungs)

    fired = {}
    for n in (1, 2, 3):
        diags = check_mishima(parse_mishima(prog(n)))
        fired[str(n)] = [d.rule for d in diags]
    return {"diagnostics_by_length": fired,
            "passed": ("rule:coherence" in fired["1"]
                       and "rule:coherence" in fired["2"]
                       and "rule:coherence" not in fired["3"])}


def exp_closure_beats_threshold() -> dict:
    """A threshold is met while an uninvoked probe still reaches a new class."""
    registries = []
    for size in (3, 4, 5, 6, 8):
        reg = [Probe("p%d" % i, 0.95, 1.0,
                     lands="A" if i == 0 else "B" if i == size - 1 else "A")
               for i in range(size)]
        registries.append(reg)
    rows = []
    for reg in registries:
        det, threshold_at = seek_to_closure("seed", reg, threshold=0.9)
        rows.append({"registry": len(reg),
                     "threshold_met_after": threshold_at,
                     "closure_after": len(det.probes_invoked),
                     "status": det.status,
                     "classes": det.classes})
    earlier = all(r["threshold_met_after"] < r["closure_after"] for r in rows)
    declined = sum(1 for r in rows if r["status"] == "declined")
    return {"rows": rows, "threshold_always_earlier": earlier,
            "declined": declined, "registries": len(rows),
            "passed": earlier and declined == len(rows)}


def exp_decline_is_typed() -> dict:
    """Contested closure terminates and carries the classes found."""
    reg = [Probe("spectral", 0.6, 1.0, lands="dub"),
           Probe("annotation", 0.5, 1.0, lands="vip"),
           Probe("model", 0.4, 1.0, lands="dub")]
    det, _ = seek_to_closure("seed", reg)
    single = [Probe("a", 0.6, 1.0, lands="one"),
              Probe("b", 0.5, 1.0, lands="one")]
    det2, _ = seek_to_closure("seed", single)
    return {"contested_status": det.status, "classes": det.classes,
            "convergent_status": det2.status, "convergent_classes": det2.classes,
            "passed": (det.status == "declined" and len(det.classes) == 2
                       and det2.status == "converged")}


def exp_water_filling() -> dict:
    """Effort goes to probes whose marginal gain clears one price."""
    gains = [lambda a: 1.0 / (1.0 + a),
             lambda a: 0.6 / (1.0 + a),
             lambda a: 0.2 / (1.0 + a)]
    alloc = water_fill(gains, 3.0)
    served = [i for i, a in enumerate(alloc) if a > 1e-6]
    marginals = [g(a) for g, a in zip(gains, alloc) if a > 1e-6]
    spread = max(marginals) - min(marginals) if marginals else 0.0
    return {"allocation": [round(a, 4) for a in alloc],
            "served": served, "budget": 3.0,
            "total_spent": round(sum(alloc), 4),
            "marginal_spread": spread,
            "passed": abs(sum(alloc) - 3.0) < 1e-3 and spread < 1e-3
                      and 2 not in served}


def exp_knapsack_priority() -> dict:
    """All-or-nothing probes are admitted in decreasing priority."""
    omega = 10.0
    probes = [("cheap_weak", 0.5, 1.0), ("dear_strong", 3.0, 5.0),
              ("cheap_strong", 3.0, 1.0)]
    ranked = sorted(probes,
                    key=lambda p: knapsack_priority(p[1], p[2], omega),
                    reverse=True)
    return {"ranking": [p[0] for p in ranked],
            "priorities": {p[0]: round(knapsack_priority(p[1], p[2], omega), 5)
                           for p in probes},
            "passed": ranked[0][0] == "cheap_strong"}


def exp_probe_kind_invisible() -> dict:
    """Nothing distinguishes probes by kind; only power and cost count."""
    a = Probe("spectral_measurement", 0.45, 1.0, lands="X")
    b = Probe("model_inference", 0.45, 1.0, lands="X")
    det_a, _ = seek_to_closure("s", [a, Probe("q", 0.3, 1.0, lands="X")])
    det_b, _ = seek_to_closure("s", [b, Probe("q", 0.3, 1.0, lands="X")])
    return {"same_power": a.power == b.power,
            "same_outcome": det_a.status == det_b.status
                            and det_a.classes == det_b.classes,
            "passed": a.power == b.power and det_a.classes == det_b.classes}


def exp_floor_negotiation() -> dict:
    """A program finer than its back end is refused before execution."""
    prog = parse_mishima(GOOD)
    coarse = check_mishima(prog, backend_resolution=0.5)
    fine = check_mishima(prog, backend_resolution=0.001)
    return {"coarse_backend_rules": [d.rule for d in coarse],
            "fine_backend_rules": [d.rule for d in fine],
            "remedy_present": all(d.remedy for d in coarse),
            "passed": ("rule:floor-negotiation" in [d.rule for d in coarse]
                       and not fine and all(d.remedy for d in coarse))}


EXPERIMENTS = [
    ("mishima_lexer_floor_suffix", exp_lexer_floor_suffix),
    ("mishima_parses_worked_example", exp_parses_worked_example),
    ("mishima_mandatory_not", exp_mandatory_not),
    ("mishima_composition_law", exp_composition_law),
    ("mishima_diversify", exp_diversify),
    ("mishima_saturation_dichotomy", exp_saturation_dichotomy),
    ("mishima_depth_closed_form", exp_depth_closed_form),
    ("mishima_refinement_diverges", exp_refinement_diverges),
    ("mishima_triangle", exp_triangle),
    ("mishima_coherence_diagnostic", exp_coherence_diagnostic),
    ("mishima_closure_beats_threshold", exp_closure_beats_threshold),
    ("mishima_decline_is_typed", exp_decline_is_typed),
    ("mishima_water_filling", exp_water_filling),
    ("mishima_knapsack_priority", exp_knapsack_priority),
    ("mishima_probe_kind_invisible", exp_probe_kind_invisible),
    ("mishima_floor_negotiation", exp_floor_negotiation),
]
