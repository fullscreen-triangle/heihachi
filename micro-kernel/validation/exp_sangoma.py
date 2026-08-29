"""Experiments for Paper III: sangoma, the instrument constructor."""

from __future__ import annotations

from hkcore import (CheckError, Stage, chain_thickness, check_sangoma,
                    composite_power, detectability, floor_negotiation,
                    headroom_after, latency_total, parse_sangoma, reachable,
                    rungs_required)

GOOD = """-- reese.sgn
floor 0.02

medium air {
  ceiling: -1.0
}

species reese {
  source { operators: 2, ratio: 1.0, index: 3.5 }
  motion { rate: 0.3, depth: 0.8 }
}

construct bass {
  stage fm_source
  stage resample
  stage saturate

  target {
    crest     >= 6.0#0.5
    midrange  >= 0.40#0.05
    width     <= 0.85#0.02
  }

  via { rung fm_source at 0.40
     >> rung resample  at 0.35
     >> rung saturate  at 0.55 }
}
"""


def exp_parses_worked_example() -> dict:
    """The constructor program parses and its declarations are recovered."""
    prog = parse_sangoma(GOOD)
    c = prog.constructs[0]
    power = composite_power([k for _, k in c.ladder])
    return {"floor": prog.ambient_floor,
            "species": sorted(prog.species.keys()),
            "stages": c.stages, "targets": len(c.targets),
            "composite_power": round(power, 4),
            "passed": (prog.ambient_floor == 0.02 and len(c.stages) == 3
                       and len(c.targets) == 3
                       and abs(power - 0.8245) < 1e-9)}


def exp_target_needs_resolution() -> dict:
    """A target magnitude with no floor annotation is refused."""
    bad = "floor 0.02\nconstruct b { target { crest >= 6.0 } }"
    rule = ""
    try:
        parse_sangoma(bad)
        refused = False
    except CheckError as exc:
        refused, rule = True, exc.rule
    good = "floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } }"
    parse_sangoma(good)
    return {"refused": refused, "rule": rule,
            "annotated_accepted": True,
            "passed": refused and rule == "rule:floor-on-target"}


def exp_over_claim_refused() -> dict:
    """A target finer than the ambient floor is rejected before rendering."""
    src = ("floor 0.05\nconstruct b { target { crest >= 6.0#0.001 } "
           "via { rung a at 0.9 } }")
    diags = check_sangoma(parse_sangoma(src))
    rules = [d.rule for d in diags]
    remedies = [d.remedy for d in diags if d.rule == "rule:over-claim"]
    return {"rules": rules, "remedy": remedies[0] if remedies else "",
            "passed": "rule:over-claim" in rules and bool(remedies[0])}


def exp_reachability_reports_depth() -> dict:
    """An unreachable target is refused with the required depth named."""
    rows = []
    for target in (0.9, 0.95, 0.99):
        stages = [Stage("a", 0.01, 0.25), Stage("b", 0.01, 0.20)]
        ok, depth = reachable(target, stages)
        closed = rungs_required(target, 0.25)
        rows.append({"target": target, "reachable": ok,
                     "reported_depth": depth, "closed_form": closed,
                     "attainable": round(composite_power([0.25, 0.20]), 4)})
    agree = all(r["reported_depth"] == r["closed_form"] for r in rows)
    return {"rows": rows, "depth_matches_closed_form": agree,
            "passed": agree and not any(r["reachable"] for r in rows)}


def exp_reachability_diagnostic_fires() -> dict:
    """The checker names the shortfall rather than failing at render time."""
    weak = ("floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } "
            "via { rung a at 0.2 >> rung b at 0.2 } }")
    strong = ("floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } "
              "via { rung a at 0.5 >> rung b at 0.5 >> rung c at 0.5 } }")
    w = [d.rule for d in check_sangoma(parse_sangoma(weak), 0.8)]
    s = [d.rule for d in check_sangoma(parse_sangoma(strong), 0.8)]
    msg = [d.remedy for d in check_sangoma(parse_sangoma(weak), 0.8)
           if d.rule == "rule:reachability"]
    return {"weak_rules": w, "strong_rules": s,
            "weak_remedy": msg[0] if msg else "",
            "passed": "rule:reachability" in w and "rule:reachability" not in s}


def exp_detectability() -> dict:
    """A stage whose separator engulfs the assembly is not observable."""
    rows = []
    for thickness, extent in ((0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0)):
        tau = detectability(thickness, extent)
        rows.append({"thickness": thickness, "extent": extent,
                     "tau": tau, "observable": tau < 1.0})
    return {"rows": rows,
            "passed": (rows[0]["observable"] and rows[1]["observable"]
                       and not rows[2]["observable"]
                       and not rows[3]["observable"])}


def exp_thickness_additive() -> dict:
    """Chained stages accumulate thickness; nothing gets sharper downstream."""
    stages = [Stage("fm", 0.02, 0.4), Stage("resample", 0.03, 0.35),
              Stage("saturate", 0.05, 0.55)]
    total = chain_thickness(stages)
    monotone = True
    running = 0.0
    for s in stages:
        nxt = running + s.thickness
        if nxt < running:
            monotone = False
        running = nxt
    return {"stage_thicknesses": [s.thickness for s in stages],
            "total": round(total, 6), "never_decreases": monotone,
            "passed": abs(total - 0.10) < 1e-9 and monotone}


def exp_headroom_refusal() -> dict:
    """A chain whose summed gain breaches the ceiling is refused."""
    peak_ok, clears_ok = headroom_after([], [-6.0, 2.0, 1.0], ceiling_db=-1.0)
    peak_bad, clears_bad = headroom_after([], [-2.0, 3.0, 2.0], ceiling_db=-1.0)
    return {"safe_peak_db": peak_ok, "safe_clears": clears_ok,
            "unsafe_peak_db": peak_bad, "unsafe_clears": clears_bad,
            "passed": clears_ok and not clears_bad}


def exp_latency_accounted() -> dict:
    """Chain latency is the sum of declared stage latencies."""
    stages = [Stage("fm", 0.02, 0.4, latency=0),
              Stage("linear_eq", 0.01, 0.2, latency=2048),
              Stage("lookahead", 0.01, 0.3, latency=512)]
    return {"stage_latencies": [s.latency for s in stages],
            "total_samples": latency_total(stages),
            "passed": latency_total(stages) == 2560}


def exp_floor_negotiation() -> dict:
    """A program finer than the render path is refused, both figures named."""
    ok, why_ok = floor_negotiation(0.05, 0.01)
    bad, why_bad = floor_negotiation(0.001, 0.01)
    return {"coarse_program_runs": ok,
            "fine_program_runs": bad,
            "refusal": why_bad,
            "names_both": "0.001" in why_bad and "0.01" in why_bad,
            "passed": ok and not bad and "0.001" in why_bad}


def exp_nesting_beats_sharpening() -> dict:
    """Composition overtakes sharpening as the target tightens.

    Under the reciprocal cost 1/t the two are exactly equal at a
    target of 0.95, which is an artefact of that particular cost law
    rather than a result. The claim that survives is the asymptotic
    one: single-boundary cost grows without bound as the target
    approaches one, while ladder cost grows only logarithmically, so
    the ratio diverges. The table records the crossover.
    """
    from hkcore import refinement_cost
    per_rung = 0.35
    rows = []
    for target in (0.90, 0.95, 0.99, 0.999, 0.9999):
        single = refinement_cost(1.0 - target)
        n = rungs_required(target, per_rung)
        ladder = n * refinement_cost(per_rung)
        rows.append({"target": target,
                     "single_cost": round(single, 2),
                     "ladder_rungs": n,
                     "ladder_cost": round(ladder, 2),
                     "ratio": round(single / ladder, 3)})
    ratios = [r["ratio"] for r in rows]
    increasing = all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
    return {"rows": rows, "per_rung_power": per_rung,
            "ratio_increases_with_target": increasing,
            "final_ratio": ratios[-1],
            "passed": increasing and ratios[-1] > 10.0}


def exp_target_required() -> dict:
    """A construct that declares no target is refused."""
    src = "floor 0.02\nconstruct b { stage a }"
    rules = [d.rule for d in check_sangoma(parse_sangoma(src))]
    return {"rules": rules,
            "passed": "rule:target-required" in rules}


def exp_every_diagnostic_carries_remedy() -> dict:
    """No refusal is a dead end."""
    sources = [
        "construct b { stage a }",
        "floor 0.05\nconstruct b { target { crest >= 6.0#0.001 } }",
        ("floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } "
         "via { rung a at 0.1 } }"),
    ]
    total, with_remedy = 0, 0
    for src in sources:
        for d in check_sangoma(parse_sangoma(src), 0.8):
            total += 1
            if d.remedy.strip():
                with_remedy += 1
    return {"diagnostics": total, "with_remedy": with_remedy,
            "passed": total > 0 and total == with_remedy}


EXPERIMENTS = [
    ("sangoma_parses_worked_example", exp_parses_worked_example),
    ("sangoma_target_needs_resolution", exp_target_needs_resolution),
    ("sangoma_over_claim_refused", exp_over_claim_refused),
    ("sangoma_reachability_reports_depth", exp_reachability_reports_depth),
    ("sangoma_reachability_diagnostic", exp_reachability_diagnostic_fires),
    ("sangoma_detectability", exp_detectability),
    ("sangoma_thickness_additive", exp_thickness_additive),
    ("sangoma_headroom_refusal", exp_headroom_refusal),
    ("sangoma_latency_accounted", exp_latency_accounted),
    ("sangoma_floor_negotiation", exp_floor_negotiation),
    ("sangoma_nesting_beats_sharpening", exp_nesting_beats_sharpening),
    ("sangoma_target_required", exp_target_required),
    ("sangoma_diagnostics_carry_remedy", exp_every_diagnostic_carries_remedy),
]
