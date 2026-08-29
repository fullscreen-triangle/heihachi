"""Cross-check figures quoted in the manuscripts against results/.

    python check_claims.py

Every number a paper states as measured must appear in the JSON record
of the experiment it attributes it to. This catches a figure that was
edited in the prose but not re-derived, which is the failure mode that
makes a validation suite worthless.
"""

from __future__ import annotations

import json
import pathlib
import sys

RESULTS = pathlib.Path(__file__).parent / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


CHECKS: list[tuple[str, str, callable]] = [
    # ---- Paper I -----------------------------------------------------
    ("I: 48 cells swept", "kernel_run_to_completion",
     lambda r: r["cells"] == 48),
    ("I: kernel completes every cell", "kernel_run_to_completion",
     lambda r: r["kernel_always_complete"] is True),
    ("I: worst halting fraction 0.094", "kernel_run_to_completion",
     lambda r: close(r["worst_halting_fraction"], 0.0938, 1e-4)),
    ("I: 29 nodes recovered", "kernel_run_to_completion",
     lambda r: r["max_nodes_recovered"] == 29),
    ("I: no verdict key in report", "kernel_no_exit_code",
     lambda r: r["verdict_keys_exposed"] == []),
    ("I: 0 record decreases", "kernel_monotone_record",
     lambda r: r["decreases"] == 0),
    ("I: repeats 401,402,403", "kernel_monotone_record",
     lambda r: r["repeat_records"] == [401, 402, 403]),
    ("I: 36 runs, 6 trajectories", "kernel_trajectory_emergence",
     lambda r: r["runs"] == 36 and r["distinct_trajectories"] == 6),
    ("I: 300 trials 0 mismatches", "kernel_mincut_brute_force",
     lambda r: r["trials"] == 300 and r["mismatches"] == 0),
    ("I: mincut error < 1e-15", "kernel_mincut_brute_force",
     lambda r: r["max_abs_difference"] < 1e-15),
    ("I: blast radius b=4 spans 1..256", "kernel_blast_radius",
     lambda r: r["spans"]["4"] == [1, 256]),
    ("I: floor ratio spread 0", "kernel_floor_is_scale",
     lambda r: close(r["spread"], 0.0)),
    ("I: 3 of 3 zero floors refused", "kernel_floor_positivity",
     lambda r: r["refused"] == 3),
    ("I: 4 of 4 chunks fired", "kernel_total_chunk_execution",
     lambda r: r["all_fired"] is True),

    # ---- Paper II ----------------------------------------------------
    ("II: worked example power 0.8267", "mishima_parses_worked_example",
     lambda r: close(r["composite_power"], 0.8267, 1e-9)),
    ("II: mandatory-not fires at parse", "mishima_mandatory_not",
     lambda r: r["rule"] == "rule:mandatory-not"),
    ("II: composition error < 1e-12", "mishima_composition_law",
     lambda r: r["max_abs_difference"] < 1e-12),
    ("II: harmonic gap 2.5e-4", "mishima_saturation_dichotomy",
     lambda r: close(r["harmonic_residual_gap"], 2.4994e-4, 1e-6)),
    ("II: geometric gap 0.289", "mishima_saturation_dichotomy",
     lambda r: close(r["geometric_residual_gap"], 0.28879, 1e-4)),
    ("II: 20 of 20 depths agree", "mishima_depth_closed_form",
     lambda r: r["combinations"] == 20 and r["mismatches"] == 0),
    ("II: refinement cost ratio 1e4", "mishima_refinement_diverges",
     lambda r: close(r["cost_ratio_first_to_last"], 10000.0, 1.0)),
    ("II: diversify counterexample stands", "mishima_diversify",
     lambda r: r["strongest_repetition_wins"] is True),
    ("II: ladder 0.8267 vs 0.9089", "mishima_diversify",
     lambda r: close(r["distinct_composite"], 0.8267, 1e-4)
               and close(r["repeated"]["0.55"], 0.9089, 1e-4)),
    ("II: chain and pair fragile", "mishima_triangle",
     lambda r: not r["chain_robust"] and not r["pair_robust"]
               and r["triangle_robust"]),
    ("II: threshold after 1 probe always", "mishima_closure_beats_threshold",
     lambda r: all(row["threshold_met_after"] == 1 for row in r["rows"])),
    ("II: closure needs whole registry", "mishima_closure_beats_threshold",
     lambda r: all(row["closure_after"] == row["registry"]
                   for row in r["rows"])),
    ("II: all 5 registries declined", "mishima_closure_beats_threshold",
     lambda r: r["declined"] == 5 and r["registries"] == 5),
    ("II: waterfill spread < 1e-3", "mishima_water_filling",
     lambda r: r["marginal_spread"] < 1e-3),

    # ---- Paper III ---------------------------------------------------
    ("III: worked example power 0.8245", "sangoma_parses_worked_example",
     lambda r: close(r["composite_power"], 0.8245, 1e-9)),
    ("III: floor 0.02, 3 stages, 3 targets", "sangoma_parses_worked_example",
     lambda r: close(r["floor"], 0.02) and len(r["stages"]) == 3
               and r["targets"] == 3),
    ("III: target needs resolution", "sangoma_target_needs_resolution",
     lambda r: r["rule"] == "rule:floor-on-target"),
    ("III: depth matches closed form", "sangoma_reachability_reports_depth",
     lambda r: r["depth_matches_closed_form"] is True),
    ("III: detectability tau<1 boundary", "sangoma_detectability",
     lambda r: [row["observable"] for row in r["rows"]]
               == [True, True, False, False]),
    ("III: chain thickness 0.10", "sangoma_thickness_additive",
     lambda r: close(r["total"], 0.10)),
    ("III: latency 2560 samples", "sangoma_latency_accounted",
     lambda r: r["total_samples"] == 2560),
    ("III: crossover at 0.95 is 1.00", "sangoma_nesting_beats_sharpening",
     lambda r: close(
         next(x["ratio"] for x in r["rows"] if close(x["target"], 0.95)),
         1.0, 1e-6)),
    ("III: ratio 20.6 at 0.999", "sangoma_nesting_beats_sharpening",
     lambda r: close(
         next(x["ratio"] for x in r["rows"] if close(x["target"], 0.999)),
         20.588, 1e-2)),
    ("III: 17 rungs at 0.999", "sangoma_nesting_beats_sharpening",
     lambda r: next(x["ladder_rungs"] for x in r["rows"]
                    if close(x["target"], 0.999)) == 17),
    ("III: every diagnostic has a remedy", "sangoma_diagnostics_carry_remedy",
     lambda r: r["diagnostics"] == r["with_remedy"] and r["diagnostics"] > 0),
]


def main() -> int:
    if not RESULTS.exists():
        print("results/ not found -- run run_all.py first")
        return 1
    bad: list[str] = []
    for label, experiment, predicate in CHECKS:
        try:
            ok = bool(predicate(load(experiment)))
        except Exception as exc:                          # noqa: BLE001
            ok = False
            label += f"  [{type(exc).__name__}: {exc}]"
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            bad.append(label)
    print(f"\n{len(CHECKS)} claims checked, {len(CHECKS) - len(bad)} confirmed, "
          f"{len(bad)} unsupported")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
