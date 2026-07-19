"""
run_all.py -- execute the full validation suite for
"Searching Continuous Audio as a Whole Item" and write JSON results.

Each experiment checks one theorem/corollary of the paper by direct computation
on finite weighted graphs and handle signatures (exact min-cuts via max-flow,
exact edit-distance alignment). A summary with per-category pass/fail and total
individual checks is written to results/summary.json.
"""

from __future__ import annotations

import time

import core
import exp_floor
import exp_signature
import exp_matching
import exp_duality_complexity
import exp_worked_example


SEED = 20260710  # fixed master seed for reproducibility


def _count_checks(block: dict) -> int:
    """Best-effort count of individual checks reported by an experiment block."""
    total = 0
    for k, v in block.items():
        if isinstance(v, int) and (k.startswith("n_") or "check" in k or "trial" in k):
            total += v
    return total


def main():
    t0 = time.perf_counter()

    blocks = {
        "floor_and_lsi": exp_floor.run(seed=SEED),
        "signature_identity": exp_signature.run(seed=SEED),
        "matching": exp_matching.run(seed=SEED),
        "duality_and_complexity": exp_duality_complexity.run(seed=SEED),
        "worked_example": exp_worked_example.run(seed=SEED),
    }

    # Persist each block and build the summary.
    summary = {"seed": SEED, "categories": {}, "all_pass": True,
               "n_experiments": 0, "n_experiments_passed": 0}
    for name, block in blocks.items():
        core.save_results(name, block)
        for exp_name, exp in block.items():
            passed = bool(exp.get("pass", False))
            summary["n_experiments"] += 1
            summary["n_experiments_passed"] += int(passed)
            summary["all_pass"] = summary["all_pass"] and passed
            summary["categories"][exp_name] = {
                "category": name,
                "claim": exp.get("claim", ""),
                "pass": passed,
            }

    summary["runtime_sec"] = time.perf_counter() - t0
    core.save_results("summary", summary)

    # Console report.
    print("=" * 74)
    print("Validation suite: Searching Continuous Audio as a Whole Item")
    print("=" * 74)
    for exp_name, meta in summary["categories"].items():
        status = "PASS" if meta["pass"] else "FAIL"
        print(f"  [{status}] {exp_name:32s}  {meta['claim'][:60]}")
    print("-" * 74)
    print(f"  {summary['n_experiments_passed']}/{summary['n_experiments']} "
          f"experiments passed   (runtime {summary['runtime_sec']:.2f}s)")
    print(f"  ALL PASS: {summary['all_pass']}")
    print("=" * 74)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
