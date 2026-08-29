"""Run every experiment and write one JSON record per experiment.

    python run_all.py

Writes results/<name>.json and results/_summary.json. Every seed is
fixed, so re-running reproduces each record bit-identically.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import exp_kernel
import exp_mishima
import exp_sangoma

SUITES = [
    ("kernel", exp_kernel.EXPERIMENTS),
    ("mishima", exp_mishima.EXPERIMENTS),
    ("sangoma", exp_sangoma.EXPERIMENTS),
]


def main() -> int:
    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    summary: dict[str, object] = {"suites": {}, "run": 0, "passed": 0,
                                  "failed": 0}
    failures: list[str] = []
    started = time.time()

    for suite_name, experiments in SUITES:
        s_pass = s_fail = 0
        for name, fn in experiments:
            try:
                record = fn()
                ok = bool(record.get("passed"))
            except Exception as exc:                      # noqa: BLE001
                record = {"error": f"{type(exc).__name__}: {exc}",
                          "passed": False}
                ok = False
            record["experiment"] = name
            record["suite"] = suite_name
            (out / f"{name}.json").write_text(
                json.dumps(record, indent=2, default=str, sort_keys=True),
                encoding="utf-8")
            summary["run"] = int(summary["run"]) + 1
            if ok:
                s_pass += 1
                summary["passed"] = int(summary["passed"]) + 1
            else:
                s_fail += 1
                summary["failed"] = int(summary["failed"]) + 1
                failures.append(name)
            print(f"{'pass' if ok else 'FAIL'}  {name}")
        summary["suites"][suite_name] = {          # type: ignore[index]
            "experiments": len(experiments), "passed": s_pass, "failed": s_fail}

    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary["failures"] = failures
    (out / "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n{summary['run']} run, {summary['passed']} passed, "
          f"{summary['failed']} failed "
          f"({summary['elapsed_seconds']}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
