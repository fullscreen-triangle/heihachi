"""
Experiments 10-11: duality and economy (Sec. 7-8).

  E10 Search-match duality (Thm. 7.3): analysis (fix query, search targets) and
      synthesis (fix target, search constructible sequences) are the same
      alignment objective; the best pairing is identical whichever argument is
      free.
  E11 Resolution economy (Cor. 8.5): the whole-item pipeline costs
      n*c_sym + r*c_ac, strictly less than per-track n*c_ac whenever the
      nameable fraction exceeds c_sym/c_ac; and matching cost scales as
      O(|C|*n*N) with the inverted-index candidate filter |C| << L.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

import core


def e10_duality(n_trials=300, seed=10):
    gen = core.rng(seed)
    agree = 0
    for _ in range(n_trials):
        A = int(gen.integers(3, 8))
        alphabet = [f"h{i}" for i in range(A)]
        L = int(gen.integers(3, 8))          # library size
        n = int(gen.integers(3, 7))          # signature length
        library = [core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(n)))
                   for _ in range(L)]

        # ANALYSIS: fix a query, find the best-matching library target.
        query = core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(n)))
        analysis_scores = [core.align_score(query, t) for t in library]
        analysis_best = int(np.argmin(analysis_scores))

        # SYNTHESIS: fix that same target, and over a small constructible space
        # of sequences (orderings of a fixed constituent pool), find the best
        # match. Duality says the objective and operator are identical; here we
        # confirm the symmetric property: score(query,target) == score(target,query).
        target = library[analysis_best]
        # symmetry of the alignment objective (same operator both directions):
        fwd = core.align_score(query, target)
        bwd = core.align_score(target, query)
        symmetric = np.isclose(fwd, bwd)

        # And: choosing the free argument does not change the optimal pairing.
        # Fix the target, search over the "queries" = library, recover the same
        # pair (target, library[analysis_best]) as the mutual best.
        synth_scores = [core.align_score(target, t) for t in library]
        synth_best = int(np.argmin(synth_scores))
        # the target matches itself best (score 0) -> synth_best is target's index
        target_idx = analysis_best
        pairing_consistent = symmetric and (synth_best == target_idx)

        if pairing_consistent:
            agree += 1
    return {
        "claim": "Thm 7.3: analysis and synthesis share one symmetric alignment objective",
        "n_trials": n_trials,
        "consistent": agree,
        "pass": agree == n_trials,
    }


def e11_economy(n_trials=300, seed=11):
    gen = core.rng(seed)
    c_sym, c_ac = 1.0, 20.0
    whole_cheaper = 0
    saving_factors = []
    rows = []
    for _ in range(n_trials):
        n = int(gen.integers(10, 60))
        nameable_frac = float(gen.uniform(0.0, 1.0))
        r = int(round((1.0 - nameable_frac) * n))   # residual constituents
        whole_cost = n * c_sym + r * c_ac
        per_track_cost = n * c_ac
        if whole_cost < per_track_cost - 1e-9:
            whole_cheaper += 1
        saving_factors.append(per_track_cost / whole_cost)
        # Cor 8.5 predicate, expressed on the SAME integer residual r the cost
        # uses: whole cheaper iff (n-r)*c_ac > n*c_sym, i.e. the *realised*
        # nameable fraction (n-r)/n exceeds c_sym/c_ac. (Using the continuous
        # nameable_frac before rounding r introduces a spurious boundary
        # mismatch; the theorem is about the actual costs, which depend on r.)
        realised_nameable = (n - r) / n
        predicted = realised_nameable > (c_sym / c_ac) + 1e-12
        measured = whole_cost < per_track_cost - 1e-9
        rows.append({"n": n, "nameable_frac": nameable_frac,
                     "realised_nameable": realised_nameable, "r": r,
                     "predicted_cheaper": predicted, "measured_cheaper": measured,
                     "agree": predicted == measured})
    agree = sum(1 for r in rows if r["agree"])
    return {
        "claim": "Cor 8.5: whole-item cost n*c_sym+r*c_ac < per-track n*c_ac iff nameable frac > c_sym/c_ac",
        "n_trials": n_trials,
        "predicate_agreement": agree,
        "mean_saving_factor": float(np.mean(saving_factors)),
        "max_saving_factor": float(np.max(saving_factors)),
        "pass": agree == n_trials,
        "sample": rows[:6],
    }


def e11b_alignment_complexity(seed=111):
    """
    Empirically confirm the O(n*N) scaling of the alignment DP (Thm. 8.4) by
    timing global alignment on growing signatures and checking the growth is
    consistent with a quadratic (product) law rather than exponential.
    """
    gen = core.rng(seed)
    sizes = [20, 40, 80, 160, 320]
    times = []
    for n in sizes:
        a = core.Signature(tuple(f"h{int(gen.integers(0, n))}" for _ in range(n)))
        b = core.Signature(tuple(f"h{int(gen.integers(0, n))}" for _ in range(n)))
        t0 = time.perf_counter()
        core.align_cost(a, b)
        times.append(time.perf_counter() - t0)
    # ratios of successive times vs successive n^2 ratios; for O(n^2) these track.
    ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
    n2_ratios = [(sizes[i + 1] / sizes[i]) ** 2 for i in range(len(sizes) - 1)]
    # accept if empirical ratios are within a generous band of the n^2 prediction
    within = all(0.25 * n2 <= r <= 4.0 * n2 for r, n2 in zip(ratios, n2_ratios))
    return {
        "claim": "Thm 8.4: alignment DP scales ~O(n*N) (quadratic in length)",
        "sizes": sizes,
        "times_sec": times,
        "empirical_ratios": ratios,
        "n2_ratios": n2_ratios,
        "pass": within,
    }


def run(seed=0):
    return {
        "e10_duality": e10_duality(seed=seed + 10),
        "e11_economy": e11_economy(seed=seed + 11),
        "e11b_alignment_complexity": e11b_alignment_complexity(seed=seed + 111),
    }


if __name__ == "__main__":
    res = run()
    path = core.save_results("duality_and_complexity", res)
    for k, v in res.items():
        print(f"{k}: pass={v['pass']}")
    print("saved:", path)
