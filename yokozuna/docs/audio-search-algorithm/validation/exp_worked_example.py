"""
Experiment 12: the paper's worked example (Sec. 9), end-to-end.

Two mixes over the pool {A,B,C,D,E}:
    Mix 1: (A,B,C,D,E)
    Mix 2: (A,C,B,E,D)
Identical content sets; distinct signatures. A degraded capture of Mix 1 has
constituent C unnamed (an "ID") and D misheard as a distinct same-titled edit
D'. Acquisition yields (A,B,_,D',E). We confirm:
  * content sets are equal (bag-of-tracks cannot separate the mixes),
  * signatures/patterns differ (sequence separates them),
  * the degraded query aligns to Mix 1 at cost delta+mu = 2, score 0.4,
  * with eps >= 0.4 the match is admissible and identifies Mix 1,
  * the residual is exactly positions {2,3} (0-indexed) -- the ID and the edit,
  * no acoustic work is owed on A,B,E.
This single instance exercises Thm 5.4, Cor 4.8, Prop 6.10, and Thm 6.6.
"""

from __future__ import annotations

import numpy as np

import core

GAP = "GAP"  # explicit unnamed-constituent handle (an "ID" track)


def run(seed=0):
    mix1 = core.Signature(("A", "B", "C", "D", "E"))
    mix2 = core.Signature(("A", "C", "B", "E", "D"))

    # 1. Content sets identical.
    content_equal = mix1.content_multiset() == mix2.content_multiset()

    # 2. Signatures differ over the FIXED public alphabet. The handles A..E name
    # specific tracks in the shared pool (Sec. 5 / Rem. 5.5): they are not freely
    # relabelled, so the discriminating identity is the signature itself, not the
    # abstract re-encoding-invariant pattern. (Two all-distinct sequences share
    # the trivial pattern (0,1,2,3,4); what tells these two mixes apart is that
    # over the fixed alphabet their ordered handle strings differ, which a
    # content-set matcher cannot see.) The alignment below operates on these
    # fixed handles and separates the mixes at positive cost.
    signatures_differ = mix1.handles != mix2.handles
    content_cannot_separate = content_equal and signatures_differ

    # 3. Degraded capture of Mix 1: C unnamed (GAP), D -> D' (substitution).
    query = core.Signature(("A", "B", GAP, "Dprime", "E"))

    mu = delta = 1.0
    cost, matched, residual = core.align_traceback(query, mix1, sub=mu, gap=delta)
    omega = delta * max(len(query), len(mix1))
    score = cost / omega

    # 4. Admissibility at eps = 0.4.
    eps = 0.4
    admissible = score <= eps + 1e-9

    # Also confirm the query aligns to Mix 1 better than to Mix 2 (correct id).
    score_vs_mix2 = core.align_score(query, mix2, sub=mu, gap=delta)
    identifies_mix1 = score < score_vs_mix2

    # 5. Residual == {2,3}; matched == {0,1,4}.
    residual_set = set(residual)
    matched_set = set(matched)
    residual_correct = residual_set == {2, 3}
    matched_correct = matched_set == {0, 1, 4}

    # 6. No acoustic owed on A,B,E: model each mix-1 constituent, ambiguity set =
    # the rest, and confirm A,B,E resolve by name while C (unnamed) and the
    # D/D' collision need acoustic.
    constituents = {
        0: core.Constituent(name="A", acoustic="acA"),
        1: core.Constituent(name="B", acoustic="acB"),
        2: core.Constituent(name=None, acoustic="acC"),          # the ID
        3: core.Constituent(name="D", acoustic="acD"),
        4: core.Constituent(name="E", acoustic="acE"),
    }
    # Inject a colliding D' elsewhere in the ambiguity set to make the name 'D'
    # insufficient (a same-title, acoustically-distinct edit).
    collider = core.Constituent(name="D", acoustic="acDprime")
    acoustic_positions = set()
    for i, c in constituents.items():
        amb = [constituents[j] for j in constituents if j != i] + [collider]
        _, kind, _ = core.least_sufficient_identifier(c, amb, c_sym=1.0, c_ac=10.0)
        if kind == "acoustic":
            acoustic_positions.add(i)
    # C (unnamed) and D (name collides with D') should need acoustic; A,B,E not.
    acoustic_correct = acoustic_positions == {2, 3}

    result = {
        "claim": "Sec 9 worked example: end-to-end identification of a degraded whole item",
        "content_sets_equal": content_equal,
        "signatures_differ": signatures_differ,
        "content_cannot_separate_but_signature_can": content_cannot_separate,
        "alignment_cost_to_mix1": cost,
        "alignment_score_to_mix1": score,
        "alignment_score_to_mix2": score_vs_mix2,
        "expected_cost": 2.0,
        "expected_score": 0.4,
        "admissible_at_eps_0.4": admissible,
        "identifies_mix1_over_mix2": identifies_mix1,
        "residual_positions": sorted(residual_set),
        "matched_positions": sorted(matched_set),
        "residual_correct": residual_correct,
        "matched_correct": matched_correct,
        "acoustic_positions": sorted(acoustic_positions),
        "acoustic_correct": acoustic_correct,
        "pass": bool(
            content_equal and signatures_differ and content_cannot_separate
            and np.isclose(cost, 2.0) and np.isclose(score, 0.4)
            and admissible and identifies_mix1
            and residual_correct and matched_correct and acoustic_correct
        ),
    }
    return {"e12_worked_example": result}


if __name__ == "__main__":
    res = run()
    path = core.save_results("worked_example", res)
    for k, v in res.items():
        print(f"{k}: pass={v['pass']}")
    print("saved:", path)
