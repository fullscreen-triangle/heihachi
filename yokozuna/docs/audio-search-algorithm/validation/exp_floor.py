"""
Experiments 1-4: the resolution floor and least-sufficient identifier (Sec. 4).

  E1  Floor Theorem (Thm. 4.3)          -- sigma(v) >= beta on random graphs.
  E2  No exact identification (Cor. 4.5) -- identity is region-valued
                                            (min-cut side often non-singleton).
  E3  Sufficiency stops resolution (Thm. 4.7) -- once a handle separates v from
      its ambiguity set, finer resolution does not change placement.
  E4  Name-first economy (Cor. 4.8) -- acoustic recognition is invoked only on
      the residual (unnamed or name-colliding constituents).
"""

from __future__ import annotations

import numpy as np

import core


def e1_floor(n_graphs=300, seed=1):
    gen = core.rng(seed)
    checks = 0
    min_ratio = np.inf
    below_floor = 0
    per = []
    for _ in range(n_graphs):
        n_items = int(gen.integers(3, 15))
        floor = float(gen.uniform(0.05, 2.0))
        g = core.random_contact_graph(
            n_items, floor,
            edge_prob=float(gen.uniform(0.2, 0.8)),
            weight_spread=float(gen.uniform(0.1, 2.0)),
            gen=gen,
        )
        items = [x for x in g.nodes if x != core.MEDIUM]
        for v in items:
            sigma, _ = core.min_cut_against_medium(g, v)
            checks += 1
            ratio = sigma / floor
            min_ratio = min(min_ratio, ratio)
            if sigma < floor - 1e-9:
                below_floor += 1
        per.append({"n_items": n_items, "floor": floor,
                    "realised_floor": core.realised_floor(g)})
    return {
        "claim": "Thm 4.3 Floor: sigma(v) >= beta for every constituent",
        "n_graphs": n_graphs,
        "n_cut_checks": checks,
        "cuts_below_floor": below_floor,
        "min_sigma_over_beta": float(min_ratio),
        "pass": below_floor == 0 and min_ratio >= 1.0 - 1e-9,
        "sample": per[:5],
    }


def e2_region(n_trials=200, seed=2):
    gen = core.rng(seed)
    non_singleton = 0
    sizes = []
    for _ in range(n_trials):
        k = int(gen.integers(2, 7))
        g = core.two_cluster_graph(k, floor=1.0,
                                   dense_weight=float(gen.uniform(3.0, 8.0)),
                                   gen=gen)
        # Cheapest nontrivial separation: cut the A-B bridge, isolating a whole
        # cluster. Verify the min cut of an A-item against the medium places a
        # multi-item set on its side (identity borne by a region).
        v = ("A", 0)
        sigma, side = core.min_cut_against_medium(g, v)
        item_side = [x for x in side if x != core.MEDIUM]
        sizes.append(len(item_side))
        if len(item_side) > 1:
            non_singleton += 1
    return {
        "claim": "Cor 4.5 / Thm (region): min-cut side is in general not a singleton",
        "n_trials": n_trials,
        "non_singleton_fraction": non_singleton / n_trials,
        "mean_side_size": float(np.mean(sizes)),
        "max_side_size": int(np.max(sizes)),
        # The two-cluster construction is designed to always give a region;
        # a healthy fraction of non-singleton sides confirms region-valuedness.
        "pass": non_singleton / n_trials >= 0.9,
    }


def e3_sufficiency(n_trials=500, seed=3):
    """
    Once a handle is sufficient (separates v from its ambiguity set), any finer
    resolution keeps the placement unchanged. We model 'placement' as: which
    alternative in the ambiguity set (if any) v is confused with. A sufficient
    handle confuses v with nobody; refining v (adding more acoustic detail)
    still confuses v with nobody -- placement is invariant. We also confirm the
    negative: an insufficient handle (name collision) DOES leave a confusion
    that only acoustic descent removes.
    """
    gen = core.rng(seed)
    sufficient_stable = 0
    insufficient_needs_acoustic = 0
    n_suff = 0
    n_insuff = 0
    for _ in range(n_trials):
        vname = "T"
        # Build an ambiguity set; sometimes inject a name collision.
        collision = gen.random() < 0.5
        amb = []
        for _ in range(int(gen.integers(1, 6))):
            other_name = vname if (collision and gen.random() < 0.4) else f"N{int(gen.integers(0, 1000))}"
            amb.append(core.Constituent(name=other_name, acoustic=f"ac{int(gen.integers(0,10000))}"))
        v = core.Constituent(name=vname, acoustic="acV")

        name_sufficient = all(o.name != v.name for o in amb)
        handle, kind, _ = core.least_sufficient_identifier(v, amb, c_sym=1.0, c_ac=10.0)

        if name_sufficient:
            n_suff += 1
            # placement under the name: confused set = alternatives sharing handle
            confused_name = [o for o in amb if o.name == v.name]
            # placement under a finer (acoustic) resolution:
            confused_ac = [o for o in amb if o.acoustic == v.acoustic]
            if len(confused_name) == 0 and len(confused_ac) == 0:
                sufficient_stable += 1
        else:
            n_insuff += 1
            # name leaves a collision; acoustic resolves it (distinct acoustics)
            confused_name = [o for o in amb if o.name == v.name]
            confused_ac = [o for o in amb if o.acoustic == v.acoustic]
            if len(confused_name) > 0 and len(confused_ac) == 0 and kind == "acoustic":
                insufficient_needs_acoustic += 1

    return {
        "claim": "Thm 4.7: sufficiency stops resolution; insufficiency requires acoustic descent",
        "n_trials": n_trials,
        "n_sufficient": n_suff,
        "sufficient_placement_stable": sufficient_stable,
        "n_insufficient": n_insuff,
        "insufficient_resolved_by_acoustic": insufficient_needs_acoustic,
        "pass": (sufficient_stable == n_suff) and (insufficient_needs_acoustic == n_insuff),
    }


def e4_name_first(n_trials=400, seed=4):
    """
    Cor 4.8: acoustic recognition is invoked only where the name is unavailable
    or insufficient. We simulate streams with a controlled unnamed/collision
    fraction and confirm the acoustic-invocation set equals exactly the residual.
    """
    gen = core.rng(seed)
    all_correct = 0
    rows = []
    for _ in range(n_trials):
        n = int(gen.integers(5, 40))
        unnamed_frac = float(gen.uniform(0.0, 0.6))
        constituents = []
        expected_acoustic = set()
        used_names = {}
        for i in range(n):
            unnamed = gen.random() < unnamed_frac
            if unnamed:
                c = core.Constituent(name=None, acoustic=f"ac{i}")
                expected_acoustic.add(i)
            else:
                nm = f"trk{int(gen.integers(0, max(2, n//2)))}"
                c = core.Constituent(name=nm, acoustic=f"ac{i}")
                used_names.setdefault(nm, []).append(i)
            constituents.append(c)

        # A name is insufficient if another constituent shares it but differs
        # acoustically -> that position also needs acoustic descent.
        for nm, idxs in used_names.items():
            if len(idxs) > 1:
                for i in idxs:
                    expected_acoustic.add(i)

        # Run the least-sufficient resolution per position, ambiguity set = the
        # rest of the stream (worst case), and record where acoustic fired.
        invoked_acoustic = set()
        for i, c in enumerate(constituents):
            amb = [constituents[j] for j in range(n) if j != i]
            _, kind, _ = core.least_sufficient_identifier(c, amb, c_sym=1.0, c_ac=10.0)
            if kind == "acoustic":
                invoked_acoustic.add(i)

        if invoked_acoustic == expected_acoustic:
            all_correct += 1
        rows.append({"n": n, "expected_residual": len(expected_acoustic),
                     "invoked_acoustic": len(invoked_acoustic)})
    return {
        "claim": "Cor 4.8: acoustic invoked exactly on the residual (unnamed or name-colliding)",
        "n_trials": n_trials,
        "trials_matching_exactly": all_correct,
        "pass": all_correct == n_trials,
        "sample": rows[:5],
    }


def run(seed=0):
    return {
        "e1_floor": e1_floor(seed=seed + 1),
        "e2_region": e2_region(seed=seed + 2),
        "e3_sufficiency": e3_sufficiency(seed=seed + 3),
        "e4_name_first_economy": e4_name_first(seed=seed + 4),
    }


if __name__ == "__main__":
    res = run()
    path = core.save_results("floor_and_lsi", res)
    for k, v in res.items():
        print(f"{k}: pass={v['pass']}")
    print("saved:", path)
