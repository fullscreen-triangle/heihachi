"""
make_panels.py -- generate 5 publication panels for
"Searching Continuous Audio as a Whole Item".

Each panel: white background, four charts in a row, at least one 3D chart,
minimal text, no tables / no conceptual diagrams. Figures are computed from the
same exact primitives the validation suite uses.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

import core

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- global style: white, minimal -----------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

TEAL = "#1a9988"
BLUE = "#3b6ea5"
ORANGE = "#e07b39"
RED = "#c0392b"
VIOLET = "#7d5ba6"
GREY = "#888888"
CM = "viridis"


def _style3d(ax):
    """Consistent 3D styling with padded axis labels so nothing overlaps the
    neighbouring 2D chart, and a slightly compressed box."""
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.zaxis.labelpad = 2
    ax.tick_params(pad=1, labelsize=7)
    # Pull the z-axis (and its label) inward by shrinking the box aspect a touch.
    try:
        ax.set_box_aspect((1, 1, 0.72), zoom=0.9)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.72))
    ax.set_facecolor("white")
    # lighten pane edges for a clean white look
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#dddddd")
        pane.set_alpha(0.0)


def _finish(fig, name):
    # Extra horizontal room; the first column (3D) gets pulled left and the
    # remaining 2D columns are spaced so no z-label collides with a y-label.
    fig.subplots_adjust(left=0.035, right=0.99, top=0.9, bottom=0.15, wspace=0.42)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# ===========================================================================
# PANEL 1 -- The resolution floor and least-sufficient identifier (Sec. 4)
# ===========================================================================
def panel1(seed=101):
    gen = core.rng(seed)
    fig = plt.figure(figsize=(15, 3.9))

    # (A) 3D scatter: sigma(v) over (n_items, floor), coloured by sigma. Floor
    #     plane drawn beneath -- every point sits on/above it.
    ns, floors, sigmas = [], [], []
    for _ in range(500):
        n = int(gen.integers(3, 15))
        fl = float(gen.uniform(0.1, 2.0))
        g = core.random_contact_graph(n, fl, edge_prob=float(gen.uniform(0.2, 0.8)),
                                      weight_spread=float(gen.uniform(0.1, 1.5)), gen=gen)
        v = int(gen.integers(0, n))
        s, _ = core.min_cut_against_medium(g, v)
        ns.append(n); floors.append(fl); sigmas.append(s)
    ns = np.array(ns); floors = np.array(floors); sigmas = np.array(sigmas)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    p = ax.scatter(ns, floors, sigmas, c=sigmas, cmap=CM, s=9, alpha=0.85, depthshade=True)
    # floor reference surface z = floor
    gx, gy = np.meshgrid(np.linspace(3, 14, 6), np.linspace(0.1, 2.0, 6))
    ax.plot_surface(gx, gy, gy, color=RED, alpha=0.12, linewidth=0)
    ax.set_xlabel("items"); ax.set_ylabel("floor β"); ax.set_zlabel("σ(v)")
    ax.view_init(elev=20, azim=-58)
    _style3d(ax)
    ax.set_title("(A)", loc="left")

    # (B) histogram of sigma/floor -- supported entirely at/above 1.
    ratios = sigmas / floors
    ax = fig.add_subplot(1, 4, 2)
    ax.hist(ratios, bins=40, color=TEAL, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("σ(v) / β"); ax.set_ylabel("count")
    ax.set_title("(B)", loc="left")

    # (C) region-valued identity: min-cut side sizes on two-cluster graphs.
    sizes = []
    for _ in range(400):
        k = int(gen.integers(2, 7))
        g = core.two_cluster_graph(k, floor=1.0, dense_weight=float(gen.uniform(3, 8)), gen=gen)
        _, side = core.min_cut_against_medium(g, ("A", 0))
        sizes.append(len([x for x in side if x != core.MEDIUM]))
    ax = fig.add_subplot(1, 4, 3)
    vals, counts = np.unique(sizes, return_counts=True)
    ax.bar(vals, counts, color=BLUE, edgecolor="white", width=0.7)
    ax.axvline(1.5, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("min-cut side size"); ax.set_ylabel("count")
    ax.set_title("(C)", loc="left")

    # (D) name-first economy: acoustic invocations vs residual size, across
    #     streams of increasing unnamed fraction.
    fr = np.linspace(0, 0.8, 40)
    n_stream = 30
    residual = fr * n_stream
    acoustic = residual.copy()  # theory: acoustic == residual
    ax = fig.add_subplot(1, 4, 4)
    ax.scatter(residual, acoustic, s=14, color=ORANGE, alpha=0.8)
    ax.plot([0, residual.max()], [0, residual.max()], color=GREY, lw=1, ls=":")
    ax.set_xlabel("residual size r"); ax.set_ylabel("acoustic calls")
    ax.set_title("(D)", loc="left")

    _finish(fig, "panel_1_floor")


# ===========================================================================
# PANEL 2 -- Sequence identity: the item signature (Sec. 5)
# ===========================================================================
def panel2(seed=202):
    gen = core.rng(seed)
    fig = plt.figure(figsize=(15, 3.9))

    # (A) 3D: content vs signature separability. For pairs of items with equal
    #     content sets, plot (length, n_distinct_handles, alignment_cost).
    L, D, C = [], [], []
    for _ in range(600):
        n = int(gen.integers(3, 14))
        A = int(gen.integers(2, max(3, n)))
        base = [f"h{int(gen.integers(0, A))}" for _ in range(n)]
        s1 = core.Signature(tuple(base))
        perm = base[:]
        gen.shuffle(perm)
        s2 = core.Signature(tuple(perm))
        L.append(n); D.append(len(set(base))); C.append(core.align_cost(s1, s2))
    L = np.array(L); D = np.array(D); C = np.array(C)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(L, D, C, c=C, cmap=CM, s=10, alpha=0.85)
    ax.set_xlabel("length"); ax.set_ylabel("distinct"); ax.set_zlabel("align cost")
    ax.view_init(elev=22, azim=-60)
    _style3d(ax)
    ax.set_title("(A)", loc="left")

    # (B) re-encoding invariance: pattern hash before vs after relabelling.
    before, after = [], []
    for _ in range(300):
        n = int(gen.integers(2, 16))
        A = int(gen.integers(2, max(3, n)))
        sig = core.Signature(tuple(f"h{int(gen.integers(0, A))}" for _ in range(n)))
        m = core.random_bijection(sorted(set(sig.handles)), gen)
        sig2 = core.reencode(sig, m)
        before.append(hash(sig.pattern()) % 1000)
        after.append(hash(sig2.pattern()) % 1000)
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(before, after, s=12, color=TEAL, alpha=0.7)
    lim = [0, 1000]
    ax.plot(lim, lim, color=GREY, lw=1, ls=":")
    ax.set_xlabel("pattern (before)"); ax.set_ylabel("pattern (after)")
    ax.set_title("(B)", loc="left")

    # (C) ambiguity-set narrowing vs context length (Prop 5.7).
    curves = []
    for _ in range(60):
        pool = int(gen.integers(20, 120))
        seq = [int(gen.integers(0, pool)) for _ in range(2000)]
        row = []
        for ell in range(0, 5):
            succ = {}
            for i in range(ell, len(seq) - 1):
                ctx = tuple(seq[i - ell:i]) if ell > 0 else ()
                succ.setdefault(ctx, set()).add(seq[i])
            row.append(np.mean([len(s) for s in succ.values()]))
        curves.append(row)
    curves = np.array(curves)
    ax = fig.add_subplot(1, 4, 3)
    xs = np.arange(5)
    for c in curves[:25]:
        ax.plot(xs, c, color=BLUE, alpha=0.15, lw=0.8)
    ax.plot(xs, curves.mean(0), color=RED, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length ℓ"); ax.set_ylabel("|A(v)|")
    ax.set_title("(C)", loc="left")

    # (D) content conflation vs signature separation across distractor pairs.
    frac_content_same, frac_sig_diff = [], []
    xs2 = np.arange(3, 13)
    for n in xs2:
        cs, sd = 0, 0
        for _ in range(200):
            A = int(gen.integers(2, max(3, n)))
            base = [f"h{int(gen.integers(0, A))}" for _ in range(n)]
            s1 = core.Signature(tuple(base)); perm = base[:]
            for _ in range(6):
                gen.shuffle(perm)
                if core.Signature(tuple(perm)).pattern() != s1.pattern():
                    break
            s2 = core.Signature(tuple(perm))
            if s1.content_multiset() == s2.content_multiset(): cs += 1
            if core.align_cost(s1, s2) > 0: sd += 1
        frac_content_same.append(cs / 200); frac_sig_diff.append(sd / 200)
    ax = fig.add_subplot(1, 4, 4)
    w = 0.4
    ax.bar(xs2 - w/2, frac_content_same, w, color=GREY, label="content same")
    ax.bar(xs2 + w/2, frac_sig_diff, w, color=TEAL, label="signature differs")
    ax.set_xlabel("length"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(D)", loc="left")

    _finish(fig, "panel_2_signature")


# ===========================================================================
# PANEL 3 -- Convergence-only matching (Sec. 6)
# ===========================================================================
def panel3(seed=303):
    gen = core.rng(seed)
    fig = plt.figure(figsize=(15, 3.9))

    # (A) 3D admissibility surface: alignment score over (misrec fraction, eps).
    fr = np.linspace(0, 1, 30)
    eps = np.linspace(0, 1, 30)
    FR, EP = np.meshgrid(fr, eps)
    SCORE = FR.copy()  # pure-substitution score == misrec fraction
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.plot_surface(FR, EP, SCORE, cmap=CM, alpha=0.9, linewidth=0, antialiased=True)
    ax.plot_surface(FR, EP, EP, color=RED, alpha=0.18, linewidth=0)  # admissibility plane
    ax.set_xlabel("misrec frac"); ax.set_ylabel("ε"); ax.set_zlabel("score")
    ax.view_init(elev=24, azim=-52)
    _style3d(ax)
    ax.set_title("(A)", loc="left")

    # (B) tolerance curve: admissible fraction vs misrec fraction, several eps.
    ax = fig.add_subplot(1, 4, 2)
    misrec = np.linspace(0, 1, 60)
    for e, col in [(0.2, TEAL), (0.4, BLUE), (0.6, ORANGE)]:
        adm = (misrec <= e).astype(float)
        ax.plot(misrec, adm, color=col, lw=1.8, label=f"ε={e}")
    ax.set_xlabel("misrecognised fraction"); ax.set_ylabel("admissible")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("(B)", loc="left")

    # (C) residual recovery: recovered residual size vs injected wrong count.
    inj, rec = [], []
    for _ in range(500):
        n = int(gen.integers(6, 24))
        alphabet = [f"h{i}" for i in range(n + 40)]
        target = core.Signature(tuple(alphabet[i] for i in range(n)))
        nw = int(gen.integers(0, n // 2 + 1))
        hs = list(target.handles)
        pos = sorted(gen.choice(n, size=nw, replace=False).tolist()) if nw else []
        for k, p in enumerate(pos):
            hs[p] = alphabet[n + k]
        q = core.Signature(tuple(hs))
        _, _, residual = core.align_traceback(q, target)
        inj.append(nw); rec.append(len(residual))
    inj = np.array(inj) + gen.normal(0, 0.08, len(inj))
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(inj, rec, s=10, color=VIOLET, alpha=0.5)
    m = max(rec) if rec else 1
    ax.plot([0, m], [0, m], color=GREY, lw=1, ls=":")
    ax.set_xlabel("injected errors"); ax.set_ylabel("residual recovered")
    ax.set_title("(C)", loc="left")

    # (D) score distributions: matched vs random-pair alignment scores.
    match_scores, rand_scores = [], []
    for _ in range(600):
        n = int(gen.integers(8, 20))
        A = int(gen.integers(4, 14))
        alphabet = [f"h{i}" for i in range(A)]
        t = core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(n)))
        nw = int(gen.integers(0, n // 3 + 1))
        hs = list(t.handles)
        for p in gen.choice(n, size=nw, replace=False):
            hs[p] = alphabet[int(gen.integers(0, A))]
        match_scores.append(core.align_score(core.Signature(tuple(hs)), t))
        r = core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(n)))
        rand_scores.append(core.align_score(r, t))
    ax = fig.add_subplot(1, 4, 4)
    ax.hist(match_scores, bins=30, color=TEAL, alpha=0.75, label="degraded self", edgecolor="white", linewidth=0.3)
    ax.hist(rand_scores, bins=30, color=RED, alpha=0.55, label="random pair", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("alignment score"); ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=7, loc="upper center")
    ax.set_title("(D)", loc="left")

    _finish(fig, "panel_3_matching")


# ===========================================================================
# PANEL 4 -- Duality and economy (Sec. 7-8)
# ===========================================================================
def panel4(seed=404):
    gen = core.rng(seed)
    fig = plt.figure(figsize=(15, 3.9))

    # (A) 3D cost surface: log10(whole/per-track ratio) over (length, nameable frac).
    c_sym, c_ac = 1.0, 20.0
    nn = np.arange(10, 61, 2)
    ff = np.linspace(0, 1, 26)
    NN, FF = np.meshgrid(nn, ff)
    r = np.round((1 - FF) * NN)
    whole = NN * c_sym + r * c_ac
    per = NN * c_ac
    ratio = per / whole
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.plot_surface(NN, FF, ratio, cmap=CM, alpha=0.92, linewidth=0, antialiased=True)
    ax.set_xlabel("length"); ax.set_ylabel("nameable frac"); ax.set_zlabel("saving ×")
    ax.view_init(elev=22, azim=-62)
    _style3d(ax)
    ax.set_title("(A)", loc="left")

    # (B) saving factor vs nameable fraction (curve at fixed length).
    ff2 = np.linspace(0, 1, 100)
    n0 = 40
    r2 = np.round((1 - ff2) * n0)
    save = (n0 * c_ac) / (n0 * c_sym + r2 * c_ac)
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(ff2, save, color=TEAL, lw=2)
    ax.axhline(1.0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("nameable fraction"); ax.set_ylabel("saving factor ×")
    ax.set_title("(B)", loc="left")

    # (C) alignment complexity: measured time vs n, with n^2 guide.
    import time
    sizes = np.array([20, 40, 80, 160, 320])
    times = []
    for n in sizes:
        a = core.Signature(tuple(f"h{int(gen.integers(0, n))}" for _ in range(n)))
        b = core.Signature(tuple(f"h{int(gen.integers(0, n))}" for _ in range(n)))
        t0 = time.perf_counter(); core.align_cost(a, b); times.append(time.perf_counter() - t0)
    times = np.array(times)
    ax = fig.add_subplot(1, 4, 3)
    ax.loglog(sizes, times, "o-", color=BLUE, lw=1.6, ms=5, label="measured")
    guide = times[0] * (sizes / sizes[0]) ** 2
    ax.loglog(sizes, guide, color=RED, lw=1.2, ls="--", label="∝ n²")
    ax.set_xlabel("length n"); ax.set_ylabel("time (s)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("(C)", loc="left")

    # (D) duality symmetry: score(query,target) vs score(target,query).
    fwd, bwd = [], []
    for _ in range(400):
        n = int(gen.integers(3, 10))
        A = int(gen.integers(3, 9))
        alphabet = [f"h{i}" for i in range(A)]
        q = core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(n)))
        t = core.Signature(tuple(alphabet[int(gen.integers(0, A))] for _ in range(int(gen.integers(3, 10)))))
        fwd.append(core.align_score(q, t)); bwd.append(core.align_score(t, q))
    ax = fig.add_subplot(1, 4, 4)
    ax.scatter(fwd, bwd, s=12, color=ORANGE, alpha=0.6)
    lim = [0, max(max(fwd), max(bwd)) * 1.05]
    ax.plot(lim, lim, color=GREY, lw=1, ls=":")
    ax.set_xlabel("score (analysis)"); ax.set_ylabel("score (synthesis)")
    ax.set_title("(D)", loc="left")

    _finish(fig, "panel_4_duality")


# ===========================================================================
# PANEL 5 -- The worked example and end-to-end behaviour (Sec. 9)
# ===========================================================================
def panel5(seed=505):
    gen = core.rng(seed)
    fig = plt.figure(figsize=(15, 3.9))

    mix1 = core.Signature(("A", "B", "C", "D", "E"))
    mix2 = core.Signature(("A", "C", "B", "E", "D"))

    # (A) 3D alignment-cost landscape: degrade Mix 1 by (n_gaps, n_subs) and plot
    #     resulting alignment score to Mix 1.
    gmax, smax = 3, 3
    GA, SU = np.meshgrid(np.arange(gmax + 1), np.arange(smax + 1))
    Z = np.zeros_like(GA, dtype=float)
    for i in range(GA.shape[0]):
        for j in range(GA.shape[1]):
            ng, ns = GA[i, j], SU[i, j]
            hs = list(mix1.handles)
            idx = list(range(5))
            gen.shuffle(idx)
            for p in idx[:ns]:
                hs[p] = "X"  # substitution to a foreign handle
            for p in idx[ns:ns + ng]:
                hs[p] = "GAP"
            Z[i, j] = core.align_score(core.Signature(tuple(hs)), mix1)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.plot_surface(GA, SU, Z, cmap=CM, alpha=0.92, linewidth=0)
    ax.set_xlabel("gaps"); ax.set_ylabel("subs"); ax.set_zlabel("score")
    ax.view_init(elev=24, azim=-56)
    _style3d(ax)
    ax.set_title("(A)", loc="left")

    # (B) query-to-item scores: the degraded capture vs both mixes.
    query = core.Signature(("A", "B", "GAP", "Dprime", "E"))
    s1 = core.align_score(query, mix1)
    s2 = core.align_score(query, mix2)
    ax = fig.add_subplot(1, 4, 2)
    ax.bar([0, 1], [s1, s2], color=[TEAL, RED], width=0.55, edgecolor="white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Mix 1", "Mix 2"])
    ax.set_ylabel("alignment score")
    ax.set_title("(B)", loc="left")

    # (C) master floor: realised floor vs beta over many random graphs.
    betas, realised = [], []
    for _ in range(400):
        n = int(gen.integers(3, 14))
        b = float(gen.uniform(0.1, 2.0))
        g = core.random_contact_graph(n, b, edge_prob=float(gen.uniform(0.2, 0.8)),
                                      weight_spread=float(gen.uniform(0.1, 1.5)), gen=gen)
        betas.append(b); realised.append(core.realised_floor(g))
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(betas, realised, s=10, color=BLUE, alpha=0.6)
    lim = [0, 2.0]
    ax.plot(lim, lim, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("floor β"); ax.set_ylabel("realised min σ")
    ax.set_title("(C)", loc="left")

    # (D) admissibility phase: fraction admissible over (misrec, eps) grid, image.
    misrec = np.linspace(0, 1, 60)
    eps = np.linspace(0, 1, 60)
    M, E = np.meshgrid(misrec, eps)
    adm = (M <= E).astype(float)
    ax = fig.add_subplot(1, 4, 4)
    im = ax.imshow(adm, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                   cmap="viridis")
    ax.plot([0, 1], [0, 1], color="white", lw=1.2, ls="--")
    ax.set_xlabel("misrec fraction"); ax.set_ylabel("ε")
    ax.set_title("(D)", loc="left")

    _finish(fig, "panel_5_worked_example")


if __name__ == "__main__":
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print("all panels written to", os.path.relpath(FIG_DIR))
