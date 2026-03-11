"""
Generate 7 panel charts for the Categorical Audio Transport paper.
Each panel: 4 charts in a row, at least one 3D, minimal text.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RESULTS_DIR = Path('yokozuna/public/results')
FIG_DIR = Path('publication/categorical-digital-audio/figures')
FIG_DIR.mkdir(exist_ok=True)

TRACKS = ['Audio_Omega', 'Spor_RunningMan', 'noisia_stigma', 'Angel_Techno']
LABELS = ['Audio Omega', 'Spor', 'Noisia', 'Angel Techno']
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
MARKERS = ['o', 's', '^', 'D']

def load_all():
    data = {}
    for track in TRACKS:
        with open(RESULTS_DIR / f'{track}_categorical_analysis.json') as f:
            data[track] = json.load(f)
    return data

data = load_all()


# =============================================================
# PANEL 1: Triple Equivalence
# =============================================================
def panel_1_triple_equivalence():
    fig = plt.figure(figsize=(16, 3.5))

    te = {t: data[t]['triple_equivalence'] for t in TRACKS}

    # 1a: Bar chart S_osc, S_cat, S_theory per track
    ax1 = fig.add_subplot(141)
    x = np.arange(4)
    w = 0.25
    s_osc = [te[t]['S_osc'] * 1e22 for t in TRACKS]
    s_cat = [te[t]['S_cat'] * 1e22 for t in TRACKS]
    s_the = [te[t]['S_theory'] * 1e22 for t in TRACKS]
    ax1.bar(x - w, s_osc, w, label=r'$S_{\mathrm{osc}}$', color='#1f77b4')
    ax1.bar(x, s_cat, w, label=r'$S_{\mathrm{cat}}$', color='#d62728')
    ax1.bar(x + w, s_the, w, label=r'$S_{\mathrm{theory}}$', color='#2ca02c', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=15, ha='right')
    ax1.set_ylabel(r'Entropy ($\times 10^{-22}$ J/K)')
    ax1.legend(loc='lower right')
    ax1.set_title('(a) Triple Equivalence')

    # 1b: Ratio S_osc/S_theory (should be 1.0)
    ax2 = fig.add_subplot(142)
    ratios = [te[t]['S_osc_over_theory'] for t in TRACKS]
    ax2.bar(x, ratios, color=COLORS, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylim(0.99, 1.01)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=15, ha='right')
    ax2.set_ylabel(r'$S_{\mathrm{osc}} / S_{\mathrm{theory}}$')
    ax2.set_title('(b) Ratio Verification')

    # 1c: 3D scatter — S_osc vs S_cat vs S_part
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        ax3.scatter(te[t]['S_osc'] * 1e22, te[t]['S_cat'] * 1e22, te[t]['S_part'] * 1e22,
                    c=COLORS[i], marker=MARKERS[i], s=80, label=LABELS[i], edgecolors='black', linewidths=0.5)
    # Identity line
    lim = [0.7, 0.8]
    ax3.plot(lim, lim, lim, 'k--', alpha=0.5, linewidth=0.8)
    ax3.set_xlabel(r'$S_{\mathrm{osc}}$', labelpad=2)
    ax3.set_ylabel(r'$S_{\mathrm{cat}}$', labelpad=2)
    ax3.set_zlabel(r'$S_{\mathrm{part}}$', labelpad=2)
    ax3.set_title('(c) Triple Equivalence Space')
    ax3.legend(loc='upper left', fontsize=6)

    # 1d: Shannon entropy vs oscillatory entropy rate
    ax4 = fig.add_subplot(144)
    h_osc = [te[t]['oscillatory_entropy_rate'] for t in TRACKS]
    h_cat = [te[t]['categorical_shannon_entropy'] for t in TRACKS]
    for i in range(4):
        ax4.scatter(h_osc[i], h_cat[i], c=COLORS[i], marker=MARKERS[i],
                    s=80, label=LABELS[i], edgecolors='black', linewidths=0.5, zorder=5)
    ax4.set_xlabel(r'$H_{\mathrm{osc}}$ (entropy rate)')
    ax4.set_ylabel(r'$H_{\mathrm{cat}}$ (Shannon entropy)')
    ax4.set_title('(d) Entropy Rate vs Shannon')
    ax4.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_1_triple_equivalence.png')
    fig.savefig(FIG_DIR / 'panel_1_triple_equivalence.pdf')
    plt.close(fig)
    print('Panel 1 done')


# =============================================================
# PANEL 2: Orthogonal Channel (Commutation)
# =============================================================
def panel_2_commutation():
    fig = plt.figure(figsize=(16, 3.5))

    te = {t: data[t]['triple_equivalence'] for t in TRACKS}

    # 2a: Bar chart commutation correlations
    ax1 = fig.add_subplot(141)
    x = np.arange(4)
    w = 0.35
    corr_amp = [abs(te[t]['commutation_test']['corr_cat_amplitude']) for t in TRACKS]
    corr_vel = [abs(te[t]['commutation_test']['corr_cat_velocity']) for t in TRACKS]
    ax1.bar(x - w/2, corr_amp, w, label=r'$|\mathrm{corr}(\mathrm{cat}, \mathrm{amp})|$', color='#1f77b4')
    ax1.bar(x + w/2, corr_vel, w, label=r'$|\mathrm{corr}(\mathrm{cat}, \mathrm{vel})|$', color='#d62728')
    ax1.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, label='threshold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=15, ha='right')
    ax1.set_ylabel('|Correlation|')
    ax1.set_title('(a) Commutation Test')
    ax1.legend(fontsize=6)

    # 2b: C_phys vs C_cat radar-like polar
    ax2 = fig.add_subplot(142, polar=True)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    c_phys = [data[t]['spectrometer']['C_phys_bits_per_s'] / 1e5 for t in TRACKS]
    c_cat = [data[t]['spectrometer']['C_cat_bits'] for t in TRACKS]
    # Normalize for display
    c_phys_norm = np.array(c_phys) / max(c_phys)
    c_cat_norm = np.array(c_cat) / max(c_cat)
    angles_closed = np.append(angles, angles[0])
    c_phys_closed = np.append(c_phys_norm, c_phys_norm[0])
    c_cat_closed = np.append(c_cat_norm, c_cat_norm[0])
    ax2.plot(angles_closed, c_phys_closed, 'o-', color='#1f77b4', label=r'$C_{\mathrm{phys}}$', linewidth=1.5)
    ax2.plot(angles_closed, c_cat_closed, 's-', color='#d62728', label=r'$C_{\mathrm{cat}}$', linewidth=1.5)
    ax2.set_xticks(angles)
    ax2.set_xticklabels(LABELS, fontsize=6)
    ax2.set_title('(b) Channel Capacity', pad=15)
    ax2.legend(loc='upper right', fontsize=6)

    # 2c: 3D — corr_amp vs corr_vel vs C_cat
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        ax3.scatter(corr_amp[i], corr_vel[i], c_cat[i],
                    c=COLORS[i], marker=MARKERS[i], s=80, label=LABELS[i],
                    edgecolors='black', linewidths=0.5)
    ax3.set_xlabel(r'$|\mathrm{corr_{amp}}|$', labelpad=2)
    ax3.set_ylabel(r'$|\mathrm{corr_{vel}}|$', labelpad=2)
    ax3.set_zlabel(r'$C_{\mathrm{cat}}$ (bits)', labelpad=2)
    ax3.set_title('(c) Orthogonality Space')

    # 2d: Enhancement over sample period
    ax4 = fig.add_subplot(144)
    enh = [data[t]['hardware_clock']['enhancement_over_sample_period'] for t in TRACKS]
    bars = ax4.bar(x, enh, color=COLORS, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(LABELS, rotation=15, ha='right')
    ax4.set_ylabel('Enhancement factor')
    ax4.set_title(r'(d) Clock Enhancement ($\times$ sample period)')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_2_commutation.png')
    fig.savefig(FIG_DIR / 'panel_2_commutation.pdf')
    plt.close(fig)
    print('Panel 2 done')


# =============================================================
# PANEL 3: Irreversibility / Second Law
# =============================================================
def panel_3_irreversibility():
    fig = plt.figure(figsize=(16, 3.5))

    ens = {t: data[t]['ensemble'] for t in TRACKS}

    # 3a: Positive vs negative transitions stacked bar
    ax1 = fig.add_subplot(141)
    x = np.arange(4)
    pos = [ens[t]['second_law']['n_positive'] for t in TRACKS]
    neg = [ens[t]['second_law']['n_negative'] for t in TRACKS]
    ax1.bar(x, pos, label='Positive ($\Delta S > 0$)', color='#2ca02c')
    ax1.bar(x, neg, bottom=pos, label='Negative ($\Delta S < 0$)', color='#d62728')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=15, ha='right')
    ax1.set_ylabel('Transitions')
    ax1.set_title('(a) Transition Direction')
    ax1.legend(fontsize=6)

    # 3b: Irreversibility fraction
    ax2 = fig.add_subplot(142)
    irrev = [ens[t]['irreversibility_fraction'] for t in TRACKS]
    bars = ax2.bar(x, irrev, color=COLORS, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, label='Equilibrium')
    ax2.set_ylim(0.35, 0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=15, ha='right')
    ax2.set_ylabel('Irreversibility fraction')
    ax2.set_title('(b) Irreversibility')
    ax2.legend(fontsize=6)

    # 3c: 3D — irreversibility vs entropy_prod vs total_entropy
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        spec = data[t]['spectrometer']
        ax3.scatter(irrev[i], spec['entropy_production'] * 1e19,
                    spec['total_entropy'] * 1e17,
                    c=COLORS[i], marker=MARKERS[i], s=80, label=LABELS[i],
                    edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Irrev. fraction', labelpad=2)
    ax3.set_ylabel(r'$\Sigma_{\mathrm{prod}}$ ($\times 10^{-19}$)', labelpad=2)
    ax3.set_zlabel(r'$S_{\mathrm{total}}$ ($\times 10^{-17}$)', labelpad=2)
    ax3.set_title('(c) Thermodynamic Space')

    # 3d: Total entropy production
    ax4 = fig.add_subplot(144)
    prod = [ens[t]['second_law']['total_production'] * 1e19 for t in TRACKS]
    ax4.bar(x, prod, color=COLORS, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(LABELS, rotation=15, ha='right')
    ax4.set_ylabel(r'$\Sigma_{\mathrm{prod}}$ ($\times 10^{-19}$ J/K)')
    ax4.set_title(r'(d) Total Entropy Production')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_3_irreversibility.png')
    fig.savefig(FIG_DIR / 'panel_3_irreversibility.pdf')
    plt.close(fig)
    print('Panel 3 done')


# =============================================================
# PANEL 4: Harmonic Coincidence Network
# =============================================================
def panel_4_harmonics():
    fig = plt.figure(figsize=(16, 3.5))

    harm = {t: data[t]['harmonics'] for t in TRACKS}

    # 4a: Mode frequency spectra (scatter by amplitude)
    ax1 = fig.add_subplot(141)
    for i, t in enumerate(TRACKS):
        freqs = np.array(harm[t]['mode_frequencies'])
        amps = np.array(harm[t]['mode_amplitudes'])
        # Normalize amps for marker size
        amps_norm = amps / amps.max() * 60 + 10
        ax1.scatter(freqs, np.ones_like(freqs) * i, s=amps_norm,
                    c=COLORS[i], alpha=0.7, edgecolors='black', linewidths=0.3)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(LABELS)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_title('(a) Mode Frequencies')
    ax1.set_xlim(0, 20000)

    # 4b: Harmonic relations & clusters
    ax2 = fig.add_subplot(142)
    x = np.arange(4)
    w = 0.35
    rels = [harm[t]['n_relations'] for t in TRACKS]
    clust = [harm[t]['n_clusters'] for t in TRACKS]
    ax2.bar(x - w/2, rels, w, label='Relations', color='#1f77b4')
    ax2b = ax2.twinx()
    ax2b.bar(x + w/2, clust, w, label='Clusters', color='#d62728')
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=15, ha='right')
    ax2.set_ylabel('Harmonic Relations', color='#1f77b4')
    ax2b.set_ylabel('Clusters', color='#d62728')
    ax2.set_title('(b) Network Structure')

    # 4c: 3D — frequency vs amplitude vs mode index
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        freqs = np.array(harm[t]['mode_frequencies'])
        amps = np.array(harm[t]['mode_amplitudes'])
        modes = np.arange(len(freqs))
        ax3.scatter(freqs, np.log10(amps), modes,
                    c=COLORS[i], marker=MARKERS[i], s=30, alpha=0.7, label=LABELS[i])
    ax3.set_xlabel('Freq (Hz)', labelpad=2)
    ax3.set_ylabel(r'$\log_{10}$(Amp)', labelpad=2)
    ax3.set_zlabel('Mode index', labelpad=2)
    ax3.set_title('(c) Mode Distribution')
    ax3.legend(fontsize=5, loc='upper left')

    # 4d: Enhancement (log scale)
    ax4 = fig.add_subplot(144)
    enh = [harm[t]['total_enhancement'] for t in TRACKS]
    log_enh = [np.log10(e) if e > 0 else 0 for e in enh]
    ax4.bar(x, log_enh, color=COLORS, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(LABELS, rotation=15, ha='right')
    ax4.set_ylabel(r'$\log_{10}$(Enhancement)')
    ax4.set_title('(d) Coincidence Enhancement')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_4_harmonics.png')
    fig.savefig(FIG_DIR / 'panel_4_harmonics.pdf')
    plt.close(fig)
    print('Panel 4 done')


# =============================================================
# PANEL 5: S-Entropy Coordinates
# =============================================================
def panel_5_s_entropy():
    fig = plt.figure(figsize=(16, 3.5))

    # 5a: Mean S-entropy grouped bar
    ax1 = fig.add_subplot(141)
    x = np.arange(4)
    w = 0.25
    sk = [data[t]['ensemble']['s_entropy_mean']['S_k'] * 1e22 for t in TRACKS]
    st = [data[t]['ensemble']['s_entropy_mean']['S_t'] * 1e22 for t in TRACKS]
    se = [data[t]['ensemble']['s_entropy_mean']['S_e'] * 1e22 for t in TRACKS]
    ax1.bar(x - w, sk, w, label=r'$\langle S_k \rangle$', color='#1f77b4')
    ax1.bar(x, [s * 10 for s in st], w, label=r'$\langle S_t \rangle \times 10$', color='#d62728')
    ax1.bar(x + w, se, w, label=r'$\langle S_e \rangle$', color='#2ca02c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=15, ha='right')
    ax1.set_ylabel(r'Entropy ($\times 10^{-22}$ J/K)')
    ax1.set_title(r'(a) Mean $\vec{S}$ Components')
    ax1.legend(fontsize=6)

    # 5b: S-entropy standard deviations
    ax2 = fig.add_subplot(142)
    sk_std = [data[t]['ensemble']['s_entropy_std']['S_k'] * 1e24 for t in TRACKS]
    st_std = [data[t]['ensemble']['s_entropy_std']['S_t'] * 1e23 for t in TRACKS]
    se_std = [data[t]['ensemble']['s_entropy_std']['S_e'] * 1e23 for t in TRACKS]
    ax2.bar(x - w, sk_std, w, label=r'$\sigma_{S_k}$ ($\times 10^{-24}$)', color='#1f77b4')
    ax2.bar(x, st_std, w, label=r'$\sigma_{S_t}$ ($\times 10^{-23}$)', color='#d62728')
    ax2.bar(x + w, se_std, w, label=r'$\sigma_{S_e}$ ($\times 10^{-23}$)', color='#2ca02c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=15, ha='right')
    ax2.set_ylabel('Std. deviation')
    ax2.set_title(r'(b) $\vec{S}$ Fluctuations')
    ax2.legend(fontsize=5)

    # 5c: 3D scatter — S_k vs S_t vs S_e means
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        ax3.scatter(sk[i], st[i] * 10, se[i],  # scale S_t for visibility
                    c=COLORS[i], marker=MARKERS[i], s=100, label=LABELS[i],
                    edgecolors='black', linewidths=0.5)
    ax3.set_xlabel(r'$\langle S_k \rangle$', labelpad=2)
    ax3.set_ylabel(r'$\langle S_t \rangle \times 10$', labelpad=2)
    ax3.set_zlabel(r'$\langle S_e \rangle$', labelpad=2)
    ax3.set_title('(c) S-Entropy Space')
    ax3.legend(fontsize=6)

    # 5d: Occupation entropy
    ax4 = fig.add_subplot(144)
    occ = [data[t]['ensemble']['occupation_entropy'] for t in TRACKS]
    ax4.bar(x, occ, color=COLORS, edgecolor='black', linewidth=0.5)
    ax4.set_ylim(6.40, 6.42)
    ax4.set_xticks(x)
    ax4.set_xticklabels(LABELS, rotation=15, ha='right')
    ax4.set_ylabel('Occupation Entropy (nats)')
    ax4.set_title('(d) State Occupation')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_5_s_entropy.png')
    fig.savefig(FIG_DIR / 'panel_5_s_entropy.pdf')
    plt.close(fig)
    print('Panel 5 done')


# =============================================================
# PANEL 6: Entropy Trajectory
# =============================================================
def panel_6_trajectory():
    fig = plt.figure(figsize=(16, 3.5))

    # 6a: S_k trajectory over time (all tracks)
    ax1 = fig.add_subplot(141)
    for i, t in enumerate(TRACKS):
        traj = data[t]['s_entropy_trajectory_sampled']
        times = np.array(traj['times'])
        s_k = np.array(traj['S_k']) * 1e22
        ax1.plot(times, s_k, color=COLORS[i], alpha=0.8, linewidth=0.6, label=LABELS[i])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$S_k$ ($\times 10^{-22}$ J/K)')
    ax1.set_title(r'(a) $S_k$ Trajectory')
    ax1.legend(fontsize=6)

    # 6b: S_e trajectory
    ax2 = fig.add_subplot(142)
    for i, t in enumerate(TRACKS):
        traj = data[t]['s_entropy_trajectory_sampled']
        times = np.array(traj['times'])
        s_e = np.array(traj['S_e']) * 1e22
        ax2.plot(times, s_e, color=COLORS[i], alpha=0.8, linewidth=0.6, label=LABELS[i])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$S_e$ ($\times 10^{-22}$ J/K)')
    ax2.set_title(r'(b) $S_e$ Trajectory')
    ax2.legend(fontsize=6)

    # 6c: 3D trajectory S_k vs S_t vs S_e (first track)
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        traj = data[t]['s_entropy_trajectory_sampled']
        s_k = np.array(traj['S_k']) * 1e22
        s_t = np.array(traj['S_t']) * 1e22
        s_e = np.array(traj['S_e']) * 1e22
        # Subsample for clarity
        step = max(1, len(s_k) // 100)
        ax3.plot(s_k[::step], s_t[::step], s_e[::step],
                 color=COLORS[i], alpha=0.6, linewidth=0.5, label=LABELS[i])
    ax3.set_xlabel(r'$S_k$', labelpad=2)
    ax3.set_ylabel(r'$S_t$', labelpad=2)
    ax3.set_zlabel(r'$S_e$', labelpad=2)
    ax3.set_title('(c) S-Entropy Trajectories')
    ax3.legend(fontsize=5)

    # 6d: S_t trajectory
    ax4 = fig.add_subplot(144)
    for i, t in enumerate(TRACKS):
        traj = data[t]['s_entropy_trajectory_sampled']
        times = np.array(traj['times'])
        s_t = np.array(traj['S_t']) * 1e22
        ax4.plot(times, s_t, color=COLORS[i], alpha=0.8, linewidth=0.6, label=LABELS[i])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel(r'$S_t$ ($\times 10^{-22}$ J/K)')
    ax4.set_title(r'(d) $S_t$ Trajectory')
    ax4.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_6_trajectory.png')
    fig.savefig(FIG_DIR / 'panel_6_trajectory.pdf')
    plt.close(fig)
    print('Panel 6 done')


# =============================================================
# PANEL 7: Channel Capacity
# =============================================================
def panel_7_capacity():
    fig = plt.figure(figsize=(16, 3.5))

    # 7a: C_phys vs C_cat stacked bar
    ax1 = fig.add_subplot(141)
    x = np.arange(4)
    c_phys = [data[t]['spectrometer']['C_phys_bits_per_s'] / 1e3 for t in TRACKS]
    c_cat = [data[t]['spectrometer']['C_cat_bits'] for t in TRACKS]
    ax1.bar(x, c_phys, label=r'$C_{\mathrm{phys}}$ (kbits/s)', color='#1f77b4')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=15, ha='right')
    ax1.set_ylabel('Capacity (kbits/s)')
    ax1.set_title(r'(a) Physical Channel $C_{\mathrm{phys}}$')

    # 7b: Categorical capacity (constant but show it)
    ax2 = fig.add_subplot(142)
    c_total = [data[t]['spectrometer']['C_total'] / 1e3 for t in TRACKS]
    cat_frac = [data[t]['spectrometer']['categorical_fraction'] * 100 for t in TRACKS]
    ax2.bar(x, c_total, color=COLORS, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=15, ha='right')
    ax2.set_ylabel(r'$C_{\mathrm{total}}$ (kbits/s)')
    ax2.set_title(r'(b) Total Channel Capacity')

    # 7c: 3D — C_phys vs C_cat vs enhancement_factor
    ax3 = fig.add_subplot(143, projection='3d')
    for i, t in enumerate(TRACKS):
        spec = data[t]['spectrometer']
        enh_log = np.log10(spec['enhancement_factor'])
        ax3.scatter(spec['C_phys_bits_per_s'] / 1e3, spec['C_cat_bits'],
                    enh_log, c=COLORS[i], marker=MARKERS[i], s=80,
                    label=LABELS[i], edgecolors='black', linewidths=0.5)
    ax3.set_xlabel(r'$C_{\mathrm{phys}}$ (kbits/s)', labelpad=2)
    ax3.set_ylabel(r'$C_{\mathrm{cat}}$ (bits)', labelpad=2)
    ax3.set_zlabel(r'$\log_{10}$(Enhancement)', labelpad=2)
    ax3.set_title('(c) Capacity-Enhancement Space')
    ax3.legend(fontsize=6)

    # 7d: Entropy rate comparison
    ax4 = fig.add_subplot(144)
    ent_rate = [data[t]['ensemble']['entropy_rate'] * 1e21 for t in TRACKS]
    ent = [data[t]['ensemble']['entropy'] * 1e21 for t in TRACKS]
    ax4.bar(x - 0.2, ent, 0.35, label=r'$S$ ($\times 10^{-21}$)', color='#1f77b4')
    ax4.bar(x + 0.2, ent_rate, 0.35, label=r'$\dot{S}$ ($\times 10^{-21}$)', color='#d62728')
    ax4.set_xticks(x)
    ax4.set_xticklabels(LABELS, rotation=15, ha='right')
    ax4.set_ylabel(r'J/K ($\times 10^{-21}$)')
    ax4.set_title('(d) Entropy vs Entropy Rate')
    ax4.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'panel_7_capacity.png')
    fig.savefig(FIG_DIR / 'panel_7_capacity.pdf')
    plt.close(fig)
    print('Panel 7 done')


# =============================================================
# Generate all
# =============================================================
if __name__ == '__main__':
    panel_1_triple_equivalence()
    panel_2_commutation()
    panel_3_irreversibility()
    panel_4_harmonics()
    panel_5_s_entropy()
    panel_6_trajectory()
    panel_7_capacity()
    print('\nAll 7 panels generated in:', FIG_DIR)
