"""
Yokozuna Categorical Audio Analysis Pipeline
Usage: python yokozuna/run_analysis.py <track_path> [analysis_duration_seconds]
"""
import sys, os, json, time, wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transplanckian.virtual_spectrometer import VirtualSpectrometer, HardwareClock
from transplanckian.harmonics import HarmonicCoincidenceNetwork
from transplanckian.ensemble import CategoricalEnsemble
from equivalence.oscillatory import OscillatoryEntropy
from equivalence.categorical import CategoricalEntropy
from equivalence.partition import PartitionEntropy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dt = dtype_map[sw]
    data = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if n_ch == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    data /= np.iinfo(dt).max
    return data, sr


def run_analysis(track_path, analysis_dur=30.0):
    track_name = os.path.splitext(os.path.basename(track_path))[0]

    print("=" * 70)
    print(f"YOKOZUNA CATEGORICAL AUDIO ANALYSIS")
    print(f"Target: {track_name}")
    print("=" * 70)

    # Load
    data, sr = load_wav(track_path)
    total_dur = len(data) / sr
    print(f"\nLoaded: {len(data)} samples, sr={sr}, duration={total_dur:.1f}s")
    print(f"Peak amplitude: {np.max(np.abs(data)):.4f}")
    print(f"RMS: {np.sqrt(np.mean(data**2)):.4f}")

    n_analysis = min(int(analysis_dur * sr), len(data))
    signal = data[:n_analysis]
    actual_dur = n_analysis / sr
    print(f"Analyzing first {actual_dur:.1f}s ({n_analysis} samples)")

    results = {
        'track_name': track_name,
        'track_path': track_path,
        'sample_rate': sr,
        'total_duration_s': total_dur,
        'analysis_duration_s': actual_dur,
        'n_samples_analyzed': n_analysis,
        'peak_amplitude': float(np.max(np.abs(data))),
        'rms': float(np.sqrt(np.mean(data**2))),
    }

    # 1. HARDWARE CLOCK
    print("\n" + "-" * 50)
    print("1. HARDWARE CLOCK")
    print("-" * 50)
    clock = HardwareClock(n_states=32)
    sample_period_ns = (1.0 / sr) * 1e9
    enhancement_over_sample = sample_period_ns / clock.categorical_resolution_ns
    print(f"Hardware resolution: {clock.hardware_resolution_ns:.1f} ns")
    print(f"Categorical resolution: {clock.categorical_resolution_ns:.3f} ns")
    print(f"Enhancement over sample period: {enhancement_over_sample:.0f}x")

    results['hardware_clock'] = {
        'hardware_resolution_ns': clock.hardware_resolution_ns,
        'categorical_resolution_ns': clock.categorical_resolution_ns,
        'enhancement_over_sample_period': enhancement_over_sample,
    }

    # 2. VIRTUAL SPECTROMETER
    print("\n" + "-" * 50)
    print("2. VIRTUAL SPECTROMETER")
    print("-" * 50)
    t0 = time.time()
    spec = VirtualSpectrometer(sample_rate=sr, n_modes=32, partition_depth=32)
    measurement = spec.measure(signal, hop_size=2048)
    t_meas = time.time() - t0

    c_cat = spec.categorical_capacity(n_modes=measurement.n_modes)
    c_phys = spec.physical_capacity(snr_db=96.0)

    print(f"Measurement time: {t_meas:.2f}s")
    print(f"Modes: {measurement.n_modes}, Trajectory: {len(measurement.trajectory)} pts")
    print(f"Total entropy: {measurement.total_entropy:.4e} J/K")
    print(f"Enhancement: {measurement.enhancement_factor:.4e}")
    print(f"C_phys={c_phys:.0f} bits/s, C_cat={c_cat:.1f} bits")

    results['spectrometer'] = {
        'measurement_time_s': t_meas,
        'n_modes': measurement.n_modes,
        'partition_depth': measurement.partition_depth,
        'trajectory_points': len(measurement.trajectory),
        'transitions': len(measurement.transitions),
        'total_entropy': measurement.total_entropy,
        'entropy_production': measurement.entropy_production,
        'enhancement_factor': float(measurement.enhancement_factor),
        'C_phys_bits_per_s': c_phys,
        'C_cat_bits': c_cat,
        'C_total': c_phys + c_cat,
        'categorical_fraction': c_cat / (c_phys + c_cat),
    }

    # 3. ENSEMBLE
    print("\n" + "-" * 50)
    print("3. CATEGORICAL ENSEMBLE")
    print("-" * 50)
    ensemble = CategoricalEnsemble(measurement)
    stats = ensemble.compute_statistics()
    law = ensemble.categorical_second_law_verification()
    occ_entropy = ensemble.occupation_entropy()
    n_occupied = len(ensemble.state_occupation_histogram())

    print(f"Omega = 32^{stats.n_modes} = {stats.total_states:.2e}")
    print(f"Entropy: {stats.entropy:.4e} J/K")
    print(f"Irreversibility: {stats.irreversibility_fraction:.1%}")
    print(f"States occupied: {n_occupied}, Occupation H: {occ_entropy:.4f}")
    print(f"Second law: {law['n_positive']}/{law['n_transitions']} positive ({law['fraction_positive']:.1%})")

    results['ensemble'] = {
        'total_states': float(stats.total_states),
        'entropy': stats.entropy,
        'categorical_capacity_bits': stats.categorical_capacity,
        'entropy_rate': stats.entropy_rate,
        'irreversibility_fraction': stats.irreversibility_fraction,
        's_entropy_mean': {
            'S_k': stats.mean_s_entropy.S_k,
            'S_t': stats.mean_s_entropy.S_t,
            'S_e': stats.mean_s_entropy.S_e,
        },
        's_entropy_std': {
            'S_k': stats.std_s_entropy.S_k,
            'S_t': stats.std_s_entropy.S_t,
            'S_e': stats.std_s_entropy.S_e,
        },
        'occupation_entropy': occ_entropy,
        'distinct_states_occupied': n_occupied,
        'second_law': law,
    }

    # 4. HARMONIC COINCIDENCE
    print("\n" + "-" * 50)
    print("4. HARMONIC COINCIDENCE NETWORK")
    print("-" * 50)
    freqs_list, amps_list, seen = [], [], set()
    for state in measurement.trajectory:
        if state.mode_index not in seen:
            freqs_list.append(state.frequency)
            amps_list.append(state.amplitude)
            seen.add(state.mode_index)

    freqs_arr = np.array(freqs_list)
    amps_arr = np.array(amps_list)

    hcn = HarmonicCoincidenceNetwork(tolerance=0.03, min_strength=0.05)
    relations = hcn.detect_harmonics(freqs_arr, amps_arr)
    clusters = hcn.build_clusters(freqs_arr, amps_arr, relations)
    total_enh = hcn.total_coincidence_enhancement(freqs_arr, amps_arr)

    print(f"Modes: {len(freqs_arr)}, Relations: {len(relations)}, Clusters: {len(clusters)}")
    print(f"Total enhancement: {total_enh:.2e}")

    cluster_data = []
    for i, cl in enumerate(clusters[:10]):
        cinfo = {
            'fundamental_freq': cl.fundamental_freq,
            'n_members': len(cl.member_indices),
            'harmonic_numbers': cl.harmonic_numbers,
            'enhancement': cl.coincidence_enhancement,
        }
        cluster_data.append(cinfo)
        if i < 3:
            print(f"  Cluster {i+1}: f0={cl.fundamental_freq:.1f}Hz, {len(cl.member_indices)} modes, enh={cl.coincidence_enhancement:.2e}")

    results['harmonics'] = {
        'n_relations': len(relations),
        'n_clusters': len(clusters),
        'total_enhancement': float(total_enh),
        'mode_frequencies': freqs_arr.tolist(),
        'mode_amplitudes': amps_arr.tolist(),
        'clusters': cluster_data,
    }

    # 5. TRIPLE EQUIVALENCE
    print("\n" + "-" * 50)
    print("5. TRIPLE EQUIVALENCE")
    print("-" * 50)
    osc_ent = OscillatoryEntropy(partition_depth=32, sample_rate=sr)
    cat_ent = CategoricalEntropy(partition_depth=32, sample_rate=sr)
    part_ent = PartitionEntropy(partition_depth=32, sample_rate=sr)

    seg = signal[:sr]
    n_modes_eq = 16
    k_B = 1.380649e-23

    S_osc = osc_ent.compute_entropy(seg, n_modes=n_modes_eq)
    S_cat = cat_ent.compute_entropy(seg, n_modes=n_modes_eq)
    S_part = part_ent.compute_entropy(seg, frequency=440.0, n_modes=n_modes_eq)
    S_theory = k_B * n_modes_eq * np.log(32)

    H_osc = osc_ent.entropy_rate(seg)
    H_cat = cat_ent.shannon_entropy(seg)

    phys_obs = np.abs(seg)
    corr_amp = cat_ent.commutation_test(seg, phys_obs)
    velocity = np.diff(seg, prepend=seg[0])
    corr_vel = cat_ent.commutation_test(seg, velocity)

    print(f"S_osc  = {S_osc:.6e},  S_osc/S_theory = {S_osc/S_theory:.4f}")
    print(f"S_cat  = {S_cat:.6e},  S_cat/S_theory = {S_cat/S_theory:.4f}")
    print(f"S_part = {S_part:.6e}")
    print(f"S_osc == S_cat: {abs(S_osc - S_cat) < 1e-30}")
    print(f"Commutation: |corr(cat,amp)|={corr_amp:.4f}, |corr(cat,vel)|={corr_vel:.4f}")

    results['triple_equivalence'] = {
        'n_modes': n_modes_eq,
        'partition_depth': 32,
        'S_osc': S_osc,
        'S_cat': S_cat,
        'S_part': S_part,
        'S_theory': S_theory,
        'S_osc_over_theory': S_osc / S_theory,
        'S_cat_over_theory': S_cat / S_theory,
        'S_osc_equals_S_cat': bool(abs(S_osc - S_cat) < 1e-30),
        'oscillatory_entropy_rate': H_osc,
        'categorical_shannon_entropy': H_cat,
        'commutation_test': {
            'corr_cat_amplitude': corr_amp,
            'corr_cat_velocity': corr_vel,
        },
    }

    # 6. S-ENTROPY TRAJECTORY (sampled for JSON size)
    s_traj = ensemble.s_entropy_trajectory()
    times = ensemble.time_series()
    # Downsample to at most 500 points for JSON
    if len(times) > 500:
        idx = np.linspace(0, len(times) - 1, 500, dtype=int)
        results['s_entropy_trajectory_sampled'] = {
            'times': times[idx].tolist(),
            'S_k': s_traj[idx, 0].tolist(),
            'S_t': s_traj[idx, 1].tolist(),
            'S_e': s_traj[idx, 2].tolist(),
        }
    else:
        results['s_entropy_trajectory_sampled'] = {
            'times': times.tolist(),
            'S_k': s_traj[:, 0].tolist(),
            'S_t': s_traj[:, 1].tolist(),
            'S_e': s_traj[:, 2].tolist(),
        }

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'public', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{track_name}_categorical_analysis.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 70)
    print(f"RESULTS SAVED: {output_path}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <track_path> [duration_seconds]")
        sys.exit(1)

    track = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    run_analysis(track, analysis_dur=dur)
