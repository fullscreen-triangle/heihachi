"""
Yokozuna Format Encoder — Converts WAV + categorical analysis into .ykz container.

File structure (zip-based):
  manifest.json        — format version, track metadata, analysis parameters
  audio.pcm            — raw PCM samples (float32, mono)
  trajectory.bin       — compact binary categorical state trajectory
  entropy.bin          — S-entropy coordinate stream (float64 x 3)
  partitions.bin       — partition coordinate stream (int32 x 4)
  harmonics.json       — harmonic coincidence network
  analysis.json        — full analysis results (same as pipeline output)
"""

import json
import struct
import zipfile
import io
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transplanckian.virtual_spectrometer import (
    VirtualSpectrometer, HardwareClock, CategoricalMeasurement, CategoricalState
)
from transplanckian.harmonics import HarmonicCoincidenceNetwork
from transplanckian.ensemble import CategoricalEnsemble
from equivalence.oscillatory import OscillatoryEntropy
from equivalence.categorical import CategoricalEntropy
from equivalence.partition import PartitionEntropy

FORMAT_VERSION = "1.0.0"
FORMAT_MAGIC = b"YKZ\x01"  # 4-byte magic number


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


class YokozunaEncoder:
    """Encodes audio into the Yokozuna (.ykz) categorical format.

    The encoder runs the full categorical analysis pipeline and packages
    both the physical signal and categorical channel data into a single
    container file.
    """

    def __init__(
        self,
        n_modes: int = 32,
        partition_depth: int = 32,
        hop_size: int = 2048,
    ):
        self.n_modes = n_modes
        self.partition_depth = partition_depth
        self.hop_size = hop_size

    def encode(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        analysis_duration: Optional[float] = None,
    ) -> str:
        """Encode a WAV file into .ykz format.

        Args:
            input_path: Path to input WAV file.
            output_path: Path for output .ykz file. If None, uses same
                         directory and name as input with .ykz extension.
            analysis_duration: Max seconds to analyze. None = full file.

        Returns:
            Path to the created .ykz file.
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix('.ykz')
        else:
            output_path = Path(output_path)

        # --- Load audio ---
        signal, sr = self._load_wav(str(input_path))
        total_duration = len(signal) / sr

        if analysis_duration is not None:
            n_analysis = min(int(analysis_duration * sr), len(signal))
        else:
            n_analysis = len(signal)
        analysis_signal = signal[:n_analysis]
        actual_duration = n_analysis / sr

        print(f"[YKZ] Encoding: {input_path.name}")
        print(f"[YKZ] {len(signal)} samples, sr={sr}, duration={total_duration:.1f}s")
        print(f"[YKZ] Analyzing {actual_duration:.1f}s ({n_analysis} samples)")

        # --- Hardware clock ---
        clock = HardwareClock(n_states=self.partition_depth)

        # --- Virtual spectrometer measurement ---
        print("[YKZ] Running virtual spectrometer...")
        spec = VirtualSpectrometer(
            sample_rate=sr,
            n_modes=self.n_modes,
            partition_depth=self.partition_depth,
        )
        measurement = spec.measure(analysis_signal, hop_size=self.hop_size)

        # --- Ensemble statistics ---
        print("[YKZ] Computing ensemble statistics...")
        ensemble = CategoricalEnsemble(measurement)
        stats = ensemble.compute_statistics()
        law = ensemble.categorical_second_law_verification()

        # --- Harmonic coincidence ---
        print("[YKZ] Detecting harmonic coincidences...")
        freqs, amps, seen = [], [], set()
        for state in measurement.trajectory:
            if state.mode_index not in seen:
                freqs.append(state.frequency)
                amps.append(state.amplitude)
                seen.add(state.mode_index)
        freqs_arr = np.array(freqs)
        amps_arr = np.array(amps)

        hcn = HarmonicCoincidenceNetwork(tolerance=0.03, min_strength=0.05)
        relations = hcn.detect_harmonics(freqs_arr, amps_arr)
        clusters = hcn.build_clusters(freqs_arr, amps_arr, relations)
        total_enh = hcn.total_coincidence_enhancement(freqs_arr, amps_arr)

        # --- Triple equivalence ---
        print("[YKZ] Verifying triple equivalence...")
        osc_ent = OscillatoryEntropy(partition_depth=self.partition_depth, sample_rate=sr)
        cat_ent = CategoricalEntropy(partition_depth=self.partition_depth, sample_rate=sr)
        part_ent = PartitionEntropy(partition_depth=self.partition_depth, sample_rate=sr)

        seg = analysis_signal[:sr]  # first second
        n_modes_eq = min(16, self.n_modes)
        k_B = 1.380649e-23

        S_osc = osc_ent.compute_entropy(seg, n_modes=n_modes_eq)
        S_cat = cat_ent.compute_entropy(seg, n_modes=n_modes_eq)
        S_part = part_ent.compute_entropy(seg, frequency=440.0, n_modes=n_modes_eq)
        S_theory = k_B * n_modes_eq * np.log(self.partition_depth)

        # --- S-entropy trajectory ---
        s_traj = ensemble.s_entropy_trajectory()
        times = ensemble.time_series()

        # --- Build manifest ---
        manifest = {
            "format": "yokozuna",
            "version": FORMAT_VERSION,
            "track_name": input_path.stem,
            "source_file": input_path.name,
            "sample_rate": sr,
            "n_samples": len(signal),
            "total_duration_s": total_duration,
            "analysis_duration_s": actual_duration,
            "n_samples_analyzed": n_analysis,
            "encoding": {
                "n_modes": self.n_modes,
                "partition_depth": self.partition_depth,
                "hop_size": self.hop_size,
                "degeneracy": 2 * self.partition_depth ** 2,
            },
            "clock": {
                "hardware_resolution_ns": clock.hardware_resolution_ns,
                "categorical_resolution_ns": clock.categorical_resolution_ns,
            },
            "channels": {
                "physical": {
                    "type": "PCM_float32_mono",
                    "C_phys_bits_per_s": spec.physical_capacity(snr_db=96.0),
                },
                "categorical": {
                    "type": "trajectory_binary",
                    "C_cat_bits": spec.categorical_capacity(n_modes=self.n_modes),
                    "trajectory_points": len(measurement.trajectory),
                    "transitions": len(measurement.transitions),
                },
            },
            "triple_equivalence": {
                "S_osc": S_osc,
                "S_cat": S_cat,
                "S_part": S_part,
                "S_theory": S_theory,
                "verified": bool(abs(S_osc - S_cat) < 1e-30),
            },
        }

        # --- Build analysis JSON (full results, same as pipeline) ---
        analysis = self._build_analysis_json(
            input_path, signal, sr, total_duration, actual_duration, n_analysis,
            clock, measurement, spec, stats, law, ensemble,
            freqs_arr, amps_arr, relations, clusters, total_enh,
            S_osc, S_cat, S_part, S_theory, osc_ent, cat_ent,
            seg, n_modes_eq, s_traj, times,
        )

        # --- Build harmonics JSON ---
        harmonics_data = {
            "n_relations": len(relations),
            "n_clusters": len(clusters),
            "total_enhancement": float(total_enh),
            "mode_frequencies": freqs_arr.tolist(),
            "mode_amplitudes": amps_arr.tolist(),
            "clusters": [
                {
                    "fundamental_freq": cl.fundamental_freq,
                    "n_members": len(cl.member_indices),
                    "harmonic_numbers": cl.harmonic_numbers,
                    "enhancement": cl.coincidence_enhancement,
                }
                for cl in clusters[:20]
            ],
        }

        # --- Pack binary streams ---
        audio_pcm = signal.astype(np.float32).tobytes()
        trajectory_bin = self._pack_trajectory(measurement.trajectory)
        entropy_bin = self._pack_entropy_stream(s_traj)
        partitions_bin = self._pack_partition_stream(measurement.trajectory)

        # --- Write .ykz container ---
        print(f"[YKZ] Writing {output_path}...")
        with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, cls=NumpyEncoder))
            zf.writestr("audio.pcm", audio_pcm)
            zf.writestr("trajectory.bin", trajectory_bin)
            zf.writestr("entropy.bin", entropy_bin)
            zf.writestr("partitions.bin", partitions_bin)
            zf.writestr("harmonics.json", json.dumps(harmonics_data, indent=2, cls=NumpyEncoder))
            zf.writestr("analysis.json", json.dumps(analysis, indent=2, cls=NumpyEncoder))

        file_size = output_path.stat().st_size
        wav_size = input_path.stat().st_size
        ratio = file_size / wav_size

        print(f"[YKZ] Done: {output_path}")
        print(f"[YKZ] WAV: {wav_size / 1e6:.1f} MB -> YKZ: {file_size / 1e6:.1f} MB (ratio: {ratio:.2f})")
        print(f"[YKZ] Contains: {len(manifest['channels'])} channels "
              f"(physical + categorical), {len(measurement.trajectory)} trajectory points")

        return str(output_path)

    def _load_wav(self, path: str):
        """Load WAV file to float32 mono."""
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

    def _pack_trajectory(self, trajectory: list) -> bytes:
        """Pack trajectory into compact binary.

        Per state: time(f64) + mode_index(i16) + phase(f32) + amplitude(f32) +
                   frequency(f32) + flat_index(i32) = 26 bytes per point.
        """
        buf = io.BytesIO()
        # Header: magic + count
        buf.write(FORMAT_MAGIC)
        buf.write(struct.pack('<I', len(trajectory)))
        for state in trajectory:
            buf.write(struct.pack(
                '<d h f f f i',
                state.time,
                state.mode_index,
                state.phase,
                state.amplitude,
                state.frequency,
                state.partition.flat_index,
            ))
        return buf.getvalue()

    def _pack_entropy_stream(self, s_traj: np.ndarray) -> bytes:
        """Pack S-entropy trajectory (Nx3 float64 array)."""
        buf = io.BytesIO()
        buf.write(FORMAT_MAGIC)
        buf.write(struct.pack('<I', len(s_traj)))
        buf.write(s_traj.astype(np.float64).tobytes())
        return buf.getvalue()

    def _pack_partition_stream(self, trajectory: list) -> bytes:
        """Pack partition coordinates: n(i16), l(i16), m(i16), s(i8) per state."""
        buf = io.BytesIO()
        buf.write(FORMAT_MAGIC)
        buf.write(struct.pack('<I', len(trajectory)))
        for state in trajectory:
            p = state.partition
            s_byte = 1 if p.s > 0 else 0
            buf.write(struct.pack('<h h h b', p.n, p.l, p.m, s_byte))
        return buf.getvalue()

    def _build_analysis_json(
        self, input_path, signal, sr, total_duration, actual_duration, n_analysis,
        clock, measurement, spec, stats, law, ensemble,
        freqs_arr, amps_arr, relations, clusters, total_enh,
        S_osc, S_cat, S_part, S_theory, osc_ent, cat_ent,
        seg, n_modes_eq, s_traj, times,
    ) -> dict:
        """Build the full analysis results dict (mirrors run_analysis.py output)."""
        k_B = 1.380649e-23

        c_cat = spec.categorical_capacity(n_modes=measurement.n_modes)
        c_phys = spec.physical_capacity(snr_db=96.0)

        H_osc = osc_ent.entropy_rate(seg)
        H_cat = cat_ent.shannon_entropy(seg)
        phys_obs = np.abs(seg)
        corr_amp = cat_ent.commutation_test(seg, phys_obs)
        velocity = np.diff(seg, prepend=seg[0])
        corr_vel = cat_ent.commutation_test(seg, velocity)

        occ_entropy = ensemble.occupation_entropy()
        n_occupied = len(ensemble.state_occupation_histogram())

        # Downsample trajectory for JSON
        if len(times) > 500:
            idx = np.linspace(0, len(times) - 1, 500, dtype=int)
            traj_sampled = {
                'times': times[idx].tolist(),
                'S_k': s_traj[idx, 0].tolist(),
                'S_t': s_traj[idx, 1].tolist(),
                'S_e': s_traj[idx, 2].tolist(),
            }
        else:
            traj_sampled = {
                'times': times.tolist(),
                'S_k': s_traj[:, 0].tolist(),
                'S_t': s_traj[:, 1].tolist(),
                'S_e': s_traj[:, 2].tolist(),
            }

        sample_period_ns = (1.0 / sr) * 1e9
        enhancement_over_sample = sample_period_ns / clock.categorical_resolution_ns

        return {
            'track_name': input_path.stem,
            'track_path': str(input_path),
            'sample_rate': sr,
            'total_duration_s': total_duration,
            'analysis_duration_s': actual_duration,
            'n_samples_analyzed': n_analysis,
            'peak_amplitude': float(np.max(np.abs(signal))),
            'rms': float(np.sqrt(np.mean(signal**2))),
            'hardware_clock': {
                'hardware_resolution_ns': clock.hardware_resolution_ns,
                'categorical_resolution_ns': clock.categorical_resolution_ns,
                'enhancement_over_sample_period': enhancement_over_sample,
            },
            'spectrometer': {
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
            },
            'ensemble': {
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
                'occupation_entropy': occ_entropy,
                'distinct_states_occupied': n_occupied,
                'second_law': law,
            },
            'harmonics': {
                'n_relations': len(relations),
                'n_clusters': len(clusters),
                'total_enhancement': float(total_enh),
                'mode_frequencies': freqs_arr.tolist(),
                'mode_amplitudes': amps_arr.tolist(),
                'clusters': [
                    {
                        'fundamental_freq': cl.fundamental_freq,
                        'n_members': len(cl.member_indices),
                        'harmonic_numbers': cl.harmonic_numbers,
                        'enhancement': cl.coincidence_enhancement,
                    }
                    for cl in clusters[:10]
                ],
            },
            'triple_equivalence': {
                'n_modes': n_modes_eq,
                'partition_depth': self.partition_depth,
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
            },
            's_entropy_trajectory_sampled': traj_sampled,
        }
