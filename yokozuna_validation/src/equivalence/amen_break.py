"""
Amen Break Ground Truth - Uses the amen break as a universal reference signal
for validating categorical audio measurements.

The amen break (from "Amen, Brother" by The Winstons, 1969) is the most
sampled drum break in music history. Its ubiquity in neurofunk, jungle,
and drum & bass makes it an ideal ground truth reference:

1. Every neurofunk track contains amen break derivatives
2. The break has known, precisely characterized temporal structure
3. Cross-track comparison using the same reference validates categorical measurements
4. The break's rhythmic micro-timing (groove) is well-documented
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AmenBreakProfile:
    """Categorical profile of an amen break instance."""
    onset_times: np.ndarray        # detected onset times (seconds)
    onset_intervals: np.ndarray    # inter-onset intervals
    swing_ratio: float             # ratio of long/short intervals
    groove_deviation: np.ndarray   # deviation from strict grid (microseconds)
    spectral_centroid: np.ndarray  # spectral centroid over time
    s_entropy_trajectory: np.ndarray  # (N, 3) S-entropy coordinates
    n_modes: int
    categorical_entropy: float
    tempo_bpm: float


class AmenBreakReference:
    """Ground truth reference using the amen break.

    Loads an amen break sample, computes its categorical profile,
    and provides methods for comparing other audio segments against
    the reference. The comparison yields:

    1. Groove similarity: how closely the target matches the reference
       micro-timing structure (geodesic distance in S-entropy space)
    2. Categorical similarity: overlap of categorical state trajectories
    3. Temporal precision: resolution of onset detection via categorical
       measurement vs. standard methods
    """

    # Standard amen break tempo: ~136-138 BPM (original recording ~136.7)
    STANDARD_BPM = 136.7
    # Number of beats in the standard amen break loop
    N_BEATS = 4
    # Number of 16th notes
    N_16THS = 16

    def __init__(
        self,
        reference_path: Optional[str] = None,
        sample_rate: int = 44100,
        partition_depth: int = 32,
    ):
        self.sample_rate = sample_rate
        self.partition_depth = partition_depth
        self.reference_signal = None
        self.reference_profile = None

        if reference_path and Path(reference_path).exists():
            self.load_reference(reference_path)

    def load_reference(self, path: str) -> None:
        """Load amen break reference audio file."""
        self.reference_signal, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        self.reference_profile = self.compute_profile(self.reference_signal)

    def compute_profile(self, signal: np.ndarray) -> AmenBreakProfile:
        """Compute the full categorical profile of an amen break instance."""
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=signal, sr=self.sample_rate,
            hop_length=128,  # high resolution
            backtrack=True,
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sample_rate, hop_length=128
        )

        # Inter-onset intervals
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
        else:
            onset_intervals = np.array([])

        # Swing ratio: ratio of odd/even intervals (shuffle feel)
        if len(onset_intervals) >= 2:
            odd_intervals = onset_intervals[0::2]
            even_intervals = onset_intervals[1::2]
            min_len = min(len(odd_intervals), len(even_intervals))
            if min_len > 0 and np.mean(even_intervals[:min_len]) > 0:
                swing_ratio = np.mean(odd_intervals[:min_len]) / np.mean(even_intervals[:min_len])
            else:
                swing_ratio = 1.0
        else:
            swing_ratio = 1.0

        # Groove deviation from strict 16th note grid
        tempo, _ = librosa.beat.beat_track(y=signal, sr=self.sample_rate)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else self.STANDARD_BPM
        tempo = float(tempo) if tempo > 0 else self.STANDARD_BPM

        sixteenth_duration = 60.0 / (tempo * 4)  # duration of one 16th note
        grid_times = np.arange(len(onset_times)) * sixteenth_duration
        if len(onset_times) > 0 and len(grid_times) > 0:
            min_len = min(len(onset_times), len(grid_times))
            groove_deviation = (onset_times[:min_len] - grid_times[:min_len]) * 1e6  # microseconds
        else:
            groove_deviation = np.array([])

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=signal, sr=self.sample_rate, hop_length=512
        )[0]

        # S-entropy trajectory (simplified: from spectral features)
        n_frames = len(spectral_centroid)
        s_entropy = np.zeros((n_frames, 3))

        # S_k from spectral centroid variation
        if n_frames > 1:
            sc_diff = np.abs(np.diff(spectral_centroid, prepend=spectral_centroid[0]))
            s_entropy[:, 0] = 1.380649e-23 * np.log(1 + sc_diff / (np.mean(sc_diff) + 1e-10))

        # S_t from onset density (temporal entropy)
        for i in range(n_frames):
            t = i * 512 / self.sample_rate
            nearby_onsets = np.sum(np.abs(onset_times - t) < 0.05)
            s_entropy[i, 1] = 1.380649e-23 * np.log(1 + nearby_onsets)

        # S_e from RMS energy variation
        rms = librosa.feature.rms(y=signal, hop_length=512)[0]
        if len(rms) == n_frames:
            s_entropy[:, 2] = 1.380649e-23 * np.log(1 + rms / (np.mean(rms) + 1e-10))

        # Categorical entropy
        n = self.partition_depth
        k_B = 1.380649e-23
        n_modes = min(len(onset_frames), 64)
        categorical_entropy = k_B * max(n_modes, 1) * np.log(max(n, 2))

        return AmenBreakProfile(
            onset_times=onset_times,
            onset_intervals=onset_intervals,
            swing_ratio=swing_ratio,
            groove_deviation=groove_deviation,
            spectral_centroid=spectral_centroid,
            s_entropy_trajectory=s_entropy,
            n_modes=n_modes,
            categorical_entropy=categorical_entropy,
            tempo_bpm=tempo,
        )

    def groove_similarity(
        self, target_profile: AmenBreakProfile
    ) -> float:
        """Compute groove similarity between reference and target.

        Uses geodesic distance in S-entropy space. Returns value in [0, 1]
        where 1 = identical groove.
        """
        if self.reference_profile is None:
            return 0.0

        ref_s = self.reference_profile.s_entropy_trajectory
        tgt_s = target_profile.s_entropy_trajectory

        # Resample to same length
        min_len = min(len(ref_s), len(tgt_s))
        if min_len == 0:
            return 0.0

        ref_s = ref_s[:min_len]
        tgt_s = tgt_s[:min_len]

        # Euclidean distance in S-entropy space (frame-wise)
        distances = np.sqrt(np.sum((ref_s - tgt_s) ** 2, axis=1))
        mean_distance = np.mean(distances)

        # Convert to similarity (exponential decay)
        similarity = np.exp(-mean_distance / (1.380649e-23 * 10))
        return float(np.clip(similarity, 0, 1))

    def temporal_precision_comparison(
        self, target_profile: AmenBreakProfile
    ) -> Dict[str, float]:
        """Compare temporal precision of onset detection.

        Returns statistics on how precisely onsets are resolved
        relative to the reference.
        """
        if self.reference_profile is None:
            return {'error': -1.0}

        ref_onsets = self.reference_profile.onset_times
        tgt_onsets = target_profile.onset_times

        if len(ref_onsets) == 0 or len(tgt_onsets) == 0:
            return {
                'n_ref_onsets': len(ref_onsets),
                'n_tgt_onsets': len(tgt_onsets),
                'mean_deviation_us': 0,
                'max_deviation_us': 0,
                'matched_fraction': 0,
            }

        # Match each reference onset to nearest target onset
        matched_deviations = []
        for ref_t in ref_onsets:
            if len(tgt_onsets) > 0:
                nearest_idx = np.argmin(np.abs(tgt_onsets - ref_t))
                deviation = abs(tgt_onsets[nearest_idx] - ref_t) * 1e6  # microseconds
                matched_deviations.append(deviation)

        matched_deviations = np.array(matched_deviations)
        tolerance_us = 5000  # 5ms tolerance for matching

        return {
            'n_ref_onsets': len(ref_onsets),
            'n_tgt_onsets': len(tgt_onsets),
            'mean_deviation_us': float(np.mean(matched_deviations)) if len(matched_deviations) > 0 else 0,
            'max_deviation_us': float(np.max(matched_deviations)) if len(matched_deviations) > 0 else 0,
            'std_deviation_us': float(np.std(matched_deviations)) if len(matched_deviations) > 0 else 0,
            'matched_fraction': float(np.mean(matched_deviations < tolerance_us)) if len(matched_deviations) > 0 else 0,
        }

    def detect_amen_in_track(
        self, track_signal: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[float, float, float]]:
        """Detect amen break occurrences in a full track.

        Returns list of (start_time, end_time, similarity_score) tuples.
        """
        if self.reference_signal is None:
            return []

        ref_len = len(self.reference_signal)
        hop = ref_len // 2  # 50% overlap
        results = []

        for start in range(0, len(track_signal) - ref_len, hop):
            segment = track_signal[start:start + ref_len]
            segment_profile = self.compute_profile(segment)
            similarity = self.groove_similarity(segment_profile)

            if similarity >= threshold:
                start_time = start / self.sample_rate
                end_time = (start + ref_len) / self.sample_rate
                results.append((start_time, end_time, similarity))

        return results
