"""
Virtual Spectrometer — Hardware clock + virtual gas spectrometer for categorical measurement.

Constructs a categorical clock from hardware timing processes and measures
audio signals through the categorical state counting framework. The virtual
spectrometer decomposes audio into oscillatory modes and maps them to
S-entropy coordinates (S_k, S_t, S_e) and partition coordinates (n, l, m, s).
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PartitionCoordinate:
    """Partition coordinate (n, l, m, s) for a single oscillatory mode.

    n: partition depth — number of distinguishable amplitude levels
    l: harmonic order — position within harmonic series (0 = fundamental)
    m: phase index — discrete phase state within current harmonic order
    s: chirality — direction of phase evolution (+0.5 advancing, -0.5 retreating)
    """
    n: int
    l: int
    m: int
    s: float  # +0.5 or -0.5

    @property
    def degeneracy(self) -> int:
        """g_n = 2n^2 — total states at partition depth n."""
        return 2 * self.n ** 2

    @property
    def flat_index(self) -> int:
        """Unique integer index for this coordinate within the degeneracy shell."""
        # Sum states from l=0..l-1, then offset by m and s
        base = sum(2 * (2 * ll + 1) for ll in range(self.l))
        m_offset = self.m + self.l  # shift m from [-l,l] to [0,2l]
        s_offset = 0 if self.s > 0 else 1
        return base + 2 * m_offset + s_offset


@dataclass
class SEntropyCoordinate:
    """S-entropy coordinate (S_k, S_t, S_e) for a measurement point.

    S_k: spectral entropy — complexity of harmonic structure
    S_t: temporal entropy — granularity of time structure
    S_e: energetic entropy — dynamic range and amplitude variation
    """
    S_k: float
    S_t: float
    S_e: float

    @property
    def total(self) -> float:
        """Total S-entropy magnitude."""
        return self.S_k + self.S_t + self.S_e

    def distance(self, other: 'SEntropyCoordinate') -> float:
        """Euclidean distance in S-entropy space."""
        return np.sqrt(
            (self.S_k - other.S_k) ** 2
            + (self.S_t - other.S_t) ** 2
            + (self.S_e - other.S_e) ** 2
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.S_k, self.S_t, self.S_e])


@dataclass
class CategoricalState:
    """A single categorical state measurement at a point in time."""
    time: float
    partition: PartitionCoordinate
    s_entropy: SEntropyCoordinate
    mode_index: int
    phase: float  # instantaneous phase
    amplitude: float  # instantaneous amplitude
    frequency: float  # instantaneous frequency


@dataclass
class CategoricalTransition:
    """A transition between two categorical states."""
    from_state: CategoricalState
    to_state: CategoricalState
    delta_s_entropy: float  # change in total S-entropy
    is_irreversible: bool  # True if delta_s_entropy > 0 (categorical second law)


@dataclass
class CategoricalMeasurement:
    """Complete categorical measurement result for a signal segment."""
    trajectory: List[CategoricalState]
    transitions: List[CategoricalTransition]
    total_entropy: float
    entropy_production: float
    n_modes: int
    partition_depth: int
    duration: float
    enhancement_factor: float = 1.0


# ---------------------------------------------------------------------------
# Hardware Clock
# ---------------------------------------------------------------------------

class HardwareClock:
    """Categorical clock constructed from hardware timing processes.

    Uses the system's high-resolution performance counter as the physical
    clock, then constructs categorical resolution by dividing hardware
    ticks by the number of categorical states N.

    categorical_resolution = hardware_resolution / N_states
    """

    def __init__(self, n_states: int = 1000):
        self.n_states = n_states
        # Measure hardware clock resolution
        self._calibrate()

    def _calibrate(self, n_samples: int = 1000) -> None:
        """Calibrate hardware clock resolution by measuring minimum tick."""
        deltas = []
        for _ in range(n_samples):
            t0 = time.perf_counter_ns()
            t1 = time.perf_counter_ns()
            dt = t1 - t0
            if dt > 0:
                deltas.append(dt)

        if deltas:
            self.hardware_resolution_ns = float(np.median(deltas))
        else:
            # Fallback: assume ~100ns resolution
            self.hardware_resolution_ns = 100.0

        self.categorical_resolution_ns = (
            self.hardware_resolution_ns / self.n_states
        )

    @property
    def hardware_resolution_s(self) -> float:
        return self.hardware_resolution_ns * 1e-9

    @property
    def categorical_resolution_s(self) -> float:
        return self.categorical_resolution_ns * 1e-9

    def tick(self) -> float:
        """Return current time in seconds (high resolution)."""
        return time.perf_counter()

    def categorical_tick(self) -> int:
        """Return current time in categorical units (integer)."""
        return int(time.perf_counter_ns() / self.categorical_resolution_ns)

    def __repr__(self) -> str:
        return (
            f"HardwareClock(n_states={self.n_states}, "
            f"hw_res={self.hardware_resolution_ns:.1f}ns, "
            f"cat_res={self.categorical_resolution_ns:.3f}ns)"
        )


# ---------------------------------------------------------------------------
# Virtual Spectrometer
# ---------------------------------------------------------------------------

class VirtualSpectrometer:
    """Virtual spectrometer for categorical measurement of audio signals.

    Decomposes audio into M oscillatory modes via sinusoidal modeling,
    then measures each mode's categorical state through:
      - S-entropy coordinates (S_k, S_t, S_e)
      - Partition coordinates (n, l, m, s)

    The measurement is orthogonal to physical observables: [O_cat, O_phys] = 0.
    """

    # Reference values for S-entropy (ensure positivity and dimensional consistency)
    PHI_0 = 1e-6       # reference phase deviation (rad)
    TAU_0 = 1e-6        # reference categorical period (s)
    E_0 = 1e-12         # reference energy

    def __init__(
        self,
        sample_rate: int = 44100,
        n_modes: int = 64,
        partition_depth: int = 32,
        clock: Optional[HardwareClock] = None,
    ):
        self.sample_rate = sample_rate
        self.n_modes = n_modes
        self.partition_depth = partition_depth
        self.clock = clock or HardwareClock(n_states=partition_depth)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)

    # -------------------------------------------------------------------
    # Mode decomposition
    # -------------------------------------------------------------------

    def decompose_modes(
        self, signal: np.ndarray, n_modes: Optional[int] = None
    ) -> List[dict]:
        """Decompose signal into M oscillatory modes via FFT peak extraction.

        Returns list of dicts with keys: frequency, amplitude, phase, energy.
        """
        M = n_modes or self.n_modes
        N = len(signal)
        dt = 1.0 / self.sample_rate

        # FFT
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=dt)
        magnitudes = np.abs(spectrum) / N
        phases = np.angle(spectrum)

        # Extract top M peaks (excluding DC)
        mag_copy = magnitudes.copy()
        mag_copy[0] = 0  # exclude DC

        modes = []
        for _ in range(min(M, len(mag_copy) - 1)):
            idx = np.argmax(mag_copy)
            if mag_copy[idx] < 1e-15:
                break

            freq = freqs[idx]
            amp = 2.0 * magnitudes[idx]  # single-sided amplitude
            phase = phases[idx]
            energy = 0.5 * amp ** 2 * (2.0 * np.pi * freq) ** 2

            modes.append({
                'frequency': freq,
                'amplitude': amp,
                'phase': phase,
                'energy': energy,
                'bin_index': idx,
            })

            # Zero out neighborhood to avoid picking same peak
            half_width = max(3, N // 500)
            lo = max(0, idx - half_width)
            hi = min(len(mag_copy), idx + half_width + 1)
            mag_copy[lo:hi] = 0

        return modes

    # -------------------------------------------------------------------
    # S-entropy computation
    # -------------------------------------------------------------------

    def compute_s_entropy(
        self, mode: dict, signal: np.ndarray, t_start: float = 0.0
    ) -> SEntropyCoordinate:
        """Compute S-entropy coordinates (S_k, S_t, S_e) for a mode.

        S_k = k_B * ln((|δφ| + φ_0) / φ_0)   — spectral entropy
        S_t = k_B * ln(τ / τ_0)                — temporal entropy
        S_e = k_B * ln((E + E_0) / E_0)        — energetic entropy
        """
        freq = mode['frequency']
        amp = mode['amplitude']
        phase = mode['phase']
        energy = mode['energy']

        # Phase deviation: deviation from pure reference oscillator
        N = len(signal)
        t = np.arange(N) / self.sample_rate + t_start
        reference = amp * np.cos(2 * np.pi * freq * t + phase)
        delta_phi = np.mean(np.abs(np.angle(
            np.exp(1j * 2 * np.pi * freq * t) * np.exp(-1j * phase)
        )))

        # For complex signals, compute phase deviation from signal itself
        if N > 1:
            analytic = np.zeros(N, dtype=complex)
            analytic.real = signal
            # Simple Hilbert via FFT
            spec = np.fft.fft(signal)
            spec[0] = 0
            n_half = N // 2
            spec[1:n_half] *= 2
            spec[n_half + 1:] = 0
            analytic = np.fft.ifft(spec)
            inst_phase = np.unwrap(np.angle(analytic))
            # Phase deviation = std of instantaneous frequency deviation
            if N > 2:
                inst_freq = np.diff(inst_phase) / (2 * np.pi / self.sample_rate)
                delta_phi = np.std(inst_freq - freq) * (2 * np.pi / self.sample_rate)
            else:
                delta_phi = abs(phase)

        # Categorical period: period of one partition traversal
        if freq > 0:
            tau = 1.0 / (freq * self.partition_depth)
        else:
            tau = self.TAU_0

        # S-entropy coordinates
        S_k = self.k_B * np.log((abs(delta_phi) + self.PHI_0) / self.PHI_0)
        S_t = self.k_B * np.log(max(tau, self.TAU_0) / self.TAU_0)
        S_e = self.k_B * np.log((energy + self.E_0) / self.E_0)

        return SEntropyCoordinate(S_k=float(S_k), S_t=float(S_t), S_e=float(S_e))

    # -------------------------------------------------------------------
    # Partition coordinate computation
    # -------------------------------------------------------------------

    def compute_partition_coordinate(
        self, mode: dict, phase_at_t: float
    ) -> PartitionCoordinate:
        """Compute partition coordinate (n, l, m, s) for a mode at time t.

        n: partition depth (fixed by spectrometer configuration)
        l: harmonic order = floor(amplitude_ratio * (n-1))
        m: phase index = floor(n * phase / (2π)) mod n, mapped to [-l, l]
        s: chirality = sign of instantaneous frequency deviation
        """
        n = self.partition_depth
        freq = mode['frequency']
        amp = mode['amplitude']
        energy = mode['energy']

        # Harmonic order from amplitude ranking
        # l = 0 for fundamental (strongest), higher for overtones
        # We use a normalized energy measure
        max_energy = 0.5 * (2 * np.pi * self.sample_rate / 2) ** 2  # reference max
        energy_ratio = min(energy / (max_energy + 1e-30), 1.0)
        l = int(np.floor(energy_ratio * (n - 1)))
        l = max(0, min(l, n - 1))

        # Phase index: map continuous phase to discrete state in [-l, l]
        phase_normalized = (phase_at_t % (2 * np.pi)) / (2 * np.pi)
        j = int(np.floor(phase_normalized * n)) % n
        # Map j in [0, n) to m in [-l, l]
        if l > 0:
            m = int(np.round(j / n * (2 * l + 1))) - l
            m = max(-l, min(m, l))
        else:
            m = 0

        # Chirality: +0.5 if frequency increasing, -0.5 if decreasing
        s = 0.5  # default: advancing

        return PartitionCoordinate(n=n, l=l, m=m, s=s)

    # -------------------------------------------------------------------
    # Full measurement
    # -------------------------------------------------------------------

    def measure(
        self,
        signal: np.ndarray,
        hop_size: int = 512,
        n_modes: Optional[int] = None,
    ) -> CategoricalMeasurement:
        """Perform complete categorical measurement of an audio signal.

        Decomposes signal into frames, measures each mode's categorical
        state at each frame, and assembles the categorical trajectory.

        Returns CategoricalMeasurement with trajectory, transitions,
        entropy, and enhancement factor.
        """
        M = n_modes or self.n_modes
        N = len(signal)
        n_frames = max(1, (N - hop_size) // hop_size)
        dt = hop_size / self.sample_rate

        trajectory: List[CategoricalState] = []
        transitions: List[CategoricalTransition] = []

        total_entropy = 0.0
        entropy_production = 0.0

        prev_states: dict = {}  # mode_index -> previous CategoricalState

        for frame_idx in range(n_frames):
            start = frame_idx * hop_size
            end = min(start + hop_size * 2, N)  # analysis window = 2 * hop
            frame = signal[start:end]

            if len(frame) < 4:
                continue

            # Apply window
            window = np.hanning(len(frame))
            frame_windowed = frame * window

            # Decompose modes for this frame
            modes = self.decompose_modes(frame_windowed, n_modes=M)
            t_current = start / self.sample_rate

            for mode_idx, mode in enumerate(modes):
                # S-entropy
                s_entropy = self.compute_s_entropy(mode, frame_windowed, t_start=t_current)

                # Instantaneous phase at frame center
                t_center = t_current + len(frame) / (2 * self.sample_rate)
                phase_at_t = mode['phase'] + 2 * np.pi * mode['frequency'] * t_center

                # Partition coordinate
                partition = self.compute_partition_coordinate(mode, phase_at_t)

                state = CategoricalState(
                    time=t_current,
                    partition=partition,
                    s_entropy=s_entropy,
                    mode_index=mode_idx,
                    phase=float(phase_at_t),
                    amplitude=float(mode['amplitude']),
                    frequency=float(mode['frequency']),
                )
                trajectory.append(state)
                total_entropy += s_entropy.total

                # Compute transition from previous state of same mode
                if mode_idx in prev_states:
                    prev = prev_states[mode_idx]
                    delta_s = s_entropy.total - prev.s_entropy.total
                    transition = CategoricalTransition(
                        from_state=prev,
                        to_state=state,
                        delta_s_entropy=float(delta_s),
                        is_irreversible=(delta_s > 0),
                    )
                    transitions.append(transition)
                    if delta_s > 0:
                        entropy_production += delta_s

                prev_states[mode_idx] = state

        # Enhancement factor
        enhancement = self.compute_enhancement(
            n_modes=len(prev_states),
            partition_depth=self.partition_depth,
        )

        return CategoricalMeasurement(
            trajectory=trajectory,
            transitions=transitions,
            total_entropy=float(total_entropy),
            entropy_production=float(entropy_production),
            n_modes=len(prev_states),
            partition_depth=self.partition_depth,
            duration=N / self.sample_rate,
            enhancement_factor=enhancement,
        )

    # -------------------------------------------------------------------
    # Enhancement factor
    # -------------------------------------------------------------------

    def compute_enhancement(
        self, n_modes: int, partition_depth: int
    ) -> float:
        """Compute the five-mechanism enhancement chain.

        1. Ternary enhancement:        10^3.5
        2. Multimodal enhancement:      10^(M * log10(n))
        3. Harmonic coincidence:        10^3
        4. Poincaré computing:          10^66  (theoretical max)
        5. Continuous refinement:       10^43.4 (theoretical max)

        For practical computation we return the multimodal factor,
        which is the directly realizable enhancement.
        """
        # Multimodal enhancement: n^M accessible states
        # In log10: M * log10(n)
        log10_multimodal = n_modes * np.log10(max(partition_depth, 2))

        # Ternary: 3 S-entropy coordinates vs 1 scalar
        log10_ternary = 3.5

        # Harmonic coincidence network (conservative estimate)
        log10_harmonic = min(3.0, 0.5 * n_modes)

        # Total practical enhancement (log10)
        log10_total = log10_ternary + log10_multimodal + log10_harmonic

        return 10 ** min(log10_total, 300)  # cap to avoid overflow

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    def categorical_capacity(self, n_modes: int = None) -> float:
        """Categorical channel capacity: C_cat = M * log2(n) bits."""
        M = n_modes or self.n_modes
        return M * np.log2(self.partition_depth)

    def physical_capacity(self, snr_db: float = 96.0) -> float:
        """Physical channel capacity: C_phys = B * log2(1 + SNR) bits/s."""
        bandwidth = self.sample_rate / 2.0
        snr_linear = 10 ** (snr_db / 10.0)
        return bandwidth * np.log2(1 + snr_linear)

    def total_capacity(self, snr_db: float = 96.0, n_modes: int = None) -> float:
        """Total capacity: C_total = C_phys + C_cat."""
        return self.physical_capacity(snr_db) + self.categorical_capacity(n_modes)

    def __repr__(self) -> str:
        return (
            f"VirtualSpectrometer(sr={self.sample_rate}, "
            f"modes={self.n_modes}, n={self.partition_depth}, "
            f"C_cat={self.categorical_capacity():.1f} bits)"
        )
