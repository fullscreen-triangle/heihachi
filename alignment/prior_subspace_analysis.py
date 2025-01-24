import librosa
import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Optional
import torch
from dataclasses import dataclass


@dataclass
class ComponentAnalysis:
    type: str  # Component type (e.g., 'subbass', 'reese', 'drums')
    energy: float  # Component energy
    pattern: np.ndarray  # Temporal pattern
    spectral: np.ndarray  # Spectral profile
    transformations: List[str]  # Detected transformations
    confidence: float  # Detection confidence


class PriorSubspaceAnalysis:
    def __init__(self, n_components: int = 8, n_iterations: int = 100):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.eps = 1e-10
        self.sr = 44100
        self.hop_length = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Component indices with neurofunk-specific components
        self.SUBBASS = 0    # Clean sub bass
        self.REESE = 1      # Reese bass (distorted, evolving)
        self.KICK = 2       # Kick drum
        self.SNARE = 3      # Snare drum
        self.HIHAT = 4      # Hi-hat
        self.AMBI = 5       # Ambient/atmospheric sounds
        self.FX = 6         # Sound effects
        self.SYNTH = 7      # Synth elements

        # Initialize transformation detection thresholds
        self.thresholds = {
            'spectral_flux': 0.2,
            'harmonic_change': 0.15,
            'amplitude_mod': 0.1,
            'min_confidence': 0.6
        }

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze audio using prior subspace analysis with neurofunk components.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing component information and relationships
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Compute STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=self.hop_length)
        S = np.abs(D)

        # Initialize spectral and temporal priors
        P_w = self._initialize_spectral_priors(S.shape[0])
        P_h = self._initialize_temporal_priors(S.shape[1])

        # Decompose into components
        W, H = self.decompose(S)

        # Analyze each component
        components = self._analyze_components(W, H, D)

        # Analyze relationships between components
        relationships = self._analyze_component_relationships(components)

        return {
            'components': [self._component_to_dict(c) for c in components],
            'relationships': relationships,
            'overall_characteristics': self._analyze_overall_characteristics(components)
        }

    def _analyze_components(self, W: np.ndarray, H: np.ndarray,
                          D: np.ndarray) -> List[ComponentAnalysis]:
        """Analyze each decomposed component."""
        components = []
        phase = np.angle(D)

        for i in range(self.n_components):
            # Reconstruct component spectrogram
            S_component = np.outer(W[:, i], H[i, :])
            
            # Detect transformations
            transformations = self._detect_transformations(S_component, H[i, :])
            
            # Calculate confidence
            confidence = self._calculate_component_confidence(
                W[:, i], H[i, :], transformations
            )
            
            # Determine component type
            comp_type = self._determine_component_type(W[:, i], H[i, :])
            
            components.append(ComponentAnalysis(
                type=comp_type,
                energy=float(np.sum(S_component)),
                pattern=H[i, :],
                spectral=W[:, i],
                transformations=transformations,
                confidence=confidence
            ))

        return components

    def _detect_transformations(self, S: np.ndarray, H: np.ndarray) -> List[str]:
        """Detect transformations applied to component."""
        transformations = []
        
        # Detect spectral flux
        flux = np.mean(np.diff(S, axis=1) ** 2)
        if flux > self.thresholds['spectral_flux']:
            transformations.append('spectral_morphing')
        
        # Detect harmonic changes
        if self._detect_harmonic_change(S):
            transformations.append('harmonic_modulation')
        
        # Detect amplitude modulation
        if self._detect_amplitude_modulation(H):
            transformations.append('amplitude_modulation')
        
        return transformations

    def _detect_harmonic_change(self, S: np.ndarray) -> bool:
        """Detect significant harmonic changes."""
        # Compute harmonic change detection function
        hcdf = librosa.feature.spectral_contrast(S=S)
        
        # Check for significant changes
        changes = np.diff(hcdf.mean(axis=0))
        return np.any(np.abs(changes) > self.thresholds['harmonic_change'])

    def _detect_amplitude_modulation(self, H: np.ndarray) -> bool:
        """Detect amplitude modulation in temporal pattern."""
        # Compute envelope
        envelope = np.abs(signal.hilbert(H))
        
        # Detect periodic modulation
        fft = np.abs(np.fft.rfft(envelope))
        return np.any(fft[1:] > self.thresholds['amplitude_mod'] * fft[0])

    def _calculate_component_confidence(self, W: np.ndarray, H: np.ndarray,
                                     transformations: List[str]) -> float:
        """Calculate confidence score for component detection."""
        # Base confidence from energy concentration
        spectral_conf = np.sum(W ** 2) / (np.sum(W) + self.eps)
        temporal_conf = np.sum(H ** 2) / (np.sum(H) + self.eps)
        
        # Adjust based on number of detected transformations
        transform_factor = 1.0 + 0.1 * len(transformations)
        
        confidence = (spectral_conf + temporal_conf) * transform_factor / 2
        return float(min(1.0, confidence))

    def _determine_component_type(self, W: np.ndarray, H: np.ndarray) -> str:
        """Determine component type based on spectral and temporal characteristics."""
        # Calculate spectral centroid
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        centroid = np.sum(W * freqs) / (np.sum(W) + self.eps)
        
        # Calculate temporal characteristics
        activity = np.mean(H > np.mean(H))
        periodicity = self._calculate_periodicity(H)
        
        # Classify based on characteristics
        if centroid < 100 and activity > 0.3:
            return 'subbass'
        elif centroid < 500 and self._is_reese_bass(W, H):
            return 'reese'
        elif centroid < 200 and periodicity > 0.7:
            return 'kick'
        elif 200 <= centroid <= 2000 and activity < 0.3:
            return 'snare'
        elif centroid > 5000 and periodicity > 0.8:
            return 'hihat'
        elif centroid > 2000 and activity < 0.2:
            return 'ambi'
        elif self._is_fx(W, H):
            return 'fx'
        else:
            return 'synth'

    def _is_reese_bass(self, W: np.ndarray, H: np.ndarray) -> bool:
        """Detect if component is a Reese bass."""
        # Check for characteristic harmonic content
        harmonics = self._detect_harmonics(W)
        
        # Check for modulation
        mod_depth = self._calculate_modulation_depth(H)
        
        return len(harmonics) >= 3 and mod_depth > 0.3

    def _is_fx(self, W: np.ndarray, H: np.ndarray) -> bool:
        """Detect if component is a sound effect."""
        # Check for non-harmonic content
        harmonic_ratio = self._calculate_harmonic_ratio(W)
        
        # Check for sparse, non-periodic activation
        sparsity = 1.0 - np.mean(H > 0.1)
        periodicity = self._calculate_periodicity(H)
        
        return harmonic_ratio < 0.3 and sparsity > 0.7 and periodicity < 0.3

    def _calculate_periodicity(self, H: np.ndarray) -> float:
        """Calculate periodicity of temporal pattern."""
        # Compute autocorrelation
        corr = np.correlate(H, H, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peaks
        peaks = signal.find_peaks(corr)[0]
        if len(peaks) > 1:
            # Calculate regularity of peak spacing
            peak_spacing = np.diff(peaks)
            regularity = 1.0 - np.std(peak_spacing) / np.mean(peak_spacing)
            return float(regularity)
        return 0.0

    def _detect_harmonics(self, W: np.ndarray) -> List[float]:
        """Detect harmonic frequencies in spectral profile."""
        # Find peaks in spectrum
        peaks = signal.find_peaks(W)[0]
        
        # Convert to frequencies
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        peak_freqs = freqs[peaks]
        
        # Find harmonically related peaks
        harmonics = []
        if len(peak_freqs) > 0:
            f0 = peak_freqs[0]  # Assume first peak is fundamental
            for f in peak_freqs[1:]:
                ratio = f / f0
                if abs(round(ratio) - ratio) < 0.1:  # 10% tolerance
                    harmonics.append(float(f))
        
        return harmonics

    def _calculate_modulation_depth(self, H: np.ndarray) -> float:
        """Calculate modulation depth of temporal pattern."""
        envelope = np.abs(signal.hilbert(H))
        return float((np.max(envelope) - np.min(envelope)) / (np.max(envelope) + self.eps))

    def _calculate_harmonic_ratio(self, W: np.ndarray) -> float:
        """Calculate ratio of harmonic to non-harmonic energy."""
        harmonics = self._detect_harmonics(W)
        if harmonics:
            harmonic_energy = sum(W[int(f)] for f in harmonics)
            total_energy = np.sum(W)
            return float(harmonic_energy / (total_energy + self.eps))
        return 0.0

    def _analyze_component_relationships(self, 
                                      components: List[ComponentAnalysis]) -> Dict:
        """Analyze relationships between components."""
        relationships = {
            'temporal': self._analyze_temporal_relationships(components),
            'spectral': self._analyze_spectral_relationships(components),
            'energy': self._analyze_energy_relationships(components)
        }
        
        return relationships

    def _analyze_temporal_relationships(self, 
                                     components: List[ComponentAnalysis]) -> List[Dict]:
        """Analyze temporal relationships between components."""
        relationships = []
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                # Calculate temporal correlation
                correlation = np.corrcoef(comp1.pattern, comp2.pattern)[0, 1]
                
                if abs(correlation) > 0.3:  # Significant correlation threshold
                    relationships.append({
                        'component1': comp1.type,
                        'component2': comp2.type,
                        'correlation': float(correlation),
                        'relationship_type': 'sync' if correlation > 0 else 'antisync'
                    })
        
        return relationships

    def _analyze_spectral_relationships(self, 
                                     components: List[ComponentAnalysis]) -> List[Dict]:
        """Analyze spectral relationships between components."""
        relationships = []
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                # Calculate spectral overlap
                overlap = np.sum(comp1.spectral * comp2.spectral) / (
                    np.sqrt(np.sum(comp1.spectral ** 2) * np.sum(comp2.spectral ** 2))
                    + self.eps
                )
                
                if overlap > 0.3:  # Significant overlap threshold
                    relationships.append({
                        'component1': comp1.type,
                        'component2': comp2.type,
                        'overlap': float(overlap),
                        'frequency_range': self._get_overlap_range(
                            comp1.spectral, comp2.spectral
                        )
                    })
        
        return relationships

    def _analyze_energy_relationships(self, 
                                   components: List[ComponentAnalysis]) -> Dict:
        """Analyze energy relationships between components."""
        total_energy = sum(c.energy for c in components)
        
        return {
            'distribution': {
                c.type: float(c.energy / (total_energy + self.eps))
                for c in components
            },
            'dominant_components': sorted(
                [(c.type, c.energy) for c in components],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

    def _get_overlap_range(self, spec1: np.ndarray, spec2: np.ndarray) -> Dict:
        """Get frequency range of spectral overlap."""
        # Find regions of significant overlap
        overlap = (spec1 > np.mean(spec1)) & (spec2 > np.mean(spec2))
        if not np.any(overlap):
            return {'start': 0, 'end': 0}
        
        # Convert to frequencies
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        overlap_freqs = freqs[overlap]
        
        return {
            'start': float(np.min(overlap_freqs)),
            'end': float(np.max(overlap_freqs))
        }

    def _analyze_overall_characteristics(self, 
                                      components: List[ComponentAnalysis]) -> Dict:
        """Analyze overall characteristics of the decomposition."""
        return {
            'complexity': self._calculate_complexity(components),
            'component_balance': self._calculate_component_balance(components),
            'transformation_stats': self._calculate_transformation_stats(components),
            'confidence_stats': {
                'mean': float(np.mean([c.confidence for c in components])),
                'std': float(np.std([c.confidence for c in components]))
            }
        }

    def _calculate_complexity(self, components: List[ComponentAnalysis]) -> float:
        """Calculate overall complexity score."""
        # Consider number of active components and their transformations
        n_active = sum(1 for c in components if c.energy > 0.1)
        n_transforms = sum(len(c.transformations) for c in components)
        
        return float((n_active + n_transforms) / (2 * self.n_components))

    def _calculate_component_balance(self, 
                                  components: List[ComponentAnalysis]) -> Dict:
        """Calculate balance between different component types."""
        type_groups = {
            'rhythm': ['kick', 'snare', 'hihat'],
            'bass': ['subbass', 'reese'],
            'ambient': ['ambi', 'fx', 'synth']
        }
        
        balance = {}
        total_energy = sum(c.energy for c in components)
        
        for group_name, types in type_groups.items():
            group_energy = sum(
                c.energy for c in components if c.type in types
            )
            balance[group_name] = float(group_energy / (total_energy + self.eps))
        
        return balance

    def _calculate_transformation_stats(self, 
                                     components: List[ComponentAnalysis]) -> Dict:
        """Calculate statistics about component transformations."""
        all_transforms = []
        for c in components:
            all_transforms.extend(c.transformations)
        
        return {
            'total': len(all_transforms),
            'types': {
                t: all_transforms.count(t)
                for t in set(all_transforms)
            }
        }

    def _component_to_dict(self, component: ComponentAnalysis) -> Dict:
        """Convert ComponentAnalysis to dictionary."""
        return {
            'type': component.type,
            'energy': float(component.energy),
            'pattern': component.pattern.tolist(),
            'spectral': component.spectral.tolist(),
            'transformations': component.transformations,
            'confidence': float(component.confidence)
        }

    def _initialize_spectral_priors(self, n_freq: int) -> np.ndarray:
        """Initialize detailed frequency-dependent priors for DnB components."""
        P_w = np.zeros((n_freq, self.n_components))
        freq_range = np.linspace(0, self.sr / 2, n_freq)

        # Sub-bass prior (20-60 Hz)
        P_w[:, self.SUBBASS] = self._gaussian_prior(freq_range, mean=40, std=20)

        # Bass harmonics prior (60-250 Hz with harmonics)
        fundamental_freq = 80
        for harmonic in range(1, 4):
            P_w[:, self.BASS_HARM] += self._gaussian_prior(
                freq_range,
                mean=fundamental_freq * harmonic,
                std=20 * harmonic
            )

        # Kick drum prior (30-100 Hz with specific attack characteristics)
        P_w[:, self.KICK] = (
                self._gaussian_prior(freq_range, mean=50, std=20) +  # Fundamental
                self._gaussian_prior(freq_range, mean=120, std=40)  # Attack
        )

        # Snare prior (200-1200 Hz with noise component)
        P_w[:, self.SNARE] = (
                self._gaussian_prior(freq_range, mean=200, std=50) +  # Body
                self._gaussian_prior(freq_range, mean=400, std=100) +  # Mid
                self._uniform_prior(freq_range, 800, 1200)  # Noise
        )

        # Hi-hat prior (5000-15000 Hz with specific characteristics)
        P_w[:, self.HIHAT] = self._uniform_prior(freq_range, 5000, 15000)

        # Cymbal prior (8000-20000 Hz with decay characteristics)
        P_w[:, self.CYMBAL] = (
                self._uniform_prior(freq_range, 8000, 20000) *
                np.linspace(0.5, 1.0, n_freq)
        )

        # Reese bass prior (complex harmonic structure)
        fundamental_freq = 100
        for harmonic in range(1, 8):
            P_w[:, self.REESE] += self._gaussian_prior(
                freq_range,
                mean=fundamental_freq * harmonic,
                std=30 * harmonic
            ) * (0.7 ** harmonic)

        # Atmospheric sounds prior (mid-high frequency emphasis)
        P_w[:, self.ATMOS] = self._uniform_prior(freq_range, 1000, 8000)

        return P_w

    def _initialize_temporal_priors(self, n_frames: int) -> np.ndarray:
        """Initialize detailed time-dependent priors for DnB components."""
        P_h = np.zeros((self.n_components, n_frames))

        # Sub-bass continuity prior
        P_h[self.SUBBASS] = self._generate_continuity_prior(n_frames, smoothness=0.95)

        # Bass harmonics temporal prior
        P_h[self.BASS_HARM] = self._generate_continuity_prior(n_frames, smoothness=0.8)

        # Kick drum rhythmic prior (emphasize typical DnB patterns)
        P_h[self.KICK] = self._generate_rhythmic_prior(n_frames, pattern_length=4)

        # Snare rhythmic prior (typically on 2 and 4)
        P_h[self.SNARE] = self._generate_rhythmic_prior(n_frames, pattern_length=4, offset=2)

        # Hi-hat sparsity prior
        P_h[self.HIHAT] = self._generate_sparsity_prior(n_frames, sparsity=0.8)

        # Cymbal decay prior
        P_h[self.CYMBAL] = self._generate_decay_prior(n_frames, decay_rate=0.1)

        # Reese bass modulation prior
        P_h[self.REESE] = self._generate_modulation_prior(n_frames, mod_freq=0.5)

        # Atmospheric sounds slow evolution prior
        P_h[self.ATMOS] = self._generate_continuity_prior(n_frames, smoothness=0.98)

        return P_h

    def _gaussian_prior(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Generate Gaussian-shaped prior."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2)

    def _uniform_prior(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        """Generate uniform prior within frequency range."""
        return np.where((x >= low) & (x <= high), 1.0, 0.0)

    def _generate_rhythmic_prior(self, length: int, pattern_length: int, offset: int = 0) -> np.ndarray:
        """Generate rhythmic prior with specific pattern."""
        prior = np.zeros(length)
        pattern = np.zeros(pattern_length)
        pattern[offset::2] = 1.0  # Set rhythm pattern

        # Repeat pattern across time
        repetitions = length // pattern_length + 1
        extended_pattern = np.tile(pattern, repetitions)[:length]

        return extended_pattern

    def _generate_modulation_prior(self, length: int, mod_freq: float) -> np.ndarray:
        """Generate modulation prior for Reese bass."""
        t = np.linspace(0, length / self.sr, length)
        return 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))

    def _generate_decay_prior(self, length: int, decay_rate: float) -> np.ndarray:
        """Generate exponential decay prior."""
        return np.exp(-decay_rate * np.arange(length))

    def _apply_component_constraints(self, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply component-specific constraints."""
        # Sub-bass monotonicity constraint
        W[:, self.SUBBASS] = np.maximum.accumulate(W[:, self.SUBBASS][::-1])[::-1]

        # Reese bass harmonic constraint
        W[:, self.REESE] = self._enforce_harmonic_structure(W[:, self.REESE])

        # Percussion sparsity constraints
        H[self.KICK] = self._enforce_sparsity(H[self.KICK], threshold=0.3)
        H[self.SNARE] = self._enforce_sparsity(H[self.SNARE], threshold=0.3)
        H[self.HIHAT] = self._enforce_sparsity(H[self.HIHAT], threshold=0.4)

        return W, H

    def _enforce_harmonic_structure(self, w: np.ndarray) -> np.ndarray:
        """Enforce harmonic structure on spectral basis."""
        # Find peaks
        peaks, _ = signal.find_peaks(w)
        if len(peaks) > 1:
            # Enforce harmonic relationships
            fundamental = peaks[0]
            ideal_harmonics = fundamental * np.arange(1, len(peaks))
            w_harmonics = np.zeros_like(w)
            for i, harmonic in enumerate(ideal_harmonics):
                idx = np.argmin(np.abs(peaks - harmonic))
                w_harmonics[peaks[idx]] = w[peaks[idx]]
            return w_harmonics
        return w

    def _enforce_sparsity(self, h: np.ndarray, threshold: float) -> np.ndarray:
        """Enforce sparsity constraint on temporal activations."""
        mask = h > threshold * np.max(h)
        return h * mask

    def _generate_continuity_prior(self, length: int, smoothness: float) -> np.ndarray:
        """Generate temporal continuity prior with given smoothness."""
        prior = np.random.rand(length)
        # Apply smoothing filter
        window = signal.gaussian(11, std=smoothness * 5)
        window = window / window.sum()
        return signal.convolve(prior, window, mode='same')

    def _generate_sparsity_prior(self, length: int, sparsity: float) -> np.ndarray:
        """Generate sparse temporal prior."""
        prior = np.random.rand(length)
        threshold = np.percentile(prior, sparsity * 100)
        return np.where(prior > threshold, prior, 0)

    def decompose(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose input spectrogram into components using PSA.

        Args:
            X: Input magnitude spectrogram (n_freq, n_time)

        Returns:
            W: Spectral bases (n_freq, n_components)
            H: Temporal activations (n_components, n_time)
        """
        n_freq, n_time = X.shape

        # Initialize spectral and temporal priors
        P_w = self._initialize_spectral_priors(n_freq)
        P_h = self._initialize_temporal_priors(n_time)

        # Initialize W and H with priors
        W = P_w + 0.1 * np.random.rand(n_freq, self.n_components)
        H = P_h + 0.1 * np.random.rand(self.n_components, n_time)

        # Normalize
        W = W / (np.sum(W, axis=0, keepdims=True) + self.eps)
        H = H / (np.sum(H, axis=1, keepdims=True) + self.eps)

        # Iterative update
        for _ in range(self.n_iterations):
            # Update H
            WtX = np.dot(W.T, X)
            WtWH = np.dot(np.dot(W.T, W), H) + self.eps
            H *= np.sqrt(WtX / WtWH)

            # Update W
            XHt = np.dot(X, H.T)
            WHHt = np.dot(W, np.dot(H, H.T)) + self.eps
            W *= np.sqrt(XHt / WHHt)

            # Apply constraints
            W, H = self._apply_component_constraints(W, H)

            # Normalize
            W = W / (np.sum(W, axis=0, keepdims=True) + self.eps)
            H = H / (np.sum(H, axis=1, keepdims=True) + self.eps)

        return W, H

    def get_component_spectrograms(self, W: np.ndarray, H: np.ndarray) -> List[np.ndarray]:
        """
        Reconstruct individual component spectrograms.

        Args:
            W: Spectral bases (n_freq, n_components)
            H: Temporal activations (n_components, n_time)

        Returns:
            List of component spectrograms
        """
        components = []
        for i in range(self.n_components):
            component = np.outer(W[:, i], H[i, :])
            components.append(component)
        return components

    def get_component_audio(self,
                            components: List[np.ndarray],
                            phase: np.ndarray,
                            hop_length: int = 512) -> List[np.ndarray]:
        """
        Reconstruct time-domain signals for each component.

        Args:
            components: List of component spectrograms
            phase: Phase information from original STFT
            hop_length: STFT hop length

        Returns:
            List of time-domain signals
        """
        audio_signals = []
        for component in components:
            # Combine magnitude with original phase
            S_complex = component * np.exp(1j * phase)
            # Inverse STFT
            audio = librosa.istft(S_complex, hop_length=hop_length)
            audio_signals.append(audio)
        return audio_signals


