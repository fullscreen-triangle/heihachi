import numpy as np
import torch
from numba import jit
from typing import Dict, List, Tuple, Optional
import librosa
import logging

logger = logging.getLogger(__name__)

class AmenBreakTemplate:
    """Standard Amen break template and variations with detailed analysis capabilities."""
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.bpm = 138  # Standard Amen break tempo
        self.duration = 2.2  # Standard length in seconds
        
        # Pattern characteristics
        self.kick_positions = [0, 8]  # Standard kick positions (in 16ths)
        self.snare_positions = [4, 12]  # Standard snare positions
        self.hihat_positions = [2, 6, 10, 14]  # Standard hi-hat positions
        
        # Drum component characteristics
        self.drum_characteristics = {
            'kick': {
                'freq_range': (40, 100),  # Hz
                'duration': 0.1,  # seconds
                'attack_time': 0.005
            },
            'snare': {
                'freq_range': (200, 2000),
                'duration': 0.08,
                'attack_time': 0.001
            },
            'hihat': {
                'freq_range': (5000, 15000),
                'duration': 0.05,
                'attack_time': 0.0005
            }
        }
        
        # Initialize patterns
        self.pattern = self._create_standard_pattern()
        self.component_patterns = self._create_component_patterns()
        self.variations = self._create_variations()
        
    def _create_standard_pattern(self) -> np.ndarray:
        """Create the standard Amen break pattern with separate drum components."""
        pattern_length = int(self.duration * self.sr)
        pattern = np.zeros(pattern_length)
        
        # Add drum components with different amplitudes and envelopes
        for pos in self.kick_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('kick') * 1.0
            end_idx = min(idx + len(envelope), len(pattern))
            pattern[idx:end_idx] = envelope[:end_idx-idx]
            
        for pos in self.snare_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('snare') * 0.8
            end_idx = min(idx + len(envelope), len(pattern))
            pattern[idx:end_idx] = envelope[:end_idx-idx]
            
        for pos in self.hihat_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('hihat') * 0.6
            end_idx = min(idx + len(envelope), len(pattern))
            pattern[idx:end_idx] = envelope[:end_idx-idx]
            
        return pattern
        
    def _create_drum_envelope(self, drum_type: str) -> np.ndarray:
        """Create realistic drum hit envelope."""
        chars = self.drum_characteristics[drum_type]
        duration_samples = int(chars['duration'] * self.sr)
        envelope = np.zeros(duration_samples)
        
        # Attack phase
        attack_samples = int(chars['attack_time'] * self.sr)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_samples = duration_samples - attack_samples
        envelope[attack_samples:] = np.exp(-np.linspace(0, 5, decay_samples))
        
        return envelope
        
    def _create_component_patterns(self) -> Dict[str, np.ndarray]:
        """Create separate patterns for each drum component."""
        pattern_length = int(self.duration * self.sr)
        components = {}
        
        # Kick pattern
        kick = np.zeros(pattern_length)
        for pos in self.kick_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('kick')
            end_idx = min(idx + len(envelope), len(kick))
            kick[idx:end_idx] = envelope[:end_idx-idx]
        components['kick'] = kick
        
        # Snare pattern
        snare = np.zeros(pattern_length)
        for pos in self.snare_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('snare')
            end_idx = min(idx + len(envelope), len(snare))
            snare[idx:end_idx] = envelope[:end_idx-idx]
        components['snare'] = snare
        
        # Hi-hat pattern
        hihat = np.zeros(pattern_length)
        for pos in self.hihat_positions:
            idx = int((pos/16) * pattern_length)
            envelope = self._create_drum_envelope('hihat')
            end_idx = min(idx + len(envelope), len(hihat))
            hihat[idx:end_idx] = envelope[:end_idx-idx]
        components['hihat'] = hihat
        
        return components
        
    def _create_variations(self) -> List[np.ndarray]:
        """Create comprehensive set of Amen break variations."""
        variations = []
        
        # 1. Half-time variation
        half_time = np.repeat(self.pattern[::2], 2)
        variations.append(half_time)
        
        # 2. Double-time variation
        double_time = np.repeat(self.pattern, 2)[:len(self.pattern)]
        variations.append(double_time)
        
        # 3. Shuffled variation (standard shuffle)
        shuffle = self.pattern.copy()
        shuffle_points = [2, 6, 10, 14]
        for point in shuffle_points:
            pos = int((point/16) * len(shuffle))
            if pos + 100 < len(shuffle):
                shuffle[pos:pos+100] = shuffle[pos:pos+100] * 0.8
        variations.append(shuffle)
        
        # 4. Ghost note variation
        ghost_notes = self.pattern.copy()
        ghost_positions = [1, 3, 5, 7, 9, 11, 13, 15]
        for pos in ghost_positions:
            idx = int((pos/16) * len(ghost_notes))
            if idx + 100 < len(ghost_notes):
                ghost_notes[idx:idx+100] = 0.4  # Lower amplitude for ghost notes
        variations.append(ghost_notes)
        
        # 5. Syncopated variation
        syncopated = self.pattern.copy()
        syncopation_shifts = [(4, 5), (12, 13)]  # Shift snare hits
        for orig, new in syncopation_shifts:
            orig_pos = int((orig/16) * len(syncopated))
            new_pos = int((new/16) * len(syncopated))
            if new_pos + 100 < len(syncopated):
                syncopated[new_pos:new_pos+100] = syncopated[orig_pos:orig_pos+100]
                syncopated[orig_pos:orig_pos+100] = 0
        variations.append(syncopated)
        
        # 6. Triplet variation
        triplet = np.zeros(len(self.pattern))
        triplet_positions = [0, 3, 5, 8, 11, 13]
        for pos in triplet_positions:
            idx = int((pos/16) * len(triplet))
            if idx + 100 < len(triplet):
                triplet[idx:idx+100] = 1.0
        variations.append(triplet)
        
        # 7. Reversed variation
        reversed_pattern = np.flip(self.pattern)
        variations.append(reversed_pattern)
        
        # 8. Stutter variation
        stutter = self.pattern.copy()
        stutter_points = [4, 12]  # Stutter on snare hits
        for point in stutter_points:
            pos = int((point/16) * len(stutter))
            if pos + 200 < len(stutter):
                # Create rapid repetition
                stutter[pos:pos+200] = np.repeat(stutter[pos:pos+100], 2) * 0.8
        variations.append(stutter)
        
        # New variations
        variations.extend([
            self._create_jungle_edit(),
            self._create_stretched(),
            self._create_layered(),
            self._create_filtered(),
            self._create_phased(),
            self._create_glitch()
        ])
        
        return variations
        
    def _create_jungle_edit(self) -> np.ndarray:
        """Create jungle-style edit with time-stretched snares."""
        jungle = self.pattern.copy()
        snare_positions = [4, 12]  # Snare positions
        
        for pos in snare_positions:
            idx = int((pos/16) * len(jungle))
            if idx + 200 < len(jungle):
                # Time-stretch snare hits
                snare_hit = jungle[idx:idx+100]
                stretched_snare = np.repeat(snare_hit, 2)
                jungle[idx:idx+200] = stretched_snare * 0.9
        
        return jungle
        
    def _create_stretched(self) -> np.ndarray:
        """Create time-stretched variation."""
        stretch_factor = 1.5
        return librosa.effects.time_stretch(self.pattern, rate=stretch_factor)
        
    def _create_layered(self) -> np.ndarray:
        """Create layered variation with overlapping components."""
        layered = self.pattern.copy()
        
        # Add delayed version
        delay_samples = int(0.125 * self.sr)  # 125ms delay
        delayed = np.roll(layered, delay_samples) * 0.6
        
        return layered + delayed
        
    def _create_filtered(self) -> np.ndarray:
        """Create filtered variation emphasizing different frequency bands."""
        from scipy.signal import butter, filtfilt
        
        filtered = np.zeros_like(self.pattern)
        
        # Apply different filters to different components
        for comp, pattern in self.component_patterns.items():
            freq_range = self.drum_characteristics[comp]['freq_range']
            nyquist = self.sr / 2
            b, a = butter(4, [freq_range[0]/nyquist, freq_range[1]/nyquist], btype='band')
            filtered_comp = filtfilt(b, a, pattern)
            filtered += filtered_comp
            
        return filtered
        
    def _create_phased(self) -> np.ndarray:
        """Create phased variation with subtle timing variations."""
        phased = self.pattern.copy()
        
        # Create phase modulation
        t = np.linspace(0, self.duration, len(phased))
        phase_mod = np.sin(2 * np.pi * 0.5 * t) * 100  # 0.5 Hz modulation
        
        # Apply phase modulation
        indices = np.arange(len(phased))
        mod_indices = indices + phase_mod.astype(int)
        mod_indices = np.clip(mod_indices, 0, len(phased)-1)
        
        return phased[mod_indices]
        
    def _create_glitch(self) -> np.ndarray:
        """Create glitch variation with random micro-edits."""
        glitch = self.pattern.copy()
        
        # Add random glitch effects
        n_glitches = 4
        for _ in range(n_glitches):
            # Random position
            pos = np.random.randint(0, len(glitch)-100)
            # Random effect
            effect = np.random.choice(['repeat', 'reverse', 'silence'])
            
            if effect == 'repeat':
                glitch[pos:pos+100] = np.repeat(glitch[pos:pos+50], 2)
            elif effect == 'reverse':
                glitch[pos:pos+100] = np.flip(glitch[pos:pos+100])
            else:  # silence
                glitch[pos:pos+100] = 0
                
        return glitch

    def get_variation_name(self, index: int) -> str:
        """Get the name of a variation by index."""
        names = [
            "Standard",
            "Half-time",
            "Double-time",
            "Shuffled",
            "Ghost Notes",
            "Syncopated",
            "Triplet",
            "Reversed",
            "Stutter",
            "Jungle Edit",
            "Stretched",
            "Layered",
            "Filtered",
            "Phased",
            "Glitch"
        ]
        return names[index] if 0 <= index < len(names) else "Unknown"

    def get_variation_characteristics(self, index: int) -> Dict:
        """Get detailed characteristics of a variation."""
        characteristics = {
            0: {"type": "Standard", "complexity": 1.0, "energy": 1.0, "groove": 1.0},
            1: {"type": "Half-time", "complexity": 0.7, "energy": 0.8, "groove": 0.9},
            2: {"type": "Double-time", "complexity": 1.3, "energy": 1.2, "groove": 0.8},
            3: {"type": "Shuffled", "complexity": 1.1, "energy": 0.9, "groove": 1.1},
            4: {"type": "Ghost Notes", "complexity": 1.2, "energy": 0.85, "groove": 1.2},
            5: {"type": "Syncopated", "complexity": 1.25, "energy": 1.1, "groove": 1.15},
            6: {"type": "Triplet", "complexity": 1.4, "energy": 1.0, "groove": 1.05},
            7: {"type": "Reversed", "complexity": 1.15, "energy": 1.0, "groove": 0.7},
            8: {"type": "Stutter", "complexity": 1.3, "energy": 1.15, "groove": 0.9},
            9: {"type": "Jungle Edit", "complexity": 1.4, "energy": 1.2, "groove": 1.1},
            10: {"type": "Stretched", "complexity": 1.1, "energy": 0.9, "groove": 0.8},
            11: {"type": "Layered", "complexity": 1.5, "energy": 1.3, "groove": 1.2},
            12: {"type": "Filtered", "complexity": 1.2, "energy": 0.95, "groove": 1.0},
            13: {"type": "Phased", "complexity": 1.3, "energy": 1.1, "groove": 1.15},
            14: {"type": "Glitch", "complexity": 1.6, "energy": 1.25, "groove": 0.7}
        }
        return characteristics.get(index, {"type": "Unknown", "complexity": 1.0, "energy": 1.0, "groove": 1.0})

    def get_component_analysis(self, variation_index: int) -> Dict:
        """Get detailed analysis of drum components for a variation."""
        variation = self.variations[variation_index] if variation_index < len(self.variations) else self.pattern
        
        analysis = {}
        for comp, pattern in self.component_patterns.items():
            # Compute correlation with variation
            correlation = np.correlate(variation, pattern, mode='full')
            max_corr = np.max(correlation) / len(pattern)
            
            # Detect hits
            hits = librosa.onset.onset_detect(
                y=pattern, 
                sr=self.sr,
                units='samples'
            )
            
            # Analyze velocity
            velocities = [pattern[hit] for hit in hits]
            
            analysis[comp] = {
                'presence': max_corr,
                'n_hits': len(hits),
                'avg_velocity': np.mean(velocities) if velocities else 0,
                'velocity_variance': np.var(velocities) if velocities else 0
            }
            
        return analysis

class SequenceAligner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amen_template = AmenBreakTemplate()
        
    def align_sequences(self, audio: np.ndarray, sr: int = 44100) -> Dict:
        """
        Align input audio with Amen break patterns.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Dict containing:
            - alignments: List of alignment points
            - confidence: Confidence scores for alignments
            - variation_type: Detected variation type
            - tempo_scale: Tempo scaling factor
        """
        # Extract onset envelope
        onset_env = self._get_onset_envelope(audio, sr)
        
        # Find potential Amen break segments
        segments = self._find_amen_segments(onset_env, sr)
        
        # Align each segment
        alignments = []
        confidences = []
        variations = []
        tempo_scales = []
        
        for segment in segments:
            # Get segment audio
            start_idx = int(segment['start'] * sr)
            end_idx = int(segment['end'] * sr)
            segment_audio = audio[start_idx:end_idx]
            
            # Align with templates
            alignment = self._align_segment(segment_audio, sr)
            alignments.append(alignment['points'])
            confidences.append(alignment['confidence'])
            variations.append(alignment['variation'])
            tempo_scales.append(alignment['tempo_scale'])
        
        return {
            'alignments': alignments,
            'confidence': confidences,
            'variation_types': variations,
            'tempo_scales': tempo_scales,
            'segments': segments
        }

    def _get_onset_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract onset envelope from audio."""
        # Check for empty audio or None sample rate
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio provided to onset envelope extraction")
            return np.array([])  # Return empty array
            
        if sr is None:
            logger.warning("Sample rate is None, using default 44100")
            sr = 44100  # Set default sample rate
            
        # Multi-band onset detection
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=sr,
            hop_length=512,
            aggregate=np.median,
            n_mels=128,
            fmax=8000,
            window=np.hanning  # Add Hann window to prevent spectral leakage warning
        )
        return onset_env

    def _find_amen_segments(self, onset_env: np.ndarray, sr: int) -> List[Dict]:
        """Find segments that potentially contain Amen break."""
        segments = []
        
        # Parameters
        min_length = 1.8  # Minimum segment length in seconds
        max_length = 2.5  # Maximum segment length in seconds
        hop_length = 512  # From onset detection
        
        # Convert to frames
        min_frames = int((min_length * sr) / hop_length)
        max_frames = int((max_length * sr) / hop_length)
        
        # Find segments with high rhythmic similarity to Amen pattern
        frame_idx = 0
        while frame_idx < len(onset_env) - min_frames:
            # Get segment onset pattern
            segment = onset_env[frame_idx:frame_idx + max_frames]
            
            # Compare with template
            similarity = self._compute_pattern_similarity(
                segment, 
                self.amen_template.pattern
            )
            
            if similarity > 0.7:  # Threshold for potential match
                segments.append({
                    'start': (frame_idx * hop_length) / sr,
                    'end': ((frame_idx + len(segment)) * hop_length) / sr,
                    'similarity': similarity
                })
                frame_idx += max_frames
            else:
                frame_idx += min_frames // 2  # Overlap for better detection
                
        return segments

    def _align_segment(self, segment: np.ndarray, sr: int) -> Dict:
        """Align a single segment with Amen break templates."""
        best_alignment = None
        best_score = -np.inf
        
        # Try each template variation
        templates = [self.amen_template.pattern] + self.amen_template.variations
        for i, template in enumerate(templates):
            # Compute DTW
            cost_matrix = self._compute_cost_matrix(
                torch.from_numpy(segment).to(self.device),
                torch.from_numpy(template).to(self.device)
            )
            
            # Find optimal path
            path = self._find_optimal_path(cost_matrix.cpu().numpy())
            
            # Compute alignment score
            score = self._compute_alignment_score(cost_matrix, path)
            
            if score > best_score:
                best_score = score
                best_alignment = {
                    'points': path,
                    'confidence': score,
                    'variation': i,
                    'tempo_scale': len(segment) / len(template)
                }
        
        return best_alignment

    @staticmethod
    def _compute_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compute similarity between two onset patterns."""
        # Normalize patterns
        pattern1 = pattern1 / np.max(pattern1) if np.max(pattern1) > 0 else pattern1
        pattern2 = pattern2 / np.max(pattern2) if np.max(pattern2) > 0 else pattern2
        
        # Compute correlation
        correlation = np.correlate(pattern1, pattern2, mode='full')
        return np.max(correlation) / len(pattern1)

    def _compute_cost_matrix(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DTW cost matrix with GPU acceleration."""
        N, M = source.shape[0], target.shape[0]
        cost_matrix = torch.zeros((N, M), device=self.device)

        for i in range(N):
            for j in range(M):
                cost_matrix[i, j] = torch.norm(source[i] - target[j])

        return cost_matrix

    @staticmethod
    @jit(nopython=True)
    def _find_optimal_path(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find optimal warping path using dynamic programming."""
        N, M = cost_matrix.shape
        D = np.zeros((N + 1, M + 1))
        D[0, :] = np.inf
        D[:, 0] = np.inf
        D[0, 0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                D[i, j] = cost_matrix[i - 1, j - 1] + min(
                    D[i - 1, j],    # insertion
                    D[i, j - 1],    # deletion
                    D[i - 1, j - 1] # match
                )

        # Backtrack to find path
        path = []
        i, j = N, M
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            if D[i - 1, j - 1] <= D[i - 1, j] and D[i - 1, j - 1] <= D[i, j - 1]:
                i, j = i - 1, j - 1
            elif D[i - 1, j] <= D[i, j - 1]:
                i -= 1
            else:
                j -= 1

        return path[::-1]

    def _compute_alignment_score(self, cost_matrix: torch.Tensor, path: List[Tuple[int, int]]) -> float:
        """Compute alignment quality score."""
        path_costs = [cost_matrix[i, j].item() for i, j in path]
        return 1.0 / (1.0 + np.mean(path_costs))  # Normalize to [0, 1]

    def _compute_warping(self, path: List[Tuple[int, int]], source_len: int, target_len: int) -> np.ndarray:
        """Compute warping function from alignment path."""
        warping = np.zeros(source_len)
        curr_target_idx = 0

        for i in range(source_len):
            while curr_target_idx < len(path) and path[curr_target_idx][0] < i:
                curr_target_idx += 1

            if curr_target_idx < len(path):
                warping[i] = path[curr_target_idx][1]
            else:
                warping[i] = target_len - 1

        return warping
