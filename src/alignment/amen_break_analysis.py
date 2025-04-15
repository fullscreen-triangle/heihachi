import numpy as np
import torch
import librosa
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AmenVariation:
    type: str  # 'original', 'vip', 'dubplate'
    confidence: float
    transformations: List[str]  # e.g., ['time_stretch', 'pitch_shift', 'chop']
    groove_score: float
    pattern_matches: Dict[str, float]  # Matching scores for different patterns


class AmenBreakAnalyzer:
    def __init__(self, template_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.hop_length = 512
        self.batch_size = 1024 * 1024  # 1MB chunks

        # Default template path or user-provided
        self.template_path = template_path or os.path.join(
            Path(__file__).parent, 'templates'
        )

        # Load templates and patterns
        self.templates = self._load_amen_templates()
        
        # Common Amen break chop patterns
        self.chop_patterns = {
            'classic': [1, 0, 1, 0, 1, 1, 0, 1],  # Classic pattern
            'think': [1, 1, 0, 1, 0, 1, 1, 0],    # Think break pattern
            'complex': [1, 0, 1, 1, 0, 1, 1, 1],  # Complex chop
            'staggered': [1, 0, 0, 1, 1, 0, 0, 1] # Staggered pattern
        }
        
        # Transformation detection thresholds
        self.thresholds = {
            'time_stretch': 0.15,  # 15% deviation from original tempo
            'pitch_shift': 2.0,    # 2 semitones
            'chop': 0.3,           # 30% pattern difference
            'vip': 0.7,            # 70% similarity for VIP detection
            'dubplate': 0.5        # 50% similarity for dubplate detection
        }

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze audio for Amen break characteristics and variations.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing detected variations and transformations
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Extract drum patterns
        patterns = self._extract_drum_patterns(audio)
        
        # Detect variations and transformations
        variations = self._detect_variations(patterns)
        
        # Analyze groove characteristics
        groove_analysis = self._analyze_groove(patterns)
        
        # Determine if this is likely a VIP or dubplate
        variation_type = self._classify_variation_type(
            patterns, variations, groove_analysis
        )
        
        return {
            'variation': self._variation_to_dict(variation_type),
            'patterns': patterns,
            'groove_analysis': groove_analysis,
            'transformations': variations
        }

    def _extract_drum_patterns(self, audio: np.ndarray) -> Dict:
        """Extract drum patterns from audio using template matching."""
        # Compute STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=self.hop_length)
        S = np.abs(D)

        patterns = {}
        
        # Extract kick pattern
        patterns['kick'] = self._extract_component_pattern(
            S, self.templates['spectral']['kick']
        )
        
        # Extract snare pattern
        patterns['snare'] = self._extract_component_pattern(
            S, self.templates['spectral']['snare']
        )
        
        # Extract hihat pattern
        patterns['hihat'] = self._extract_component_pattern(
            S, self.templates['spectral']['hihat']
        )
        
        # Detect tempo and grid
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        patterns['tempo'] = tempo
        patterns['beat_frames'] = beat_frames
        
        return patterns

    def _extract_component_pattern(self, S: np.ndarray, 
                                 template: np.ndarray) -> np.ndarray:
        """Extract specific drum component pattern using template matching."""
        # Convert to tensors for GPU acceleration
        S_tensor = torch.from_numpy(S).to(self.device)
        template_tensor = torch.from_numpy(template).to(self.device)
        
        # Compute correlation
        correlation = torch.conv1d(
            S_tensor.unsqueeze(0),
            template_tensor.unsqueeze(0).unsqueeze(0),
            padding='same'
        )
        
        # Threshold and normalize
        pattern = torch.relu(correlation)
        pattern = pattern / (torch.max(pattern) + 1e-8)
        
        return pattern.squeeze().cpu().numpy()

    def _detect_variations(self, patterns: Dict) -> Dict:
        """Detect variations and transformations in the patterns."""
        variations = {
            'time_stretching': self._detect_time_stretching(patterns),
            'pitch_shifting': self._detect_pitch_shifting(patterns),
            'chopping': self._detect_chopping(patterns),
            'pattern_complexity': self._analyze_pattern_complexity(patterns)
        }
        
        return variations

    def _detect_time_stretching(self, patterns: Dict) -> Dict:
        """Detect time stretching transformations."""
        results = {}
        original_tempo = 168.0  # Original Amen break tempo
        
        # Calculate tempo ratio
        tempo_ratio = patterns['tempo'] / original_tempo
        
        # Check against common time stretch factors
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            similarity = 1.0 - abs(tempo_ratio - factor)
            if similarity > (1.0 - self.thresholds['time_stretch']):
                results[f'factor_{factor}'] = float(similarity)
        
        return results

    def _detect_pitch_shifting(self, patterns: Dict) -> Dict:
        """Detect pitch shifting transformations."""
        results = {}
        
        # Compare spectral centroid with templates
        for semitones, template in self.templates['pitch_shifted'].items():
            similarity = self._compute_spectral_similarity(
                patterns['snare'],
                template
            )
            if similarity > (1.0 - self.thresholds['pitch_shift']):
                results[f'semitones_{semitones}'] = float(similarity)
        
        return results

    def _detect_chopping(self, patterns: Dict) -> Dict:
        """Detect chopping patterns and variations."""
        results = {}
        
        # Compare with known chop patterns
        for name, pattern in self.chop_patterns.items():
            similarity = self._compare_chop_pattern(
                patterns['snare'],
                np.array(pattern)
            )
            if similarity > (1.0 - self.thresholds['chop']):
                results[name] = float(similarity)
        
        return results

    def _analyze_pattern_complexity(self, patterns: Dict) -> float:
        """Analyze the complexity of the drum patterns."""
        # Combine patterns
        combined = (
            patterns['kick'] +
            patterns['snare'] +
            patterns['hihat']
        )
        
        # Calculate entropy and syncopation
        entropy = librosa.feature.spectral_entropy(S=combined)
        
        return float(np.mean(entropy))

    def _analyze_groove(self, patterns: Dict) -> Dict:
        """Analyze groove characteristics."""
        # Calculate swing ratio
        swing = self._calculate_swing(patterns)
        
        # Calculate syncopation
        syncopation = self._calculate_syncopation(patterns)
        
        # Calculate ghost note presence
        ghost_notes = self._detect_ghost_notes(patterns)
        
        return {
            'swing': float(swing),
            'syncopation': float(syncopation),
            'ghost_notes': ghost_notes,
            'groove_score': float((swing + syncopation) / 2)
        }

    def _calculate_swing(self, patterns: Dict) -> float:
        """Calculate swing ratio from patterns."""
        # Focus on hihat pattern for swing
        hihat = patterns['hihat']
        
        # Find consecutive eighth notes
        pairs = zip(hihat[::2], hihat[1::2])
        ratios = [b/a if a > 0 else 1.0 for a, b in pairs]
        
        return np.mean(ratios) if ratios else 1.0

    def _calculate_syncopation(self, patterns: Dict) -> float:
        """Calculate syncopation level from patterns."""
        # Combine patterns
        combined = (
            patterns['kick'] +
            patterns['snare'] * 0.8 +
            patterns['hihat'] * 0.5
        )
        
        # Calculate offbeat energy ratio
        onbeat = combined[::2]
        offbeat = combined[1::2]
        
        return np.mean(offbeat) / (np.mean(onbeat) + 1e-8)

    def _detect_ghost_notes(self, patterns: Dict) -> Dict:
        """Detect ghost notes in the patterns."""
        ghost_notes = {}
        
        for component in ['kick', 'snare', 'hihat']:
            pattern = patterns[component]
            # Find weak hits
            threshold = np.mean(pattern) * 0.5
            ghost_mask = (pattern > 0) & (pattern < threshold)
            ghost_notes[component] = {
                'count': int(np.sum(ghost_mask)),
                'positions': np.where(ghost_mask)[0].tolist()
            }
        
        return ghost_notes

    def _classify_variation_type(self, patterns: Dict,
                               variations: Dict,
                               groove: Dict) -> AmenVariation:
        """Classify whether this is original, VIP, or dubplate."""
        # Calculate base similarity to original Amen break
        base_similarity = self._compute_pattern_similarity(
            patterns,
            self.templates['original']
        )
        
        # Calculate transformation complexity
        transformation_score = (
            len(variations['time_stretching']) +
            len(variations['pitch_shifting']) +
            len(variations['chopping'])
        ) / 10.0  # Normalize to 0-1
        
        # Determine type based on characteristics
        if base_similarity > self.thresholds['vip']:
            variation_type = 'vip'
        elif transformation_score > 0.7 and groove['groove_score'] > 0.8:
            variation_type = 'dubplate'
        else:
            variation_type = 'original'
        
        # Collect detected transformations
        transformations = []
        for category, results in variations.items():
            if results:  # If any transformations detected
                transformations.append(category)
        
        return AmenVariation(
            type=variation_type,
            confidence=base_similarity * (1.0 + transformation_score),
            transformations=transformations,
            groove_score=groove['groove_score'],
            pattern_matches={
                name: score
                for category in variations.values()
                for name, score in category.items()
                if isinstance(category, dict)
            }
        )

    def _compute_pattern_similarity(self, patterns: Dict,
                                  template: np.ndarray) -> float:
        """Compute similarity between patterns and template."""
        # Convert patterns to feature vector
        pattern_features = np.concatenate([
            patterns['kick'],
            patterns['snare'],
            patterns['hihat']
        ])
        
        # Normalize
        pattern_features = librosa.util.normalize(pattern_features)
        template = librosa.util.normalize(template)
        
        # Compute correlation
        correlation = np.correlate(pattern_features, template, mode='full')
        
        return float(np.max(correlation))

    def _compute_spectral_similarity(self, pattern: np.ndarray,
                                   template: np.ndarray) -> float:
        """Compute similarity between spectral patterns."""
        # Normalize patterns
        pattern = librosa.util.normalize(pattern)
        template = librosa.util.normalize(template)
        
        # Compute cosine similarity
        similarity = np.dot(pattern, template) / (
            np.linalg.norm(pattern) * np.linalg.norm(template)
        )
        
        return float(similarity)

    def _compare_chop_pattern(self, pattern: np.ndarray,
                            template: np.ndarray) -> float:
        """Compare extracted pattern with chop template."""
        # Quantize pattern to binary
        binary_pattern = (pattern > np.mean(pattern)).astype(float)
        
        # Compute correlation
        correlation = np.correlate(binary_pattern, template, mode='full')
        
        return float(np.max(correlation))

    def _variation_to_dict(self, variation: AmenVariation) -> Dict:
        """Convert AmenVariation to dictionary."""
        return {
            'type': variation.type,
            'confidence': float(variation.confidence),
            'transformations': variation.transformations,
            'groove_score': float(variation.groove_score),
            'pattern_matches': {
                k: float(v) for k, v in variation.pattern_matches.items()
            }
        }

    def _load_amen_templates(self) -> dict:
        """
        Load Amen Break templates and their variations.
        Returns a dictionary containing:
        - Original Amen Break template
        - Time-stretched variations
        - Pitch-shifted variations
        - Common rhythmic patterns
        """
        templates = {}

        # Load original Amen Break template
        original_path = os.path.join(self.template_path, 'amen_original.wav')
        if os.path.exists(original_path):
            audio, _ = librosa.load(original_path, sr=self.sr)
            # Compute STFT
            stft = librosa.stft(audio)
            templates['original'] = torch.from_numpy(np.abs(stft)).to(self.device)
        else:
            # Create synthetic template if file doesn't exist
            templates['original'] = self._create_synthetic_template()

        # Create standard rhythm patterns
        templates['patterns'] = {
            'basic': self._create_basic_pattern(),
            'ghost_notes': self._create_ghost_note_pattern(),
            'syncopated': self._create_syncopated_pattern()
        }

        # Create time-stretched variations
        templates['time_stretched'] = {}
        stretch_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        for factor in stretch_factors:
            templates['time_stretched'][factor] = self._create_time_stretched_template(
                templates['original'], factor
            )

        # Create pitch-shifted variations
        templates['pitch_shifted'] = {}
        semitones = range(-3, 4)  # -3 to +3 semitones
        for st in semitones:
            templates['pitch_shifted'][st] = self._create_pitch_shifted_template(
                templates['original'], st
            )

        # Add spectral templates
        templates['spectral'] = {
            'kick': self._create_kick_template(),
            'snare': self._create_snare_template(),
            'hihat': self._create_hihat_template()
        }

        return templates

    def _create_synthetic_template(self) -> torch.Tensor:
        """Create a synthetic Amen Break template based on typical characteristics."""
        # Create basic template shape (2048 frequency bins x 1024 time frames)
        template = np.zeros((2048, 1024), dtype=np.float32)

        # Add characteristic frequency components
        # Kick drum (50-100 Hz)
        template[5:10, ::4] = 1.0

        # Snare (200-400 Hz)
        template[20:40, 2::4] = 0.8

        # Hi-hats (5000-10000 Hz)
        template[500:1000, ::2] = 0.4

        return torch.from_numpy(template).to(self.device)

    def _create_basic_pattern(self) -> np.ndarray:
        """Create basic Amen Break rhythm pattern."""
        pattern = np.zeros(16)  # 16th note grid for one bar
        # Kick pattern
        pattern[0] = 1.0  # 1
        pattern[8] = 0.8  # 3
        # Snare pattern
        pattern[4] = 1.0  # 2
        pattern[12] = 1.0  # 4
        # Hi-hat pattern
        pattern[2::2] = 0.6  # eighth notes
        return pattern

    def _create_ghost_note_pattern(self) -> np.ndarray:
        """Create Amen Break pattern with ghost notes."""
        pattern = self._create_basic_pattern()
        # Add ghost notes
        ghost_positions = [3, 7, 11, 15]
        pattern[ghost_positions] = 0.3
        return pattern

    def _create_syncopated_pattern(self) -> np.ndarray:
        """Create syncopated Amen Break pattern."""
        pattern = np.zeros(16)
        # Syncopated rhythm
        syncopated_positions = [0, 3, 6, 10, 12, 15]
        pattern[syncopated_positions] = 1.0
        return pattern

    def _create_kick_template(self) -> torch.Tensor:
        """Create spectral template for kick drum."""
        template = np.zeros(2048)
        # Fundamental (50-100 Hz)
        template[5:10] = 1.0
        # First harmonic
        template[10:20] = 0.5
        return torch.from_numpy(template).to(self.device)

    def _create_snare_template(self) -> torch.Tensor:
        """Create spectral template for snare drum."""
        template = np.zeros(2048)
        # Body (200-400 Hz)
        template[20:40] = 0.8
        # Noise component
        template[40:200] = 0.4
        return torch.from_numpy(template).to(self.device)

    def _create_hihat_template(self) -> torch.Tensor:
        """Create spectral template for hi-hat."""
        template = np.zeros(2048)
        # High frequency content
        template[500:1000] = 0.6
        return torch.from_numpy(template).to(self.device)

    def _create_time_stretched_template(
            self,
            original: torch.Tensor,
            factor: float
    ) -> torch.Tensor:
        """Create time-stretched variation of template."""
        # Convert to numpy for processing
        orig_np = original.cpu().numpy()

        # Apply time stretching
        stretched = librosa.effects.time_stretch(orig_np, rate=factor)

        # Convert back to tensor
        return torch.from_numpy(stretched).to(self.device)

    def _create_pitch_shifted_template(
            self,
            original: torch.Tensor,
            semitones: int
    ) -> torch.Tensor:
        """Create pitch-shifted variation of template."""
        # Convert to numpy for processing
        orig_np = original.cpu().numpy()

        # Apply pitch shifting
        shifted = librosa.effects.pitch_shift(
            orig_np,
            sr=self.sr,
            n_steps=semitones
        )

        # Convert back to tensor
        return torch.from_numpy(shifted).to(self.device)
