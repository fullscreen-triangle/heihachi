import librosa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import butter, filtfilt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    score: float
    component_scores: Dict[str, float]
    transformations: List[str]
    confidence: float
    details: Dict


class CompositeSimilarity:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.hop_length = 512
        
        # Default weights for different similarity aspects
        self.weights = weights or {
            'spectral': 0.25,
            'rhythmic': 0.25,
            'timbral': 0.2,
            'structural': 0.15,
            'groove': 0.15
        }
        
        # Frequency bands for component-wise analysis
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000)
        }
        
        # Thresholds for transformation detection
        self.thresholds = {
            'time_stretch': 0.15,
            'pitch_shift': 2.0,
            'spectral_flux': 0.2
        }

    def analyze(self, source: np.ndarray, target: np.ndarray) -> Dict:
        """Analyze similarity between two audio segments.
        
        Args:
            source (np.ndarray): Source audio data
            target (np.ndarray): Target audio data
            
        Returns:
            Dict: Analysis results containing similarity scores and details
        """
        # Convert to mono if stereo
        if source.ndim > 1:
            source = np.mean(source, axis=0)
        if target.ndim > 1:
            target = np.mean(target, axis=0)

        # Compute component-wise similarities
        with ThreadPoolExecutor() as executor:
            futures = {
                'spectral': executor.submit(
                    self._spectral_similarity, source, target
                ),
                'rhythmic': executor.submit(
                    self._rhythmic_similarity, source, target
                ),
                'timbral': executor.submit(
                    self._timbral_similarity, source, target
                ),
                'structural': executor.submit(
                    self._structural_similarity, source, target
                ),
                'groove': executor.submit(
                    self._groove_similarity, source, target
                )
            }

            component_scores = {
                k: f.result() for k, f in futures.items()
            }

        # Detect transformations
        transformations = self._detect_transformations(source, target)
        
        # Calculate overall similarity
        similarity = self._weighted_combination(component_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            component_scores,
            transformations
        )
        
        return self._create_result(
            similarity,
            component_scores,
            transformations,
            confidence
        )

    def _spectral_similarity(self, source: np.ndarray, 
                           target: np.ndarray) -> Dict[str, float]:
        """Compute spectral similarity with band-wise analysis."""
        similarities = {}
        
        # Compute full-band spectral similarity
        source_spec = np.abs(librosa.stft(source))
        target_spec = np.abs(librosa.stft(target))
        
        similarities['full_band'] = float(
            1 - cdist(source_spec.T, target_spec.T, metric='cosine').mean()
        )
        
        # Compute band-wise similarities
        for band_name, (low, high) in self.bands.items():
            try:
                # Apply bandpass filters using scipy
                nyquist = self.sr // 2
                b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
                source_filtered = filtfilt(b, a, source)
                target_filtered = filtfilt(b, a, target)
                
                # Trim the filtered audio
                source_band = librosa.effects.trim(source_filtered)[0]
                target_band = librosa.effects.trim(target_filtered)[0]
                
                # Compute band-specific spectrograms
                source_band_spec = np.abs(librosa.stft(source_band))
                target_band_spec = np.abs(librosa.stft(target_band))
                
                # Calculate similarity
                similarities[band_name] = float(
                    1 - cdist(
                        source_band_spec.T,
                        target_band_spec.T,
                        metric='cosine'
                    ).mean()
                )
            except Exception as e:
                print(f"Error in band {band_name}: {str(e)}")
                similarities[band_name] = 0.0
        
        return similarities

    def _rhythmic_similarity(self, source: np.ndarray,
                           target: np.ndarray) -> Dict[str, float]:
        """Compute rhythmic similarity with detailed analysis."""
        similarities = {}
        
        # Compute onset envelopes for different bands
        source_onsets = self._compute_multiband_onsets(source)
        target_onsets = self._compute_multiband_onsets(target)
        
        # Calculate band-wise rhythmic similarities
        for band in self.bands:
            if band in source_onsets and band in target_onsets:
                correlation = np.correlate(
                    source_onsets[band],
                    target_onsets[band],
                    mode='full'
                )
                similarities[band] = float(np.max(correlation))
        
        # Compute tempo-based similarity
        source_tempo = librosa.beat.tempo(y=source, sr=self.sr)[0]
        target_tempo = librosa.beat.tempo(y=target, sr=self.sr)[0]
        
        similarities['tempo'] = float(
            1.0 - abs(source_tempo - target_tempo) / max(source_tempo, target_tempo)
        )
        
        return similarities

    def _timbral_similarity(self, source: np.ndarray,
                          target: np.ndarray) -> Dict[str, float]:
        """Compute timbral similarity using multiple features."""
        similarities = {}
        
        # Compute MFCCs
        source_mfcc = librosa.feature.mfcc(y=source, sr=self.sr)
        target_mfcc = librosa.feature.mfcc(y=target, sr=self.sr)
        
        similarities['mfcc'] = float(
            1 - cdist(source_mfcc.T, target_mfcc.T, metric='euclidean').mean()
        )
        
        # Compute spectral features
        source_cent = librosa.feature.spectral_centroid(y=source, sr=self.sr)[0]
        target_cent = librosa.feature.spectral_centroid(y=target, sr=self.sr)[0]
        
        similarities['centroid'] = float(
            1 - abs(np.mean(source_cent) - np.mean(target_cent)) / (
                max(np.mean(source_cent), np.mean(target_cent)) + 1e-8
            )
        )
        
        # Compute spectral contrast
        source_contrast = librosa.feature.spectral_contrast(y=source, sr=self.sr)
        target_contrast = librosa.feature.spectral_contrast(y=target, sr=self.sr)
        
        similarities['contrast'] = float(
            1 - cdist(source_contrast.T, target_contrast.T, metric='euclidean').mean()
        )
        
        return similarities

    def _structural_similarity(self, source: np.ndarray,
                             target: np.ndarray) -> Dict[str, float]:
        """Compute structural similarity using multiple features."""
        similarities = {}
        
        # Compute chromagrams
        source_chroma = librosa.feature.chroma_stft(y=source, sr=self.sr)
        target_chroma = librosa.feature.chroma_stft(y=target, sr=self.sr)
        
        similarities['chroma'] = float(
            1 - cdist(source_chroma.T, target_chroma.T, metric='correlation').mean()
        )
        
        # Compute tonnetz (harmonic network) features
        source_tonnetz = librosa.feature.tonnetz(y=source, sr=self.sr)
        target_tonnetz = librosa.feature.tonnetz(y=target, sr=self.sr)
        
        similarities['tonnetz'] = float(
            1 - cdist(source_tonnetz.T, target_tonnetz.T, metric='euclidean').mean()
        )
        
        return similarities

    def _groove_similarity(self, source: np.ndarray,
                         target: np.ndarray) -> Dict[str, float]:
        """Compute groove similarity focusing on rhythm patterns."""
        similarities = {}
        
        # Extract beat frames
        source_tempo, source_beats = librosa.beat.beat_track(y=source, sr=self.sr)
        target_tempo, target_beats = librosa.beat.beat_track(y=target, sr=self.sr)
        
        # Extract groove patterns
        source_pattern = self._extract_groove_pattern(source, source_beats)
        target_pattern = self._extract_groove_pattern(target, target_beats)
        
        # Compare patterns
        similarities['pattern'] = float(
            self._pattern_similarity(source_pattern, target_pattern)
        )
        
        # Compare microtiming
        source_micro = self._analyze_microtiming(source, source_beats)
        target_micro = self._analyze_microtiming(target, target_beats)
        
        similarities['microtiming'] = float(
            1 - abs(np.mean(source_micro) - np.mean(target_micro))
        )
        
        return similarities

    def _compute_multiband_onsets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute onset strength envelopes for different frequency bands."""
        onsets = {}
        
        for band_name, (low, high) in self.bands.items():
            try:
                # Apply bandpass filter using scipy
                nyquist = self.sr // 2
                b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
                filtered_audio = filtfilt(b, a, audio)
                
                # Trim the filtered audio
                y_band = librosa.effects.trim(filtered_audio)[0]
                
                # Compute onset strength
                onset_env = librosa.onset.onset_strength(
                    y=y_band,
                    sr=self.sr,
                    hop_length=self.hop_length
                )
                
                onsets[band_name] = librosa.util.normalize(onset_env)
            except Exception as e:
                print(f"Error in band {band_name}: {str(e)}")
                onsets[band_name] = np.zeros(128)  # Fallback pattern
        
        return onsets

    def _extract_groove_pattern(self, audio: np.ndarray,
                              beats: np.ndarray) -> np.ndarray:
        """Extract groove pattern from audio."""
        # Convert beats to frames
        beat_frames = librosa.frames_to_time(beats, sr=self.sr)
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Extract pattern
        pattern = librosa.util.normalize(onset_env[beats])
        
        return pattern

    def _analyze_microtiming(self, audio: np.ndarray,
                           beats: np.ndarray) -> np.ndarray:
        """Analyze microtiming deviations in groove."""
        # Convert beats to time
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Find actual onset times
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        # Calculate timing deviations
        deviations = []
        for beat in beat_times:
            # Find closest onset
            closest_onset = onset_times[
                np.argmin(np.abs(onset_times - beat))
            ]
            deviations.append(closest_onset - beat)
        
        return np.array(deviations)

    def _detect_transformations(self, source: np.ndarray,
                              target: np.ndarray) -> List[str]:
        """Detect transformations between source and target."""
        transformations = []
        
        # Detect time stretching
        source_tempo = librosa.beat.tempo(y=source, sr=self.sr)[0]
        target_tempo = librosa.beat.tempo(y=target, sr=self.sr)[0]
        
        if abs(source_tempo - target_tempo) / source_tempo > self.thresholds['time_stretch']:
            transformations.append('time_stretch')
        
        # Detect pitch shifting
        source_pitch = librosa.feature.spectral_centroid(y=source, sr=self.sr).mean()
        target_pitch = librosa.feature.spectral_centroid(y=target, sr=self.sr).mean()
        
        if abs(source_pitch - target_pitch) / source_pitch > self.thresholds['pitch_shift']:
            transformations.append('pitch_shift')
        
        # Detect spectral transformations
        source_flux = librosa.onset.onset_strength(y=source, sr=self.sr).mean()
        target_flux = librosa.onset.onset_strength(y=target, sr=self.sr).mean()
        
        if abs(source_flux - target_flux) / source_flux > self.thresholds['spectral_flux']:
            transformations.append('spectral_transform')
        
        return transformations

    def _weighted_combination(self, similarities: Dict[str, Dict[str, float]]) -> float:
        """Combine component similarities using weights."""
        # Calculate weighted average for each component
        component_scores = {}
        
        for component, scores in similarities.items():
            if isinstance(scores, dict):
                # Average sub-scores if component has multiple aspects
                component_scores[component] = np.mean(list(scores.values()))
            else:
                component_scores[component] = scores
        
        # Apply weights and combine
        weighted_sum = sum(
            self.weights[component] * score
            for component, score in component_scores.items()
        )
        
        return float(weighted_sum)

    def _calculate_confidence(self, similarities: Dict[str, Dict[str, float]],
                            transformations: List[str]) -> float:
        """Calculate confidence in similarity assessment."""
        # Base confidence on consistency of component scores
        component_scores = []
        for scores in similarities.values():
            if isinstance(scores, dict):
                component_scores.extend(scores.values())
            else:
                component_scores.append(scores)
        
        score_std = np.std(component_scores)
        consistency = 1.0 - min(1.0, score_std * 2)
        
        # Adjust based on number of detected transformations
        transform_factor = 1.0 - 0.1 * len(transformations)
        
        return float(consistency * transform_factor)

    def _create_result(self, similarity: float,
                      component_scores: Dict[str, Dict[str, float]],
                      transformations: List[str],
                      confidence: float) -> Dict:
        """Create final analysis result."""
        return {
            'similarity': float(similarity),
            'component_scores': {
                k: {
                    sk: float(sv) for sk, sv in v.items()
                } if isinstance(v, dict) else float(v)
                for k, v in component_scores.items()
            },
            'transformations': transformations,
            'confidence': float(confidence),
            'details': {
                'weight_contribution': {
                    component: float(
                        self.weights[component] * (
                            np.mean(list(scores.values()))
                            if isinstance(scores, dict)
                            else scores
                        )
                    )
                    for component, scores in component_scores.items()
                }
            }
        }

    @staticmethod
    def _pattern_similarity(pattern1, pattern2):
        return 1 - cdist(pattern1.reshape(1, -1), pattern2.reshape(1, -1), metric='correlation')[0, 0]

