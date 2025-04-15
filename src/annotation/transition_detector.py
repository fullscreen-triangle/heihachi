from dataclasses import dataclass
import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import List, Dict, Optional, Tuple
import torch


@dataclass
class Transition:
    start_time: float
    end_time: float
    confidence: float
    type: str  # 'cut', 'fade', 'blend', 'swap', 'filter'
    features: Dict
    components: List[str]  # Which components were involved ('bass', 'drums', 'melody', etc.)


class TransitionDetector:
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.min_transition_len = 4.0  # Minimum transition length in seconds
        self.max_transition_len = 32.0  # Maximum transition length in seconds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency bands for component analysis
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'very_high': (8000, 16000)
        }

    def detect(self, audio: np.ndarray) -> Dict:
        """Detect transitions in a DJ mix.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing detected transitions and their characteristics
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Compute multi-band spectrograms
        specs = self._compute_multiband_specs(audio)
        
        # Detect novelty curves for each band
        novelty_curves = self._compute_novelty_curves(specs)
        
        # Find potential transition points
        transitions = self._find_transition_candidates(novelty_curves)
        
        # Analyze and classify transitions
        classified = self._classify_transitions(transitions, specs, audio)
        
        # Post-process to merge overlapping transitions
        final_transitions = self._post_process_transitions(classified)
        
        return {
            'transitions': [self._transition_to_dict(t) for t in final_transitions],
            'count': len(final_transitions),
            'average_length': np.mean([t.end_time - t.start_time for t in final_transitions]),
            'types': {t.type: sum(1 for x in final_transitions if x.type == t.type) 
                     for t in final_transitions}
        }

    def _compute_multiband_specs(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute spectrograms for different frequency bands."""
        specs = {}
        for band_name, (low, high) in self.bands.items():
            # Apply bandpass filter
            y_band = librosa.effects.trim(
                librosa.filtfilt(
                    audio,
                    self.sr,
                    low,
                    high
                )
            )[0]
            
            # Compute spectrogram
            spec = librosa.stft(y_band, n_fft=2048, hop_length=self.hop_length)
            specs[band_name] = np.abs(spec)
        
        return specs

    def _compute_novelty_curves(self, specs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute novelty curves for each frequency band."""
        novelty_curves = {}
        for band_name, spec in specs.items():
            # Compute novelty curve using spectral flux
            diff = np.diff(spec, axis=1)
            pos_diff = np.maximum(0, diff)  # Keep only increases in energy
            novelty = np.sum(pos_diff, axis=0)
            novelty_curves[band_name] = librosa.util.normalize(novelty)
        
        return novelty_curves

    def _find_transition_candidates(self, novelty_curves: Dict[str, np.ndarray]) -> List[Dict]:
        """Find potential transition points using multi-band analysis."""
        candidates = []
        min_frames = int(self.min_transition_len * self.sr / self.hop_length)
        max_frames = int(self.max_transition_len * self.sr / self.hop_length)
        
        # Combine novelty curves with weights
        weights = {
            'sub_bass': 0.15,
            'bass': 0.2,
            'low_mid': 0.15,
            'mid': 0.15,
            'high_mid': 0.15,
            'high': 0.1,
            'very_high': 0.1
        }
        
        combined = np.zeros_like(list(novelty_curves.values())[0])
        for band, curve in novelty_curves.items():
            combined += weights[band] * curve
        
        # Find peaks in combined novelty
        peaks = find_peaks(
            combined,
            distance=min_frames,
            prominence=0.1,
            width=(min_frames, max_frames)
        )
        
        for peak, prominence, left_base, right_base in zip(
            peaks[0], peaks[1]['prominences'], 
            peaks[1]['left_bases'], peaks[1]['right_bases']
        ):
            candidates.append({
                'peak_frame': peak,
                'start_frame': left_base,
                'end_frame': right_base,
                'prominence': prominence,
                'band_activities': {
                    band: np.mean(curve[left_base:right_base])
                    for band, curve in novelty_curves.items()
                }
            })
        
        return candidates

    def _classify_transitions(self, candidates: List[Dict], 
                            specs: Dict[str, np.ndarray],
                            audio: np.ndarray) -> List[Transition]:
        """Classify transitions based on their characteristics."""
        transitions = []
        
        for candidate in candidates:
            # Analyze spectral changes during transition
            spectral_changes = self._analyze_spectral_changes(
                specs, 
                candidate['start_frame'],
                candidate['end_frame']
            )
            
            # Determine transition type and components
            t_type, components = self._determine_transition_type(
                spectral_changes,
                candidate['band_activities']
            )
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                candidate['prominence'],
                spectral_changes,
                candidate['band_activities']
            )
            
            transitions.append(Transition(
                start_time=candidate['start_frame'] * self.hop_length / self.sr,
                end_time=candidate['end_frame'] * self.hop_length / self.sr,
                confidence=confidence,
                type=t_type,
                features=spectral_changes,
                components=components
            ))
        
        return transitions

    def _analyze_spectral_changes(self, specs: Dict[str, np.ndarray], 
                                start_frame: int, end_frame: int) -> Dict:
        """Analyze how different spectral components change during transition."""
        changes = {}
        
        for band_name, spec in specs.items():
            band_spec = spec[:, start_frame:end_frame]
            
            # Calculate various spectral features
            changes[band_name] = {
                'energy_change': np.mean(np.diff(np.sum(band_spec, axis=0))),
                'spectral_flux': np.mean(np.diff(band_spec, axis=1)),
                'max_energy': np.max(np.sum(band_spec, axis=0)),
                'min_energy': np.min(np.sum(band_spec, axis=0))
            }
        
        return changes

    def _determine_transition_type(self, spectral_changes: Dict,
                                 band_activities: Dict) -> Tuple[str, List[str]]:
        """Determine transition type and involved components."""
        # Initialize component tracking
        active_components = []
        
        # Check for significant changes in each frequency band
        for band, changes in spectral_changes.items():
            if abs(changes['energy_change']) > 0.1:
                active_components.append(band)
        
        # Determine transition type based on spectral characteristics
        if len(active_components) <= 2 and 'bass' in active_components:
            return 'swap', active_components  # Likely a bass swap
        elif all(changes['spectral_flux'] < 0.05 for changes in spectral_changes.values()):
            return 'fade', active_components  # Smooth fade
        elif any(changes['spectral_flux'] > 0.2 for changes in spectral_changes.values()):
            return 'cut', active_components  # Sharp transition
        elif any('high' in band for band in active_components):
            return 'filter', active_components  # Filter-based transition
        else:
            return 'blend', active_components  # General crossfade/blend

    def _calculate_confidence(self, prominence: float, 
                            spectral_changes: Dict,
                            band_activities: Dict) -> float:
        """Calculate confidence score for transition detection."""
        # Base confidence from novelty prominence
        confidence = min(1.0, prominence * 2)
        
        # Adjust based on spectral stability
        spectral_stability = np.mean([
            abs(changes['energy_change']) 
            for changes in spectral_changes.values()
        ])
        confidence *= (1.0 - min(1.0, spectral_stability))
        
        # Adjust based on band activity consistency
        activity_std = np.std(list(band_activities.values()))
        confidence *= (1.0 - min(1.0, activity_std * 2))
        
        return float(confidence)

    def _post_process_transitions(self, transitions: List[Transition]) -> List[Transition]:
        """Merge overlapping transitions and remove duplicates."""
        if not transitions:
            return []
        
        # Sort by start time
        transitions.sort(key=lambda x: x.start_time)
        
        # Merge overlapping transitions
        merged = []
        current = transitions[0]
        
        for next_trans in transitions[1:]:
            if next_trans.start_time <= current.end_time:
                # Merge transitions
                current = Transition(
                    start_time=min(current.start_time, next_trans.start_time),
                    end_time=max(current.end_time, next_trans.end_time),
                    confidence=max(current.confidence, next_trans.confidence),
                    type='blend' if current.type != next_trans.type else current.type,
                    features={**current.features, **next_trans.features},
                    components=list(set(current.components + next_trans.components))
                )
            else:
                merged.append(current)
                current = next_trans
        
        merged.append(current)
        return merged

    def _transition_to_dict(self, transition: Transition) -> Dict:
        """Convert Transition object to dictionary."""
        return {
            'start_time': float(transition.start_time),
            'end_time': float(transition.end_time),
            'confidence': float(transition.confidence),
            'type': transition.type,
            'components': transition.components,
            'features': {
                k: float(v) if isinstance(v, (np.floating, float))
                else v for k, v in transition.features.items()
            }
        }
