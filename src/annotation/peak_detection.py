import numpy as np
import librosa
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass


@dataclass
class Peak:
    time: float
    amplitude: float
    confidence: float
    type: str  # 'onset', 'beat', 'downbeat', 'phrase'
    features: Dict


class PeakDetector:
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Analysis parameters
        self.min_peak_distance = 0.1  # Minimum time between peaks (seconds)
        self.prominence_threshold = 0.1
        self.adaptive_window = 8192  # Window size for adaptive thresholding
        
        # Frequency bands for multi-band analysis
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000)
        }

    def detect(self, audio: np.ndarray) -> Dict:
        """Detect peaks in audio using multiple methods.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing detected peaks and their characteristics
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Multi-band analysis
        band_onsets = self._detect_multiband_onsets(audio)
        
        # Detect different types of peaks
        peaks = {
            'onsets': self._detect_onsets(audio, band_onsets),
            'beats': self._detect_beats(audio, band_onsets),
            'downbeats': self._detect_downbeats(audio, band_onsets),
            'phrases': self._detect_phrases(audio, band_onsets)
        }
        
        # Combine and post-process peaks
        combined = self._combine_peaks(peaks)
        
        # Calculate statistics
        stats = self._calculate_statistics(combined)
        
        return {
            'peaks': [self._peak_to_dict(p) for p in combined],
            'statistics': stats,
            'band_activity': self._analyze_band_activity(band_onsets)
        }

    def _detect_multiband_onsets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect onsets in different frequency bands."""
        band_onsets = {}
        
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
                    hop_length=self.hop_length,
                    aggregate=np.median
                )
                
                # Apply adaptive thresholding
                band_onsets[band_name] = self._adaptive_threshold(onset_env)
                
            except Exception as e:
                print(f"Error in band {band_name}: {str(e)}")
                band_onsets[band_name] = np.zeros_like(
                    librosa.frames_to_time(
                        np.arange(len(audio) // self.hop_length),
                        sr=self.sr,
                        hop_length=self.hop_length
                    )
                )
        
        return band_onsets

    def _adaptive_threshold(self, onset_env: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to onset envelope."""
        # Smooth the onset envelope
        smoothed = savgol_filter(onset_env, 7, 3)
        
        # Calculate local mean and std
        local_mean = np.convolve(
            smoothed,
            np.ones(self.adaptive_window) / self.adaptive_window,
            mode='same'
        )
        local_std = np.sqrt(
            np.convolve(
                (smoothed - local_mean) ** 2,
                np.ones(self.adaptive_window) / self.adaptive_window,
                mode='same'
            )
        )
        
        # Apply adaptive threshold
        threshold = local_mean + 2 * local_std
        return np.maximum(0, smoothed - threshold)

    def _detect_onsets(self, audio: np.ndarray, 
                      band_onsets: Dict[str, np.ndarray]) -> List[Peak]:
        """Detect onsets using combined band information."""
        # Combine band onsets with weights
        weights = {
            'sub_bass': 0.15,
            'bass': 0.2,
            'low_mid': 0.15,
            'mid': 0.15,
            'high_mid': 0.2,
            'high': 0.15
        }
        
        combined = np.zeros_like(list(band_onsets.values())[0])
        for band, env in band_onsets.items():
            combined += weights[band] * librosa.util.normalize(env)
        
        # Find peaks in combined envelope
        peaks, properties = find_peaks(
            combined,
            distance=int(self.min_peak_distance * self.sr / self.hop_length),
            prominence=self.prominence_threshold
        )
        
        # Convert to Peak objects
        onset_peaks = []
        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        
        for time, peak, prom in zip(times, peaks, properties['prominences']):
            # Calculate confidence based on prominence
            confidence = min(1.0, prom / np.max(properties['prominences']))
            
            # Get peak features
            features = self._extract_peak_features(audio, peak, band_onsets)
            
            onset_peaks.append(Peak(
                time=float(time),
                amplitude=float(combined[peak]),
                confidence=float(confidence),
                type='onset',
                features=features
            ))
        
        return onset_peaks

    def _detect_beats(self, audio: np.ndarray, 
                     band_onsets: Dict[str, np.ndarray]) -> List[Peak]:
        """Detect beats using rhythm information."""
        # Focus on low frequency bands for beat detection
        low_bands = ['sub_bass', 'bass', 'low_mid']
        combined = np.zeros_like(list(band_onsets.values())[0])
        
        for band in low_bands:
            if band in band_onsets:
                combined += librosa.util.normalize(band_onsets[band])
        
        # Detect tempo and beats
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=combined,
            sr=self.sr,
            hop_length=self.hop_length,
            tightness=100
        )
        
        # Convert to Peak objects
        beat_peaks = []
        times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
        
        for time, beat in zip(times, beats):
            # Calculate confidence based on local onset strength
            local_env = combined[max(0, beat-1):min(len(combined), beat+2)]
            confidence = np.mean(local_env) if len(local_env) > 0 else 0.0
            
            features = self._extract_peak_features(audio, beat, band_onsets)
            features['tempo'] = tempo
            
            beat_peaks.append(Peak(
                time=float(time),
                amplitude=float(combined[beat]) if beat < len(combined) else 0.0,
                confidence=float(confidence),
                type='beat',
                features=features
            ))
        
        return beat_peaks

    def _detect_downbeats(self, audio: np.ndarray, 
                         band_onsets: Dict[str, np.ndarray]) -> List[Peak]:
        """Detect downbeats using rhythm and energy information."""
        # Get beats first
        beats = self._detect_beats(audio, band_onsets)
        if not beats:
            return []
        
        # Analyze beat strengths
        strengths = [b.amplitude for b in beats]
        
        # Find peaks in beat strengths (potential downbeats)
        peak_indices = find_peaks(
            strengths,
            distance=4,  # Assume 4/4 time signature
            prominence=0.1
        )[0]
        
        downbeats = []
        for idx in peak_indices:
            if idx < len(beats):
                beat = beats[idx]
                downbeats.append(Peak(
                    time=beat.time,
                    amplitude=beat.amplitude,
                    confidence=beat.confidence * 0.8,  # Slightly lower confidence for downbeats
                    type='downbeat',
                    features={
                        **beat.features,
                        'beat_position': idx
                    }
                ))
        
        return downbeats

    def _detect_phrases(self, audio: np.ndarray, 
                       band_onsets: Dict[str, np.ndarray]) -> List[Peak]:
        """Detect phrase boundaries using long-term structure."""
        # Combine all band onsets
        combined = np.sum([
            librosa.util.normalize(env) for env in band_onsets.values()
        ], axis=0)
        
        # Compute novelty curve with large window
        novelty = librosa.onset.onset_strength(
            onset_envelope=combined,
            sr=self.sr,
            hop_length=self.hop_length * 4,
            aggregate=np.median
        )
        
        # Find peaks in novelty curve
        peaks = find_peaks(
            novelty,
            distance=int(8 * self.sr / self.hop_length),  # Minimum 8 seconds between phrases
            prominence=0.2
        )[0]
        
        # Convert to Peak objects
        phrase_peaks = []
        times = librosa.frames_to_time(peaks * 4, sr=self.sr, hop_length=self.hop_length)
        
        for time, peak in zip(times, peaks):
            features = self._extract_peak_features(
                audio,
                peak * 4,  # Adjust for hop_length multiplier
                band_onsets
            )
            
            phrase_peaks.append(Peak(
                time=float(time),
                amplitude=float(novelty[peak]),
                confidence=0.7,  # Lower confidence for phrase detection
                type='phrase',
                features=features
            ))
        
        return phrase_peaks

    def _extract_peak_features(self, audio: np.ndarray, frame: int,
                             band_onsets: Dict[str, np.ndarray]) -> Dict:
        """Extract features for a detected peak."""
        features = {
            'band_energies': {},
            'local_context': {}
        }
        
        # Get band-wise energy at peak
        for band, env in band_onsets.items():
            if frame < len(env):
                features['band_energies'][band] = float(env[frame])
            else:
                features['band_energies'][band] = 0.0
        
        # Analyze local context (±100ms)
        context_frames = int(0.1 * self.sr / self.hop_length)
        start = max(0, frame - context_frames)
        end = min(len(audio) // self.hop_length, frame + context_frames)
        
        for band, env in band_onsets.items():
            if start < len(env):
                context = env[start:end]
                features['local_context'][band] = {
                    'mean': float(np.mean(context)),
                    'std': float(np.std(context)),
                    'max': float(np.max(context))
                }
        
        return features

    def _combine_peaks(self, peaks: Dict[str, List[Peak]]) -> List[Peak]:
        """Combine peaks from different detection methods."""
        all_peaks = []
        for peak_type, peak_list in peaks.items():
            all_peaks.extend(peak_list)
        
        # Sort by time
        all_peaks.sort(key=lambda x: x.time)
        
        # Merge close peaks
        merged = []
        if not all_peaks:
            return merged
        
        current = all_peaks[0]
        for next_peak in all_peaks[1:]:
            if next_peak.time - current.time < self.min_peak_distance:
                # Keep the peak with higher confidence
                if next_peak.confidence > current.confidence:
                    current = next_peak
            else:
                merged.append(current)
                current = next_peak
        
        merged.append(current)
        return merged

    def _calculate_statistics(self, peaks: List[Peak]) -> Dict:
        """Calculate statistics about detected peaks."""
        if not peaks:
            return {
                'total_peaks': 0,
                'peak_density': 0.0,
                'average_confidence': 0.0,
                'type_distribution': {}
            }
        
        # Calculate basic statistics
        total_duration = peaks[-1].time - peaks[0].time
        stats = {
            'total_peaks': len(peaks),
            'peak_density': len(peaks) / total_duration if total_duration > 0 else 0.0,
            'average_confidence': np.mean([p.confidence for p in peaks]),
            'type_distribution': {}
        }
        
        # Count peak types
        for peak in peaks:
            if peak.type not in stats['type_distribution']:
                stats['type_distribution'][peak.type] = 0
            stats['type_distribution'][peak.type] += 1
        
        return stats

    def _analyze_band_activity(self, band_onsets: Dict[str, np.ndarray]) -> Dict:
        """Analyze activity in different frequency bands."""
        activity = {}
        
        for band, env in band_onsets.items():
            activity[band] = {
                'mean': float(np.mean(env)),
                'std': float(np.std(env)),
                'max': float(np.max(env)),
                'active_ratio': float(np.mean(env > 0.1))
            }
        
        return activity

    def _peak_to_dict(self, peak: Peak) -> Dict:
        """Convert Peak object to dictionary."""
        return {
            'time': float(peak.time),
            'amplitude': float(peak.amplitude),
            'confidence': float(peak.confidence),
            'type': peak.type,
            'features': {
                k: v if isinstance(v, dict) else float(v)
                for k, v in peak.features.items()
            }
        }

