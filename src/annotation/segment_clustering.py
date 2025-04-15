from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class Segment:
    start_time: float
    end_time: float
    confidence: float
    features: Dict
    cluster_label: Optional[int] = None
    transition_in: Optional[Dict] = None
    transition_out: Optional[Dict] = None


class SegmentClusterer:
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.min_segment_len = 30.0  # Minimum segment length in seconds
        self.max_segment_len = 480.0  # Maximum segment length in seconds (8 minutes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze(self, audio: np.ndarray, transitions: List[Dict]) -> Dict:
        """Analyze and cluster segments in a DJ mix.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            transitions (List[Dict]): List of detected transitions
            
        Returns:
            Dict: Analysis results containing segments and their relationships
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Create initial segments based on transitions
        segments = self._create_segments(audio, transitions)
        
        if not segments:
            return self._get_fallback_results()

        # Extract features for each segment
        features = self._extract_segment_features(segments, audio)
        
        # Cluster segments
        labels, confidences = self._cluster_segments(features)
        
        # Assign cluster labels to segments
        for segment, label, confidence in zip(segments, labels, confidences):
            segment.cluster_label = int(label)
            segment.confidence = float(confidence)

        # Analyze relationships between segments
        relationships = self._analyze_segment_relationships(segments, features)
        
        return {
            'segments': [self._segment_to_dict(s) for s in segments],
            'relationships': relationships,
            'num_clusters': len(set(labels)),
            'average_confidence': float(np.mean(confidences))
        }

    def _create_segments(self, audio: np.ndarray, 
                        transitions: List[Dict]) -> List[Segment]:
        """Create initial segments based on detected transitions."""
        if not transitions:
            return []

        segments = []
        audio_duration = len(audio) / self.sr

        # Sort transitions by start time
        sorted_trans = sorted(transitions, key=lambda x: x['start_time'])
        
        # Create segments between transitions
        current_time = 0.0
        for i, trans in enumerate(sorted_trans):
            # Skip if transition starts before current time
            if trans['start_time'] < current_time:
                continue
                
            # Create segment if long enough
            if trans['start_time'] - current_time >= self.min_segment_len:
                segments.append(Segment(
                    start_time=current_time,
                    end_time=trans['start_time'],
                    confidence=1.0,  # Will be updated later
                    features={},
                    transition_in=sorted_trans[i-1] if i > 0 else None,
                    transition_out=trans
                ))
            
            current_time = trans['end_time']

        # Add final segment if needed
        if audio_duration - current_time >= self.min_segment_len:
            segments.append(Segment(
                start_time=current_time,
                end_time=audio_duration,
                confidence=1.0,
                features={},
                transition_in=sorted_trans[-1] if sorted_trans else None,
                transition_out=None
            ))

        return segments

    def _extract_segment_features(self, segments: List[Segment], 
                                audio: np.ndarray) -> np.ndarray:
        """Extract features for each segment."""
        features_list = []
        
        for segment in segments:
            # Convert times to samples
            start_sample = int(segment.start_time * self.sr)
            end_sample = int(segment.end_time * self.sr)
            
            # Extract audio segment
            segment_audio = audio[start_sample:end_sample]
            
            # Compute features
            segment_features = self._compute_segment_features(segment_audio)
            
            # Store features in segment object
            segment.features = segment_features
            
            # Prepare feature vector for clustering
            feature_vector = self._prepare_feature_vector(segment_features)
            features_list.append(feature_vector)
        
        return np.array(features_list)

    def _compute_segment_features(self, audio: np.ndarray) -> Dict:
        """Compute various features for a segment."""
        features = {}
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=self.hop_length)
        mag_spec = np.abs(stft)
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(S=mag_spec)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(S=mag_spec)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(S=mag_spec)[0]
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features['tempo'] = tempo
        
        # Mel features
        mel_spec = librosa.feature.melspectrogram(S=mag_spec)
        features['mfcc'] = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(S=mag_spec)
        
        return features

    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for clustering."""
        feature_vector = []
        
        # Add tempo
        feature_vector.append(features['tempo'])
        
        # Add statistics of other features
        for feature_name in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
            feature = features[feature_name]
            feature_vector.extend([
                np.mean(feature),
                np.std(feature),
                np.median(feature)
            ])
        
        # Add MFCC statistics
        mfcc_means = np.mean(features['mfcc'], axis=1)
        mfcc_stds = np.std(features['mfcc'], axis=1)
        feature_vector.extend(mfcc_means[:8])  # Use first 8 MFCCs
        feature_vector.extend(mfcc_stds[:8])
        
        # Add chroma statistics
        chroma_means = np.mean(features['chroma'], axis=1)
        feature_vector.extend(chroma_means)
        
        return np.array(feature_vector)

    def _cluster_segments(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster segments using multiple clustering methods."""
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Try DBSCAN first for automatic cluster detection
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(normalized_features)
        
        # If DBSCAN fails to find meaningful clusters, use hierarchical clustering
        if len(set(dbscan_labels)) < 2 or -1 in dbscan_labels:
            n_clusters = max(2, len(features) // 3)  # Estimate number of clusters
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='ward'
            )
            labels = clustering.fit_predict(normalized_features)
        else:
            labels = dbscan_labels
        
        # Calculate confidence scores
        confidences = self._calculate_cluster_confidence(normalized_features, labels)
        
        return labels, confidences

    def _calculate_cluster_confidence(self, features: np.ndarray, 
                                   labels: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for cluster assignments."""
        confidences = np.zeros(len(labels))
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get points in cluster
            mask = labels == label
            cluster_points = features[mask]
            
            if len(cluster_points) > 0:
                # Calculate cluster center
                center = np.mean(cluster_points, axis=0)
                
                # Calculate distances to center
                distances = np.linalg.norm(cluster_points - center, axis=1)
                
                # Calculate confidence based on distance to center
                cluster_confidences = 1.0 / (1.0 + distances)
                
                # Assign confidences
                confidences[mask] = cluster_confidences
        
        return confidences

    def _analyze_segment_relationships(self, segments: List[Segment], 
                                    features: np.ndarray) -> Dict:
        """Analyze relationships between segments."""
        relationships = {
            'transitions': [],
            'similar_segments': [],
            'cluster_stats': {}
        }
        
        # Analyze transitions between segments
        for i in range(len(segments) - 1):
            if segments[i].transition_out and segments[i+1].transition_in:
                relationships['transitions'].append({
                    'from_segment': i,
                    'to_segment': i + 1,
                    'transition_type': segments[i].transition_out['type'],
                    'components': segments[i].transition_out['components']
                })
        
        # Find similar segments
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if segments[i].cluster_label == segments[j].cluster_label:
                    similarity = 1.0 - np.linalg.norm(
                        features[i] - features[j]
                    ) / np.sqrt(features.shape[1])
                    
                    if similarity > 0.8:  # High similarity threshold
                        relationships['similar_segments'].append({
                            'segment1': i,
                            'segment2': j,
                            'similarity': float(similarity)
                        })
        
        # Calculate cluster statistics
        for label in set(s.cluster_label for s in segments):
            cluster_segments = [s for s in segments if s.cluster_label == label]
            relationships['cluster_stats'][str(label)] = {
                'count': len(cluster_segments),
                'avg_duration': np.mean([
                    s.end_time - s.start_time for s in cluster_segments
                ]),
                'avg_confidence': np.mean([s.confidence for s in cluster_segments])
            }
        
        return relationships

    def _segment_to_dict(self, segment: Segment) -> Dict:
        """Convert Segment object to dictionary."""
        return {
            'start_time': float(segment.start_time),
            'end_time': float(segment.end_time),
            'confidence': float(segment.confidence),
            'cluster_label': int(segment.cluster_label) if segment.cluster_label is not None else None,
            'features': {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                for k, v in segment.features.items()
            },
            'transition_in': segment.transition_in,
            'transition_out': segment.transition_out
        }

    def _get_fallback_results(self) -> Dict:
        """Return default results when analysis fails."""
        return {
            'segments': [],
            'relationships': {
                'transitions': [],
                'similar_segments': [],
                'cluster_stats': {}
            },
            'num_clusters': 0,
            'average_confidence': 0.0
        }
