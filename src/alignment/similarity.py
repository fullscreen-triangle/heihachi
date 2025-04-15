import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import librosa



logger = logging.getLogger("mix_analyzer")


@dataclass
class SimilarityResult:
    segment_pairs: List[Tuple[int, int]]
    similarity_score: float
    feature_correlations: Dict[str, float]


class SimilarityAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.config = {
            'processing': {
                'n_workers': 4,
                'use_gpu': True
            },
            'similarity': {
                'clustering_eps': 0.3,
                'min_samples': 2
            }
        }

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze similarities in the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing similarity matrix and segment groups
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Extract features
        features_list = self._extract_features(audio)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(features_list)

        # Find similar segments
        similar_groups = self.find_similar_segments(similarity_matrix)

        # Compute detailed statistics
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'similar_groups': similar_groups,
            'statistics': {
                'avg_similarity': float(np.mean(similarity_matrix)),
                'max_similarity': float(np.max(similarity_matrix)),
                'min_similarity': float(np.min(similarity_matrix[np.nonzero(similarity_matrix)])),
                'num_groups': len(similar_groups),
                'group_sizes': [len(group) for group in similar_groups]
            }
        }

    def _extract_features(self, audio: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Extract features from audio segments."""
        # Split audio into segments
        segment_length = self.sr * 5  # 5-second segments
        hop_length = self.sr * 2  # 2-second hop
        
        features_list = []
        for i in range(0, len(audio) - segment_length, hop_length):
            segment = audio[i:i + segment_length]
            features = {
                'mfcc': librosa.feature.mfcc(y=segment, sr=self.sr),
                'chroma': librosa.feature.chroma_stft(y=segment, sr=self.sr),
                'tempo': librosa.beat.tempo(y=segment, sr=self.sr)[0],
                'spectral_centroid': librosa.feature.spectral_centroid(y=segment, sr=self.sr)
            }
            features_list.append(features)
        
        return features_list

    def compute_similarity_matrix(self, features_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Compute similarity matrix using parallel processing."""
        n_segments = len(features_list)
        similarity_matrix = np.zeros((n_segments, n_segments))

        # Prepare feature vectors for efficient computation
        feature_vectors = self._prepare_feature_vectors(features_list)

        # Compute similarities in parallel
        with ThreadPoolExecutor(max_workers=self.config['processing']['n_workers']) as executor:
            futures = []
            for i in range(n_segments):
                for j in range(i + 1, n_segments):
                    futures.append(
                        executor.submit(
                            self._compute_similarity_pair,
                            feature_vectors[i],
                            feature_vectors[j]
                        )
                    )

            # Collect results
            idx = 0
            for i in range(n_segments):
                for j in range(i + 1, n_segments):
                    similarity_matrix[i, j] = futures[idx].result()
                    similarity_matrix[j, i] = similarity_matrix[i, j]
                    idx += 1

        return similarity_matrix

    def _prepare_feature_vectors(self, features_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Convert feature dictionaries to normalized vectors."""
        feature_vectors = []
        for features in features_list:
            vector = np.concatenate([
                features['mfcc'].flatten(),
                features['chroma'].flatten(),
                [features['tempo']],
                features['spectral_centroid'].flatten()
            ])
            feature_vectors.append(vector)

        # Normalize features
        scaler = StandardScaler()
        return scaler.fit_transform(feature_vectors)

    def _compute_similarity_pair(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two feature vectors."""
        # Convert to torch tensors for GPU acceleration if available
        if self.device.type == 'cuda':
            vec1 = torch.from_numpy(vec1).to(self.device)
            vec2 = torch.from_numpy(vec2).to(self.device)
            similarity = torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
            return similarity.cpu().numpy()[0]
        else:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_similar_segments(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Find groups of similar segments using DBSCAN clustering."""
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.config['similarity']['clustering_eps'],
            min_samples=self.config['similarity']['min_samples'],
            metric='precomputed'
        )

        # Convert similarity matrix to distance matrix
        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)

        # Group similar segments
        unique_labels = set(labels)
        similar_groups = []
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                group = np.where(labels == label)[0].tolist()
                if len(group) >= 2:  # Only include groups with at least 2 segments
                    similar_groups.append(group)

        return similar_groups
