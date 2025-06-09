import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
from openai import OpenAI
from semantic.feature_mapping import EmotionalFeatureMapper

class EmbeddingGenerator:
    """Generates embeddings for tracks based on their analysis results."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            config: Configuration dictionary (optional)
        """
        # Try to get API key from different sources in order of precedence:
        # 1. Directly provided api_key parameter
        # 2. From config dictionary (semantic.openai_api_key)
        # 3. Environment variable OPENAI_API_KEY
        self.config = config or {}
        
        if api_key is not None:
            self.api_key = api_key
        elif self.config.get('semantic', {}).get('openai_api_key'):
            self.api_key = self.config.get('semantic', {}).get('openai_api_key')
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            
        # Debug output
        if self.api_key:
            print(f"Using OpenAI API key (first 5 chars): {self.api_key[:5]}...")
        else:
            print("No OpenAI API key found in parameters, config, or environment")
            
        # Initialize OpenAI client with API key
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.feature_mapper = EmotionalFeatureMapper()
            self.embedding_model = "text-embedding-ada-002"
            self.embedding_dim = 1536  # Dimension of the embedding vector
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def generate_track_embedding(self, track_info: Dict[str, Any], analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for a track based on its analysis.
        
        Args:
            track_info: Basic track information (title, artist, etc.)
            analysis_result: Complete analysis result from Triangle pipeline
            
        Returns:
            Embedding vector as numpy array
        """
        # Create rich text description
        description = self.create_track_description(track_info, analysis_result)
        
        try:
            # Generate embedding using OpenAI API
            response = self.client.embeddings.create(
                input=description,
                model=self.embedding_model
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def create_track_description(self, track_info: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """
        Create rich text description of track for embedding.
        
        Args:
            track_info: Basic track information
            analysis_result: Complete analysis result
            
        Returns:
            Text description
        """
        # Map technical features to emotional dimensions
        emotions = self.feature_mapper.map_features_to_emotions(analysis_result)
        
        # Extract key features from analysis
        bpm = self._extract_bpm(analysis_result)
        key = self._extract_key(analysis_result)
        
        # Create basic description
        description = f"""
        Track: {track_info.get('title', 'Unknown Title')}
        Artist: {track_info.get('artist', 'Unknown Artist')}
        BPM: {bpm}
        Key: {key}
        
        Emotional characteristics:
        - Energy: {emotions['energy']:.1f}/10 - {self.feature_mapper.describe_emotion('energy', emotions['energy'])}
        - Brightness: {emotions['brightness']:.1f}/10 - {self.feature_mapper.describe_emotion('brightness', emotions['brightness'])}
        - Tension: {emotions['tension']:.1f}/10 - {self.feature_mapper.describe_emotion('tension', emotions['tension'])}
        - Warmth: {emotions['warmth']:.1f}/10 - {self.feature_mapper.describe_emotion('warmth', emotions['warmth'])}
        - Groove: {emotions['groove']:.1f}/10 - {self.feature_mapper.describe_emotion('groove', emotions['groove'])}
        - Aggression: {emotions['aggression']:.1f}/10 - {self.feature_mapper.describe_emotion('aggression', emotions['aggression'])}
        - Atmosphere: {emotions['atmosphere']:.1f}/10 - {self.feature_mapper.describe_emotion('atmosphere', emotions['atmosphere'])}
        - Melancholy: {emotions['melancholy']:.1f}/10 - {self.feature_mapper.describe_emotion('melancholy', emotions['melancholy'])}
        - Euphoria: {emotions['euphoria']:.1f}/10 - {self.feature_mapper.describe_emotion('euphoria', emotions['euphoria'])}
        
        Technical characteristics:
        """
        
        # Add bass characteristics if available
        bass_info = self._extract_bass_info(analysis_result)
        if bass_info:
            description += f"\nBass characteristics: {bass_info}"
        
        # Add drum characteristics if available
        drum_info = self._extract_drum_info(analysis_result)
        if drum_info:
            description += f"\nDrum characteristics: {drum_info}"
        
        # Add spatial characteristics if available
        spatial_info = self._extract_spatial_info(analysis_result)
        if spatial_info:
            description += f"\nSpatial characteristics: {spatial_info}"
        
        # Add Amen break info if detected
        if self._has_amen_break(analysis_result):
            amen_info = self._extract_amen_info(analysis_result)
            description += f"\nAmen break patterns: {amen_info}"
        
        # Add transition info if available
        transition_info = self._extract_transition_info(analysis_result)
        if transition_info:
            description += f"\nTransition characteristics: {transition_info}"
        
        # Add summary
        description += f"""
        
        Overall, this track can be described as {self._generate_summary_description(emotions)}.
        It would be suitable for {self._generate_mood_situations(emotions)}.
        """
        
        return description
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a user query.
        
        Args:
            query: User's emotional query
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Generate embedding using OpenAI API
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def _extract_bpm(self, analysis_result: Dict[str, Any]) -> str:
        """Extract BPM information from analysis result."""
        try:
            bpm_data = analysis_result.get('features', {}).get('bpm', {})
            if bpm_data and 'tempo' in bpm_data:
                return f"{bpm_data['tempo']:.1f}"
            
            # Fallback to mix analyzer tempo
            mix_data = analysis_result.get('analysis', {}).get('mix', {})
            rhythmic_data = mix_data.get('rhythmic', [])
            if len(rhythmic_data) > 0:
                rhythmic_features = np.mean(rhythmic_data.cpu().numpy(), axis=0) if hasattr(rhythmic_data, 'cpu') else np.mean(rhythmic_data, axis=0)
                return f"{float(rhythmic_features[0]):.1f}"
            
            return "Unknown"
        except:
            return "Unknown"
    
    def _extract_key(self, analysis_result: Dict[str, Any]) -> str:
        """Extract key information from analysis result."""
        # This would ideally come from key detection
        # For now, return a placeholder
        return "Unknown"
    
    def _extract_bass_info(self, analysis_result: Dict[str, Any]) -> str:
        """Extract bass information from analysis result."""
        try:
            baseline = analysis_result.get('features', {}).get('baseline', {})
            if not baseline:
                return ""
            
            info_parts = []
            
            if 'presence' in baseline:
                presence = float(baseline['presence'])
                if presence > 7:
                    info_parts.append("prominent bass")
                elif presence > 4:
                    info_parts.append("moderate bass")
                else:
                    info_parts.append("subtle bass")
            
            if 'sub_bass_energy' in baseline:
                sub_energy = float(baseline['sub_bass_energy'])
                if sub_energy > 7:
                    info_parts.append("strong sub-bass")
                elif sub_energy > 4:
                    info_parts.append("moderate sub-bass")
            
            if 'distortion' in baseline:
                distortion = float(baseline['distortion'])
                if distortion > 7:
                    info_parts.append("heavily distorted")
                elif distortion > 4:
                    info_parts.append("slightly distorted")
            
            if 'movement' in baseline:
                movement = float(baseline['movement'])
                if movement > 7:
                    info_parts.append("dynamic movement")
                elif movement > 4:
                    info_parts.append("some movement")
            
            return ", ".join(info_parts) if info_parts else "standard bass characteristics"
            
        except:
            return ""
    
    def _extract_drum_info(self, analysis_result: Dict[str, Any]) -> str:
        """Extract drum information from analysis result."""
        try:
            drum = analysis_result.get('features', {}).get('drum', {})
            if not drum:
                return ""
            
            info_parts = []
            
            if 'complexity' in drum:
                complexity = float(drum['complexity'])
                if complexity > 7:
                    info_parts.append("complex drum patterns")
                elif complexity > 4:
                    info_parts.append("moderately complex drums")
                else:
                    info_parts.append("simple drum patterns")
            
            if 'intensity' in drum:
                intensity = float(drum['intensity'])
                if intensity > 7:
                    info_parts.append("intense drums")
                elif intensity > 4:
                    info_parts.append("moderately intense drums")
                else:
                    info_parts.append("gentle drums")
            
            if 'break_presence' in drum:
                break_presence = float(drum['break_presence'])
                if break_presence > 7:
                    info_parts.append("prominent drum breaks")
                elif break_presence > 4:
                    info_parts.append("some drum breaks")
            
            return ", ".join(info_parts) if info_parts else "standard drum characteristics"
            
        except:
            return ""
    
    def _extract_spatial_info(self, analysis_result: Dict[str, Any]) -> str:
        """Extract spatial information from analysis result."""
        try:
            scene_data = analysis_result.get('analysis', {}).get('scene', {})
            if not scene_data or 'spatial' not in scene_data:
                return ""
            
            spatial = scene_data['spatial']
            info_parts = []
            
            if 'width' in spatial:
                width = float(spatial['width'])
                if width > 0.7:
                    info_parts.append("wide stereo field")
                elif width > 0.4:
                    info_parts.append("moderate stereo width")
                else:
                    info_parts.append("narrow stereo field")
            
            if 'correlation' in spatial:
                correlation = float(spatial['correlation'])
                if correlation > 0.7:
                    info_parts.append("highly correlated channels")
                elif correlation < 0.3:
                    info_parts.append("decorrelated channels")
            
            if 'pan' in spatial:
                pan = float(spatial['pan'])
                if abs(pan) > 0.3:
                    direction = "right" if pan > 0 else "left"
                    info_parts.append(f"panned toward {direction}")
            
            return ", ".join(info_parts) if info_parts else "balanced spatial characteristics"
            
        except:
            return ""
    
    def _has_amen_break(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if track contains Amen break patterns."""
        try:
            sequence = analysis_result.get('alignment', {}).get('sequence', {})
            return bool(sequence.get('segments', []))
        except:
            return False
    
    def _extract_amen_info(self, analysis_result: Dict[str, Any]) -> str:
        """Extract Amen break information from analysis result."""
        try:
            sequence = analysis_result.get('alignment', {}).get('sequence', {})
            segments = sequence.get('segments', [])
            
            if not segments:
                return ""
            
            n_segments = len(segments)
            variation_names = sequence.get('variation_names', [])
            
            if variation_names:
                variations = ", ".join(set(variation_names))
                return f"{n_segments} instances detected with variations: {variations}"
            else:
                return f"{n_segments} instances detected"
            
        except:
            return "present"
    
    def _extract_transition_info(self, analysis_result: Dict[str, Any]) -> str:
        """Extract transition information from analysis result."""
        try:
            transitions = analysis_result.get('annotation', {}).get('transitions', {})
            if not transitions or 'points' not in transitions:
                return ""
            
            points = transitions.get('points', [])
            n_transitions = len(points)
            
            if n_transitions == 0:
                return ""
            
            types = transitions.get('types', [])
            if types and len(types) == n_transitions:
                type_counts = {}
                for t in types:
                    if t in type_counts:
                        type_counts[t] += 1
                    else:
                        type_counts[t] = 1
                
                type_info = ", ".join([f"{count} {t}" for t, count in type_counts.items()])
                return f"{n_transitions} transitions detected: {type_info}"
            else:
                return f"{n_transitions} transitions detected"
            
        except:
            return ""
    
    def _generate_summary_description(self, emotions: Dict[str, float]) -> str:
        """Generate summary description based on emotional profile."""
        # Find top 3 emotional characteristics
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]
        
        descriptions = []
        for emotion, value in top_emotions:
            if value >= 7.0:
                descriptions.append(f"highly {emotion}")
            elif value >= 5.0:
                descriptions.append(f"moderately {emotion}")
            elif value >= 3.0:
                descriptions.append(f"slightly {emotion}")
        
        if not descriptions:
            return "balanced and neutral"
        
        return " and ".join(descriptions)
    
    def _generate_mood_situations(self, emotions: Dict[str, float]) -> str:
        """Generate suitable situations based on emotional profile."""
        situations = []
        
        # High energy situations
        if emotions['energy'] > 7.0:
            if emotions['aggression'] > 6.0:
                situations.append("high-intensity workouts")
            else:
                situations.append("energetic dancing")
        
        # Medium energy situations
        elif emotions['energy'] > 4.0:
            if emotions['groove'] > 6.0:
                situations.append("social gatherings")
            else:
                situations.append("active listening")
        
        # Low energy situations
        else:
            if emotions['atmosphere'] > 6.0:
                situations.append("relaxation")
            else:
                situations.append("background ambience")
        
        # Additional situations based on other emotions
        if emotions['melancholy'] > 7.0:
            situations.append("introspective moments")
        
        if emotions['euphoria'] > 7.0:
            situations.append("uplifting experiences")
        
        if emotions['tension'] > 7.0 and emotions['atmosphere'] > 6.0:
            situations.append("focused concentration")
        
        if emotions['warmth'] > 7.0 and emotions['energy'] < 5.0:
            situations.append("winding down")
        
        return ", ".join(situations) if situations else "various listening contexts"
