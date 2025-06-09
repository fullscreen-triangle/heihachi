import numpy as np
from typing import Dict, Any, List, Tuple

class EmotionalFeatureMapper:
    """Maps technical audio features to emotional dimensions."""
    
    def __init__(self):
        # Feature importance weights for each emotional dimension
        self.energy_weights = {
            'loudness': 0.4,
            'tempo': 0.3,
            'dynamic_range': 0.1,
            'drum_intensity': 0.2
        }
        
        self.brightness_weights = {
            'spectral_centroid': 0.5,
            'spectral_rolloff': 0.3,
            'high_freq_energy': 0.2
        }
        
        self.tension_weights = {
            'dissonance': 0.4,
            'rhythmic_complexity': 0.3,
            'spectral_flatness': 0.3
        }
        
        self.warmth_weights = {
            'low_mid_energy': 0.5,
            'spectral_contrast': 0.3,
            'harmonic_richness': 0.2
        }
        
        self.groove_weights = {
            'microtiming': 0.4,
            'syncopation': 0.3,
            'bass_drum_interaction': 0.3
        }
        
        self.aggression_weights = {
            'transient_sharpness': 0.4,
            'distortion': 0.3,
            'high_freq_intensity': 0.3
        }
        
        self.atmosphere_weights = {
            'reverb_amount': 0.4,
            'stereo_width': 0.3,
            'background_texture': 0.3
        }
        
        self.melancholy_weights = {
            'minor_key': 0.5,
            'slow_tempo': 0.3,
            'sparse_arrangement': 0.2
        }
        
        self.euphoria_weights = {
            'major_key': 0.3,
            'uplifting_progression': 0.4,
            'bright_timbre': 0.3
        }
    
    def map_features_to_emotions(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Map technical features to emotional dimensions.
        
        Args:
            analysis_result: Complete analysis result from Triangle pipeline
            
        Returns:
            Dictionary of emotional dimensions with normalized values (0-10)
        """
        try:
            # Extract relevant features from analysis result
            features = self._extract_features(analysis_result)
            
            # Calculate emotional dimensions
            emotions = {
                "energy": self._calculate_energy(features, analysis_result),
                "brightness": self._calculate_brightness(features, analysis_result),
                "tension": self._calculate_tension(features, analysis_result),
                "warmth": self._calculate_warmth(features, analysis_result),
                "groove": self._calculate_groove(features, analysis_result),
                "aggression": self._calculate_aggression(features, analysis_result),
                "atmosphere": self._calculate_atmosphere(features, analysis_result),
                "melancholy": self._calculate_melancholy(features, analysis_result),
                "euphoria": self._calculate_euphoria(features, analysis_result)
            }
            
            # Normalize all values to 0-10 scale
            for key in emotions:
                emotions[key] = min(10, max(0, emotions[key]))
                
            return emotions
            
        except Exception as e:
            print(f"Error mapping features to emotions: {str(e)}")
            # Return default values if mapping fails
            return {
                "energy": 5.0,
                "brightness": 5.0,
                "tension": 5.0,
                "warmth": 5.0,
                "groove": 5.0,
                "aggression": 5.0,
                "atmosphere": 5.0,
                "melancholy": 5.0,
                "euphoria": 5.0
            }
    
    def _extract_features(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features from analysis result."""
        features = {}
        
        # Extract mix features
        mix_data = analysis_result.get('analysis', {}).get('mix', {})
        if mix_data:
            # Global features
            global_features = mix_data.get('global_features', {})
            features['loudness'] = float(global_features.get('loudness', 0))
            features['dynamic_range'] = float(global_features.get('dynamic_range', 0))
            features['crest_factor'] = float(global_features.get('crest_factor', 0))
            
            # Spectral features (average across chunks)
            spectral_data = mix_data.get('spectral', [])
            if len(spectral_data) > 0:
                spectral_features = np.mean(spectral_data.cpu().numpy(), axis=0) if hasattr(spectral_data, 'cpu') else np.mean(spectral_data, axis=0)
                features['spectral_centroid'] = float(spectral_features[0])
                features['spectral_bandwidth'] = float(spectral_features[1])
                features['spectral_flatness'] = float(spectral_features[2])
            
            # Rhythmic features
            rhythmic_data = mix_data.get('rhythmic', [])
            if len(rhythmic_data) > 0:
                rhythmic_features = np.mean(rhythmic_data.cpu().numpy(), axis=0) if hasattr(rhythmic_data, 'cpu') else np.mean(rhythmic_data, axis=0)
                features['tempo'] = float(rhythmic_features[0])
                features['tempo_strength'] = float(rhythmic_features[1])
        
        # Extract scene features
        scene_data = analysis_result.get('analysis', {}).get('scene', {})
        if scene_data:
            # Spatial features
            spatial = scene_data.get('spatial', {})
            if 'width' in spatial:
                features['stereo_width'] = float(spatial.get('width', 0))
            if 'correlation' in spatial:
                features['stereo_correlation'] = float(spatial.get('correlation', 0))
            if 'pan' in spatial:
                features['pan_position'] = float(spatial.get('pan', 0))
            
            # Background features
            bg_features = scene_data.get('background', {}).get('features', {})
            spectral_bg = bg_features.get('spectral', {})
            if spectral_bg:
                features['background_flatness'] = float(spectral_bg.get('flatness', 0))
                features['background_rolloff'] = float(spectral_bg.get('rolloff', 0))
                features['background_contrast'] = float(spectral_bg.get('contrast', 0))
        
        # Extract other analysis features
        features['bass_presence'] = self._extract_bass_presence(analysis_result)
        features['drum_complexity'] = self._extract_drum_complexity(analysis_result)
        features['groove_quality'] = self._extract_groove_quality(analysis_result)
        features['has_amen_break'] = 1.0 if self._has_amen_break(analysis_result) else 0.0
        
        return features
    
    def _extract_bass_presence(self, analysis_result: Dict[str, Any]) -> float:
        """Extract bass presence from analysis result."""
        try:
            baseline = analysis_result.get('features', {}).get('baseline', {})
            if baseline and 'presence' in baseline:
                return float(baseline['presence'])
            return 5.0  # Default value
        except:
            return 5.0
    
    def _extract_drum_complexity(self, analysis_result: Dict[str, Any]) -> float:
        """Extract drum complexity from analysis result."""
        try:
            drum = analysis_result.get('features', {}).get('drum', {})
            if drum and 'complexity' in drum:
                return float(drum['complexity'])
            return 5.0  # Default value
        except:
            return 5.0
    
    def _extract_groove_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Extract groove quality from analysis result."""
        try:
            groove = analysis_result.get('features', {}).get('groove', {})
            if groove and 'quality' in groove:
                return float(groove['quality'])
            return 5.0  # Default value
        except:
            return 5.0
    
    def _has_amen_break(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if track contains Amen break patterns."""
        try:
            sequence = analysis_result.get('alignment', {}).get('sequence', {})
            return bool(sequence.get('segments', []))
        except:
            return False
    
    def _calculate_energy(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate energy level from features."""
        energy = 0.0
        
        # Loudness contribution
        if 'loudness' in features:
            # Normalize loudness (assuming typical range)
            norm_loudness = min(10, features['loudness'] * 10)
            energy += self.energy_weights['loudness'] * norm_loudness
        
        # Tempo contribution
        if 'tempo' in features:
            # Normalize tempo (60-180 BPM range mapped to 0-10)
            norm_tempo = min(10, max(0, (features['tempo'] - 60) / 12))
            energy += self.energy_weights['tempo'] * norm_tempo
        
        # Dynamic range contribution (inverse relationship)
        if 'dynamic_range' in features:
            # Less dynamic range often means more energy in EDM
            norm_dynamic = max(0, 10 - min(10, features['dynamic_range'] * 5))
            energy += self.energy_weights['dynamic_range'] * norm_dynamic
        
        # Drum intensity from drum analysis
        drum_intensity = self._extract_drum_intensity(analysis_result)
        energy += self.energy_weights['drum_intensity'] * drum_intensity
        
        return energy
    
    def _calculate_brightness(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate brightness from features."""
        brightness = 5.0  # Default mid-level brightness
        
        # Spectral centroid contribution
        if 'spectral_centroid' in features:
            # Normalize centroid (assuming typical range for EDM)
            norm_centroid = min(10, features['spectral_centroid'] / 500)
            brightness += self.brightness_weights['spectral_centroid'] * norm_centroid
        
        # Spectral rolloff contribution
        if 'spectral_rolloff' in features:
            norm_rolloff = min(10, features['spectral_rolloff'] / 2000)
            brightness += self.brightness_weights['spectral_rolloff'] * norm_rolloff
        
        # High frequency energy
        high_freq = self._extract_high_freq_energy(analysis_result)
        brightness += self.brightness_weights['high_freq_energy'] * high_freq
        
        return brightness
    
    def _calculate_tension(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate tension from features."""
        tension = 5.0  # Default mid-level tension
        
        # Spectral flatness contribution (more noise-like = more tension)
        if 'spectral_flatness' in features:
            norm_flatness = min(10, features['spectral_flatness'] * 20)
            tension += self.tension_weights['spectral_flatness'] * norm_flatness
        
        # Rhythmic complexity
        rhythmic_complexity = self._extract_rhythmic_complexity(analysis_result)
        tension += self.tension_weights['rhythmic_complexity'] * rhythmic_complexity
        
        # Dissonance (estimated from spectral features)
        dissonance = self._estimate_dissonance(features)
        tension += self.tension_weights['dissonance'] * dissonance
        
        return tension
    
    def _calculate_warmth(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate warmth from features."""
        warmth = 5.0  # Default mid-level warmth
        
        # Low-mid energy
        low_mid = self._extract_low_mid_energy(analysis_result)
        warmth += self.warmth_weights['low_mid_energy'] * low_mid
        
        # Spectral contrast (lower contrast often feels warmer)
        if 'background_contrast' in features:
            norm_contrast = min(10, 10 - features['background_contrast'] * 2)
            warmth += self.warmth_weights['spectral_contrast'] * norm_contrast
        
        # Harmonic richness
        harmonic_richness = self._estimate_harmonic_richness(features)
        warmth += self.warmth_weights['harmonic_richness'] * harmonic_richness
        
        return warmth
    
    def _calculate_groove(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate groove quality from features."""
        groove = 5.0  # Default mid-level groove
        
        # Direct groove quality if available
        if 'groove_quality' in features:
            groove = features['groove_quality']
        else:
            # Microtiming deviations
            microtiming = self._extract_microtiming(analysis_result)
            groove += self.groove_weights['microtiming'] * microtiming
            
            # Syncopation
            syncopation = self._extract_syncopation(analysis_result)
            groove += self.groove_weights['syncopation'] * syncopation
            
            # Bass-drum interaction
            bass_drum = self._extract_bass_drum_interaction(analysis_result)
            groove += self.groove_weights['bass_drum_interaction'] * bass_drum
        
        return groove
    
    def _calculate_aggression(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate aggression from features."""
        aggression = 5.0  # Default mid-level aggression
        
        # Transient sharpness
        transients = self._extract_transient_sharpness(analysis_result)
        aggression += self.aggression_weights['transient_sharpness'] * transients
        
        # Distortion estimate
        distortion = self._estimate_distortion(features)
        aggression += self.aggression_weights['distortion'] * distortion
        
        # High frequency intensity
        high_freq = self._extract_high_freq_intensity(analysis_result)
        aggression += self.aggression_weights['high_freq_intensity'] * high_freq
        
        return aggression
    
    def _calculate_atmosphere(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate atmospheric quality from features."""
        atmosphere = 5.0  # Default mid-level atmosphere
        
        # Reverb amount
        reverb = self._estimate_reverb(features)
        atmosphere += self.atmosphere_weights['reverb_amount'] * reverb
        
        # Stereo width
        if 'stereo_width' in features:
            norm_width = min(10, features['stereo_width'] * 10)
            atmosphere += self.atmosphere_weights['stereo_width'] * norm_width
        
        # Background texture
        texture = self._estimate_background_texture(features)
        atmosphere += self.atmosphere_weights['background_texture'] * texture
        
        return atmosphere
    
    def _calculate_melancholy(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate melancholy from features."""
        melancholy = 5.0  # Default mid-level melancholy
        
        # Minor key estimate
        minor_key = self._estimate_minor_key(analysis_result)
        melancholy += self.melancholy_weights['minor_key'] * minor_key
        
        # Slow tempo contribution
        if 'tempo' in features:
            # Slower tempos tend to feel more melancholic
            slow_tempo = max(0, 10 - min(10, (features['tempo'] - 70) / 10))
            melancholy += self.melancholy_weights['slow_tempo'] * slow_tempo
        
        # Sparse arrangement
        sparseness = self._estimate_sparseness(features)
        melancholy += self.melancholy_weights['sparse_arrangement'] * sparseness
        
        return melancholy
    
    def _calculate_euphoria(self, features: Dict[str, float], analysis_result: Dict[str, Any]) -> float:
        """Calculate euphoria from features."""
        euphoria = 5.0  # Default mid-level euphoria
        
        # Major key estimate
        major_key = 10 - self._estimate_minor_key(analysis_result)
        euphoria += self.euphoria_weights['major_key'] * major_key
        
        # Uplifting progression estimate
        uplifting = self._estimate_uplifting_progression(analysis_result)
        euphoria += self.euphoria_weights['uplifting_progression'] * uplifting
        
        # Bright timbre
        if 'spectral_centroid' in features:
            bright_timbre = min(10, features['spectral_centroid'] / 500)
            euphoria += self.euphoria_weights['bright_timbre'] * bright_timbre
        
        return euphoria
    
    # Helper methods for extracting complex features
    
    def _extract_drum_intensity(self, analysis_result: Dict[str, Any]) -> float:
        """Extract drum intensity from analysis."""
        try:
            drum = analysis_result.get('features', {}).get('drum', {})
            if drum and 'intensity' in drum:
                return min(10, float(drum['intensity']))
            
            # Fallback: estimate from percussion analysis
            percussion = analysis_result.get('features', {}).get('percussion', {})
            if percussion and 'intensity' in percussion:
                return min(10, float(percussion['intensity']))
            
            return 5.0  # Default value
        except:
            return 5.0
    
    def _extract_high_freq_energy(self, analysis_result: Dict[str, Any]) -> float:
        """Estimate high frequency energy."""
        # This would ideally come from a spectral band analysis
        # For now, use a simple estimate based on spectral centroid
        try:
            mix_data = analysis_result.get('analysis', {}).get('mix', {})
            spectral_data = mix_data.get('spectral', [])
            if len(spectral_data) > 0:
                spectral_features = np.mean(spectral_data.cpu().numpy(), axis=0) if hasattr(spectral_data, 'cpu') else np.mean(spectral_data, axis=0)
                centroid = float(spectral_features[0])
                return min(10, centroid / 500)
            return 5.0
        except:
            return 5.0
    
    def _extract_rhythmic_complexity(self, analysis_result: Dict[str, Any]) -> float:
        """Extract rhythmic complexity."""
        try:
            rhythmic = analysis_result.get('features', {}).get('rhythmic', {})
            if rhythmic and 'complexity' in rhythmic:
                return min(10, float(rhythmic['complexity']))
            return 5.0
        except:
            return 5.0
    
    def _estimate_dissonance(self, features: Dict[str, float]) -> float:
        """Estimate dissonance from spectral features."""
        # This is a simplified estimate
        if 'spectral_flatness' in features and 'spectral_bandwidth' in features:
            return min(10, (features['spectral_flatness'] * 10 + features['spectral_bandwidth'] / 200) / 2)
        return 5.0
    
    def _extract_low_mid_energy(self, analysis_result: Dict[str, Any]) -> float:
        """Estimate low-mid frequency energy."""
        # This would ideally come from a spectral band analysis
        try:
            bass_presence = self._extract_bass_presence(analysis_result)
            # Adjust based on spectral centroid (lower centroid often means more low-mid energy)
            mix_data = analysis_result.get('analysis', {}).get('mix', {})
            spectral_data = mix_data.get('spectral', [])
            if len(spectral_data) > 0:
                spectral_features = np.mean(spectral_data.cpu().numpy(), axis=0) if hasattr(spectral_data, 'cpu') else np.mean(spectral_data, axis=0)
                centroid = float(spectral_features[0])
                centroid_factor = max(0, 10 - min(10, centroid / 300))
                return (bass_presence + centroid_factor) / 2
            return bass_presence
        except:
            return 5.0
    
    def _estimate_harmonic_richness(self, features: Dict[str, float]) -> float:
        """Estimate harmonic richness."""
        # This is a simplified estimate
        if 'spectral_flatness' in features:
            # Less noise-like (lower flatness) often means more harmonic content
            return min(10, 10 - features['spectral_flatness'] * 10)
        return 5.0
    
    def _extract_microtiming(self, analysis_result: Dict[str, Any]) -> float:
        """Extract microtiming deviations."""
        try:
            groove = analysis_result.get('features', {}).get('groove', {})
            if groove and 'microtiming' in groove:
                return min(10, float(groove['microtiming']) * 10)
            return 5.0
        except:
            return 5.0
    
    def _extract_syncopation(self, analysis_result: Dict[str, Any]) -> float:
        """Extract syncopation level."""
        try:
            rhythmic = analysis_result.get('features', {}).get('rhythmic', {})
            if rhythmic and 'syncopation' in rhythmic:
                return min(10, float(rhythmic['syncopation']) * 10)
            return 5.0
        except:
            return 5.0
    
    def _extract_bass_drum_interaction(self, analysis_result: Dict[str, Any]) -> float:
        """Extract bass-drum interaction quality."""
        # This would be a complex feature combining bass and drum analysis
        try:
            bass = self._extract_bass_presence(analysis_result)
            drums = self._extract_drum_complexity(analysis_result)
            # Simple estimate based on both being present and complex
            return (bass + drums) / 2
        except:
            return 5.0
    
    def _extract_transient_sharpness(self, analysis_result: Dict[str, Any]) -> float:
        """Extract transient sharpness."""
        try:
            percussion = analysis_result.get('features', {}).get('percussion', {})
            if percussion and 'transient_sharpness' in percussion:
                return min(10, float(percussion['transient_sharpness']) * 10)
            
            # Fallback: estimate from drum analysis
            drum = analysis_result.get('features', {}).get('drum', {})
            if drum and 'attack_strength' in drum:
                return min(10, float(drum['attack_strength']) * 10)
            
            return 5.0
        except:
            return 5.0
    
    def _estimate_distortion(self, features: Dict[str, float]) -> float:
        """Estimate distortion level."""
        # This is a simplified estimate
        if 'crest_factor' in features:
            # Lower crest factor often indicates more compression/distortion
            return min(10, 10 - features['crest_factor'])
        return 5.0
    
    def _extract_high_freq_intensity(self, analysis_result: Dict[str, Any]) -> float:
        """Extract high frequency intensity."""
        # Similar to high_freq_energy but focused on intensity rather than presence
        return self._extract_high_freq_energy(analysis_result)
    
    def _estimate_reverb(self, features: Dict[str, float]) -> float:
        """Estimate reverb amount."""
        # This is a simplified estimate
        if 'stereo_correlation' in features:
            # Higher correlation in stereo often indicates more reverb
            return min(10, features['stereo_correlation'] * 10)
        return 5.0
    
    def _estimate_background_texture(self, features: Dict[str, float]) -> float:
        """Estimate background texture complexity."""
        if 'background_flatness' in features and 'background_contrast' in features:
            return min(10, (features['background_flatness'] * 5 + features['background_contrast'] * 5) / 2)
        return 5.0
    
    def _estimate_minor_key(self, analysis_result: Dict[str, Any]) -> float:
        """Estimate likelihood of minor key."""
        # This would ideally come from key detection
        # For now, use a placeholder value
        return 5.0
    
    def _estimate_sparseness(self, features: Dict[str, float]) -> float:
        """Estimate arrangement sparseness."""
        # This is a simplified estimate
        try:
            # Less dynamic range often means less sparseness
            if 'dynamic_range' in features:
                return min(10, features['dynamic_range'] * 2)
            return 5.0
        except:
            return 5.0
    
    def _estimate_uplifting_progression(self, analysis_result: Dict[str, Any]) -> float:
        """Estimate uplifting quality of harmonic progression."""
        # This would ideally come from harmonic analysis
        # For now, use a placeholder value
        return 5.0

    def describe_emotion(self, emotion_name: str, value: float) -> str:
        """
        Generate a textual description of an emotional dimension.
        
        Args:
            emotion_name: Name of the emotion
            value: Value on a 0-10 scale
            
        Returns:
            Textual description
        """
        if value < 3.0:
            level = "low"
        elif value < 7.0:
            level = "moderate"
        else:
            level = "high"
            
        descriptions = {
            "energy": {
                "low": "relaxed and calm",
                "moderate": "moderately energetic",
                "high": "highly energetic and intense"
            },
            "brightness": {
                "low": "dark and subdued",
                "moderate": "balanced brightness",
                "high": "bright and vibrant"
            },
            "tension": {
                "low": "relaxed and resolved",
                "moderate": "moderately tense",
                "high": "highly tense and unresolved"
            },
            "warmth": {
                "low": "cold and clinical",
                "moderate": "moderately warm",
                "high": "very warm and rich"
            },
            "groove": {
                "low": "rigid and mechanical",
                "moderate": "moderately groovy",
                "high": "deeply groovy and flowing"
            },
            "aggression": {
                "low": "gentle and soft",
                "moderate": "moderately aggressive",
                "high": "highly aggressive and hard-hitting"
            },
            "atmosphere": {
                "low": "direct and immediate",
                "moderate": "moderately atmospheric",
                "high": "deeply atmospheric and immersive"
            },
            "melancholy": {
                "low": "upbeat and positive",
                "moderate": "slightly melancholic",
                "high": "deeply melancholic and emotional"
            },
            "euphoria": {
                "low": "grounded and serious",
                "moderate": "moderately uplifting",
                "high": "euphoric and ecstatic"
            }
        }
        
        return descriptions.get(emotion_name, {}).get(level, f"{level} {emotion_name}")
