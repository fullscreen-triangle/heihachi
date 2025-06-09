class LLMConnector:
    """Connects to Language Models for advanced text processing"""
    
    def __init__(self, api_key=None, config=None):
        self.config = config or {}
        
        # API key priority: direct param > config param > environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self.config.get("api_key", "")
            
        self.endpoint = self.config.get("endpoint", "")
        self.model = self.config.get("model", "text-davinci-003")
        
        # Check if we have a valid API key
        self.api_available = bool(self.api_key)
    
    def parse_query(self, query):
        """
        Parse a natural language query to determine intent and extract parameters
        
        This is a fallback implementation since we don't have actual API access.
        In production, this would call an LLM API.
        """
        query_lower = query.lower()
        
        # Check for emotional query
        emotional_terms = {
            "dark": ["dark", "darkness", "gloomy", "sinister", "ominous"],
            "bright": ["bright", "brightness", "uplifting", "cheerful", "happy"],
            "aggressive": ["aggressive", "hard", "intense", "angry", "heavy"],
            "atmospheric": ["atmospheric", "ambient", "spacious", "ethereal"],
            "deep": ["deep", "depth", "low", "bass", "sub"],
            "rolling": ["rolling", "flowing", "smooth", "liquid"]
        }
        
        # Check for technical query
        technical_terms = {
            "bass_energy": ["bass", "low end", "sub"],
            "spectral_centroid": ["bright", "dark", "tone", "tonal"],
            "percussion_intensity": ["drums", "percussion", "beats", "rhythm"],
            "stereo_width": ["stereo", "width", "wide", "spatial"],
            "reese_bass": ["reese", "bass", "modulation", "wobble"]
        }
        
        # Check for comparison
        if "compare" in query_lower or "difference" in query_lower or "versus" in query_lower or " vs " in query_lower:
            # This would extract track names in production
            return {
                "type": "comparative",
                "tracks": ["track1.wav", "track2.wav"]
            }
        
        # Check for recommendations
        if "recommend" in query_lower or "suggest" in query_lower:
            context = {}
            if "headphones" in query_lower:
                context["context"] = "headphones"
            elif "car" in query_lower:
                context["context"] = "car"
            else:
                context["context"] = "general"
                
            if "quiet" in query_lower:
                context["environment"] = "quiet"
            elif "loud" in query_lower or "noisy" in query_lower:
                context["environment"] = "noisy"
                
            return {
                "type": "recommendation",
                "context": context,
                "preferences": {
                    "emotional": self._extract_emotional_terms(query_lower, emotional_terms)
                }
            }
        
        # Check for emotional or technical query
        emotional_matches = self._extract_emotional_terms(query_lower, emotional_terms)
        if emotional_matches:
            return {
                "type": "emotional",
                "emotions": {term: {"intensity": score, "confidence": 0.8} 
                            for term, score in emotional_matches.items()}
            }
        
        technical_matches = self._extract_technical_terms(query_lower, technical_terms)
        if technical_matches:
            return {
                "type": "technical",
                "parameters": technical_matches
            }
        
        # Default to unknown intent
        return {
            "type": "unknown",
            "raw_query": query
        }
    
    def _extract_emotional_terms(self, text, term_dict):
        """Extract emotional terms and their intensities from text"""
        results = {}
        
        for term, synonyms in term_dict.items():
            for synonym in synonyms:
                if synonym in text:
                    # Look for intensity modifiers
                    intensity = 0.7  # Default intensity
                    
                    # Check for modifiers before the term
                    words = text.split()
                    for i, word in enumerate(words):
                        if word == synonym and i > 0:
                            modifier = words[i-1]
                            if modifier in ["very", "extremely", "incredibly"]:
                                intensity = 0.9
                            elif modifier in ["quite", "rather", "fairly"]:
                                intensity = 0.7
                            elif modifier in ["slightly", "somewhat", "a bit"]:
                                intensity = 0.5
                            elif modifier in ["not", "barely", "hardly"]:
                                intensity = 0.2
                    
                    results[term] = intensity
                    break
        
        return results
    
    def _extract_technical_terms(self, text, term_dict):
        """Extract technical terms and their values from text"""
        results = {}
        
        for term, synonyms in term_dict.items():
            for synonym in synonyms:
                if synonym in text:
                    # Look for value modifiers
                    value = 0.7  # Default value
                    
                    # Check for modifiers
                    if "high " + synonym in text or "strong " + synonym in text:
                        value = 0.9
                    elif "moderate " + synonym in text or "medium " + synonym in text:
                        value = 0.6
                    elif "low " + synonym in text or "light " + synonym in text:
                        value = 0.3
                    
                    results[term] = value
                    break
        
        return results
