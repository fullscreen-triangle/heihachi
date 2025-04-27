# src/huggingface/audio_analyzer.py
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import numpy as np
import torch
import logging
from typing import Dict, List, Union, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class HuggingFaceAudioAnalyzer:
    """Audio analysis using HuggingFace transformer models."""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", api_key=None, 
                genre_model=None, instrument_model=None, config=None):
        """Initialize the HuggingFace audio analyzer with specified models.
        
        Args:
            model_name: Base model name for general audio analysis
            api_key: HuggingFace API key for accessing models
            genre_model: Specific model for genre classification
            instrument_model: Specific model for instrument detection
            config: Additional configuration settings
        """
        self.api_key = api_key
        self.model_name = model_name
        self.genre_model = genre_model or "anton-l/wav2vec2-base-superb-gc"
        self.instrument_model = instrument_model or "alefiury/wav2vec2-large-xlsr-53-musical-instrument-classification"
        self.config = config or {}
        
        # Set the API key environment variable if provided
        if api_key:
            os.environ["HF_API_TOKEN"] = api_key
            logger.info("HuggingFace API token set from configuration")
            
        # Load the base feature extractor and model
        logger.info(f"Loading base model: {model_name}")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            logger.info(f"Base model loaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            self.feature_extractor = None
            self.model = None
            
        # Initialize specialized pipelines lazily
        self._genre_classifier = None
        self._instrument_detector = None
        
        # GPU settings
        if torch.cuda.is_available():
            gpu_memory_fraction = self.config.get("gpu_memory_fraction", 0.8)
            logger.info(f"Using GPU with memory fraction: {gpu_memory_fraction}")
            # Set GPU memory fraction
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            except:
                logger.warning("Failed to set GPU memory fraction")
        
    def analyze(self, audio_array: np.ndarray, sample_rate: int) -> Dict:
        """Analyze audio using the base pretrained model.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sampling rate of the audio
            
        Returns:
            Dict containing analysis results
        """
        if self.feature_extractor is None or self.model is None:
            return {"error": "Base model not loaded"}
            
        try:
            # Process audio in batches if needed
            batch_size_seconds = self.config.get("batch_size", 10)
            max_length = self.config.get("max_audio_length", 600)
            
            # Truncate if audio is too long
            if len(audio_array) > sample_rate * max_length:
                logger.warning(f"Audio exceeds maximum length of {max_length}s, truncating")
                audio_array = audio_array[:int(sample_rate * max_length)]
            
            # Process batches if needed
            if len(audio_array) > sample_rate * batch_size_seconds:
                return self._process_in_batches(audio_array, sample_rate, batch_size_seconds)
            
            # Process the entire audio at once
            inputs = self.feature_extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                self.model = self.model.to("cuda")
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Convert outputs to usable results
            results = self._format_outputs(outputs)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in HuggingFace analysis: {str(e)}")
            return {"error": str(e)}
    
    def _process_in_batches(self, audio_array: np.ndarray, sample_rate: int, 
                          batch_size_seconds: int) -> Dict:
        """Process long audio in batches to avoid memory issues.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sampling rate of the audio
            batch_size_seconds: Size of each batch in seconds
            
        Returns:
            Dict containing aggregated results
        """
        logger.info(f"Processing audio in batches of {batch_size_seconds}s")
        
        # Calculate batch size in samples
        batch_size_samples = int(sample_rate * batch_size_seconds)
        
        # Initialize results
        all_results = []
        
        # Process each batch
        for i in range(0, len(audio_array), batch_size_samples):
            batch = audio_array[i:i+batch_size_samples]
            logger.debug(f"Processing batch {i//batch_size_samples + 1}, samples {i} to {i+len(batch)}")
            
            # Process this batch
            inputs = self.feature_extractor(batch, sampling_rate=sample_rate, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Add results
            batch_results = self._format_outputs(outputs)
            all_results.append(batch_results)
            
            # Clear GPU cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Aggregate results from all batches
        return self._aggregate_batch_results(all_results)
    
    def _format_outputs(self, outputs) -> Dict:
        """Format model outputs into usable results.
        
        Args:
            outputs: Model output tensors
            
        Returns:
            Dict containing formatted results
        """
        # Get probabilities from logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Move to CPU for numpy conversion
            if probs.is_cuda:
                probs = probs.cpu()
                
            # Convert to numpy
            probs_np = probs.numpy()
            
            # Get top k predictions
            top_k = self.config.get("top_k", 5)
            if probs_np.shape[1] < top_k:
                top_k = probs_np.shape[1]
                
            top_indices = np.argsort(probs_np[0])[-top_k:][::-1]
            top_probs = probs_np[0][top_indices]
            
            # Get class labels if available
            if hasattr(self.model.config, "id2label"):
                labels = [self.model.config.id2label[idx] for idx in top_indices]
            else:
                labels = [f"class_{idx}" for idx in top_indices]
                
            # Format results
            results = {
                "predictions": [
                    {"label": label, "score": float(prob)} 
                    for label, prob in zip(labels, top_probs)
                ]
            }
            
            return results
        
        return {"error": "No logits found in model output"}
    
    def _aggregate_batch_results(self, batch_results: List[Dict]) -> Dict:
        """Aggregate results from multiple batches.
        
        Args:
            batch_results: List of results from individual batches
            
        Returns:
            Dict containing aggregated results
        """
        if not batch_results:
            return {"error": "No batch results to aggregate"}
            
        # Combine predictions from all batches
        all_predictions = []
        for result in batch_results:
            if "predictions" in result:
                all_predictions.extend(result["predictions"])
                
        # Combine similar predictions (same label)
        combined_predictions = {}
        for pred in all_predictions:
            label = pred["label"]
            score = pred["score"]
            
            if label in combined_predictions:
                combined_predictions[label]["score"] += score
                combined_predictions[label]["count"] += 1
            else:
                combined_predictions[label] = {"score": score, "count": 1}
                
        # Average the scores
        for label in combined_predictions:
            combined_predictions[label]["score"] /= combined_predictions[label]["count"]
            
        # Convert to list and sort by score
        predictions = [
            {"label": label, "score": data["score"]} 
            for label, data in combined_predictions.items()
        ]
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply confidence threshold
        threshold = self.config.get("confidence_threshold", 0.6)
        predictions = [p for p in predictions if p["score"] >= threshold]
        
        # Limit to top k
        top_k = self.config.get("top_k", 5)
        predictions = predictions[:top_k]
        
        return {"predictions": predictions}
    
    def classify_genre(self, audio_array: np.ndarray, sample_rate: int) -> Dict:
        """Classify music genre using specialized models.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sampling rate of the audio
            
        Returns:
            Dict containing genre classification results
        """
        try:
            # Initialize classifier if needed
            if self._genre_classifier is None:
                logger.info(f"Initializing genre classifier with model: {self.genre_model}")
                self._genre_classifier = pipeline(
                    "audio-classification", 
                    model=self.genre_model,
                    device=0 if torch.cuda.is_available() else -1
                )
                
            # Truncate if audio is too long
            max_length = self.config.get("max_audio_length", 600)
            if len(audio_array) > sample_rate * max_length:
                logger.warning(f"Audio exceeds maximum length of {max_length}s for genre classification, truncating")
                audio_array = audio_array[:int(sample_rate * max_length)]
                
            # Process the audio
            result = self._genre_classifier({"raw": audio_array, "sampling_rate": sample_rate})
            
            # Format the result
            return {
                "genres": [
                    {"genre": item["label"], "confidence": item["score"]} 
                    for item in result
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in genre classification: {str(e)}")
            return {"error": str(e)}
    
    def detect_instruments(self, audio_array: np.ndarray, sample_rate: int) -> Dict:
        """Detect instruments in audio.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sampling rate of the audio
            
        Returns:
            Dict containing instrument detection results
        """
        try:
            # Initialize detector if needed
            if self._instrument_detector is None:
                logger.info(f"Initializing instrument detector with model: {self.instrument_model}")
                self._instrument_detector = pipeline(
                    "audio-classification", 
                    model=self.instrument_model,
                    device=0 if torch.cuda.is_available() else -1
                )
                
            # Truncate if audio is too long
            max_length = self.config.get("max_audio_length", 600)
            if len(audio_array) > sample_rate * max_length:
                logger.warning(f"Audio exceeds maximum length of {max_length}s for instrument detection, truncating")
                audio_array = audio_array[:int(sample_rate * max_length)]
                
            # Process the audio
            result = self._instrument_detector({"raw": audio_array, "sampling_rate": sample_rate})
            
            # Format the result
            return {
                "instruments": [
                    {"instrument": item["label"], "confidence": item["score"]} 
                    for item in result
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in instrument detection: {str(e)}")
            return {"error": str(e)}