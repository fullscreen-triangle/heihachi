# src/api/autobahn_integration.py
"""
Autobahn Probabilistic Reasoning Integration Layer

This module provides Python integration with the Autobahn oscillatory bio-metabolic
RAG system through the Heihachi Rust bridge interface. It handles all probabilistic
reasoning delegation while maintaining high-level API simplicity.

Architecture:
- Python API layer orchestrates requests
- Rust AutobahnBridge handles low-level communication
- Autobahn engine performs all probabilistic reasoning
- Results flow back through the bridge to Python
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel, Field
import aiohttp
from contextlib import asynccontextmanager

# Rust bridge integration (assuming PyO3 bindings)
try:
    from heihachi_rust import AutobahnBridge, AutobahnError
except ImportError:
    # Fallback for development/testing
    class AutobahnBridge:
        def __init__(self, *args, **kwargs):
            pass
    
    class AutobahnError(Exception):
        pass

logger = logging.getLogger(__name__)

@dataclass
class AutobahnConfig:
    """Configuration for Autobahn integration"""
    base_url: str = "http://localhost:8080"
    timeout_seconds: float = 30.0
    
    # Oscillatory dynamics configuration
    max_frequency_hz: float = 1000.0
    hierarchy_levels: List[str] = None
    coupling_strength: float = 0.85
    resonance_threshold: float = 0.7
    
    # Biological intelligence configuration
    atp_budget_per_query: float = 150.0
    metabolic_mode: str = "Mammalian"
    coherence_threshold: float = 0.85
    membrane_optimization: bool = True
    
    # Consciousness emergence configuration
    phi_calculation_enabled: bool = True
    emergence_threshold: float = 0.7
    workspace_integration: bool = True
    self_awareness_monitoring: bool = True
    
    def __post_init__(self):
        if self.hierarchy_levels is None:
            self.hierarchy_levels = [
                "Planck", "Quantum", "Atomic", "Molecular", "Cellular",
                "Neural", "Biological", "Behavioral", "Social", "Cosmic"
            ]

class FirePatternAnalysis(BaseModel):
    """Fire pattern data for Autobahn analysis"""
    dynamics: Dict[str, Any] = Field(..., description="Fire dynamics including flame height, color, etc.")
    emotional_context: Dict[str, Any] = Field(..., description="Emotional context from user")
    temporal_sequence: List[Dict[str, Any]] = Field(..., description="Time series of fire states")
    integration_info: Dict[str, Any] = Field(..., description="Information integration for consciousness")
    uncertainty_context: Dict[str, Any] = Field(..., description="Uncertainty parameters")

class AudioOptimizationRequest(BaseModel):
    """Request for Autobahn-optimized audio generation"""
    current_audio_state: Dict[str, Any] = Field(..., description="Current audio parameters and features")
    user_feedback: Dict[str, Any] = Field(..., description="User preference signals and engagement")
    fire_context: Dict[str, Any] = Field(..., description="Fire state and interaction history")
    optimization_goals: List[str] = Field(default_factory=list, description="Optimization objectives")

class ProbabilisticAnalysisResult(BaseModel):
    """Result from Autobahn probabilistic analysis"""
    audio_feature_distributions: Dict[str, Dict[str, float]]
    uncertainty_bounds: Dict[str, float]
    consciousness_influence: Dict[str, Any]
    metabolic_recommendations: Dict[str, Any]
    oscillatory_patterns: Dict[str, Any]
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

class OptimizedAudioResult(BaseModel):
    """Optimized audio parameters from Autobahn"""
    parameters: Dict[str, Any]
    confidence_intervals: List[Tuple[str, float, float]]
    metabolic_cost: float
    consciousness_alignment: float
    uncertainty_quantification: Dict[str, Any]
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict)

class AutobahnHealthStatus(BaseModel):
    """Health status of Autobahn system"""
    status: str
    capabilities: List[Dict[str, Any]]
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_check: datetime = Field(default_factory=datetime.now)

class AutobahnIntegrationManager:
    """
    High-level manager for Autobahn probabilistic reasoning integration
    
    This class provides a clean Python API for delegating all probabilistic
    tasks to the Autobahn engine through the Rust bridge interface.
    """
    
    def __init__(self, config: AutobahnConfig):
        self.config = config
        self.bridge: Optional[AutobahnBridge] = None
        self.health_status: Optional[AutobahnHealthStatus] = None
        self.last_health_check: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info(f"Initialized Autobahn integration with config: {config}")
    
    async def initialize(self) -> None:
        """Initialize the Autobahn bridge and verify connectivity"""
        try:
            # Initialize Rust bridge
            self.bridge = AutobahnBridge(
                base_url=self.config.base_url,
                oscillatory_config={
                    "max_frequency_hz": self.config.max_frequency_hz,
                    "hierarchy_levels": self.config.hierarchy_levels,
                    "coupling_strength": self.config.coupling_strength,
                    "resonance_threshold": self.config.resonance_threshold,
                },
                biological_config={
                    "atp_budget_per_query": self.config.atp_budget_per_query,
                    "metabolic_mode": self.config.metabolic_mode,
                    "coherence_threshold": self.config.coherence_threshold,
                    "membrane_optimization": self.config.membrane_optimization,
                },
                consciousness_config={
                    "phi_calculation_enabled": self.config.phi_calculation_enabled,
                    "emergence_threshold": self.config.emergence_threshold,
                    "workspace_integration": self.config.workspace_integration,
                    "self_awareness_monitoring": self.config.self_awareness_monitoring,
                }
            )
            
            # Create HTTP session for direct API calls if needed
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
            
            # Perform initial health check
            await self.health_check()
            
            logger.info("Autobahn integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Autobahn integration: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._session:
            await self._session.close()
        logger.info("Autobahn integration cleaned up")
    
    @asynccontextmanager
    async def session(self):
        """Async context manager for the integration"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def health_check(self) -> AutobahnHealthStatus:
        """Check Autobahn system health and capabilities"""
        try:
            start_time = datetime.now()
            
            # Use Rust bridge for health check
            health_data = await self.bridge.health_check()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.health_status = AutobahnHealthStatus(
                status=health_data.get("status", "Unknown"),
                capabilities=health_data.get("capabilities", []),
                resource_usage=health_data.get("resource_usage", {}),
                performance_metrics=health_data.get("performance_metrics", {}),
                last_check=datetime.now()
            )
            
            self.last_health_check = datetime.now()
            
            logger.info(f"Autobahn health check completed in {processing_time:.3f}s - Status: {self.health_status.status}")
            
            return self.health_status
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"Autobahn health check failed: {e}")
            raise
    
    async def analyze_fire_pattern(
        self,
        fire_pattern: FirePatternAnalysis,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ProbabilisticAnalysisResult:
        """
        Analyze fire pattern using Autobahn's probabilistic reasoning
        
        Delegates all probabilistic analysis to Autobahn including:
        - Oscillatory dynamics processing
        - Biological intelligence analysis
        - Consciousness emergence modeling
        - Uncertainty quantification
        """
        try:
            start_time = datetime.now()
            self.request_count += 1
            
            # Prepare data for Rust bridge
            fire_data = {
                "dynamics": fire_pattern.dynamics,
                "emotional_context": fire_pattern.emotional_context,
                "temporal_sequence": fire_pattern.temporal_sequence,
                "integration_info": fire_pattern.integration_info,
            }
            
            uncertainty_data = fire_pattern.uncertainty_context
            
            # Add user context if provided
            if user_context:
                fire_data["user_context"] = user_context
            
            # Delegate to Autobahn through Rust bridge
            analysis_result = await self.bridge.analyze_fire_pattern_probabilistic(
                fire_pattern=fire_data,
                uncertainty_context=uncertainty_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            result = ProbabilisticAnalysisResult(
                audio_feature_distributions=analysis_result.get("audio_feature_distributions", {}),
                uncertainty_bounds=analysis_result.get("uncertainty_bounds", {}),
                consciousness_influence=analysis_result.get("consciousness_influence", {}),
                metabolic_recommendations=analysis_result.get("metabolic_recommendations", {}),
                oscillatory_patterns=analysis_result.get("oscillatory_patterns", {}),
                processing_metadata={
                    "processing_time_ms": processing_time * 1000,
                    "autobahn_version": analysis_result.get("version", "unknown"),
                    "atp_consumed": analysis_result.get("atp_consumed", 0.0),
                    "consciousness_phi": analysis_result.get("phi_value", 0.0),
                    "biological_coherence": analysis_result.get("coherence_level", 0.0),
                }
            )
            
            logger.info(f"Fire pattern analysis completed in {processing_time:.3f}s")
            return result
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"Fire pattern analysis failed: {e}")
            raise
    
    async def optimize_audio_generation(
        self,
        optimization_request: AudioOptimizationRequest,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> OptimizedAudioResult:
        """
        Optimize audio generation using Autobahn's Bayesian inference
        
        Delegates optimization to Autobahn including:
        - Bayesian parameter optimization
        - Entropy optimization
        - Biological membrane processing
        - Uncertainty quantification
        """
        try:
            start_time = datetime.now()
            self.request_count += 1
            
            # Prepare optimization data
            audio_state = optimization_request.current_audio_state
            user_feedback = optimization_request.user_feedback
            fire_context = optimization_request.fire_context
            
            # Add user preferences if provided
            if user_preferences:
                user_feedback["preferences"] = user_preferences
            
            # Delegate to Autobahn through Rust bridge
            optimization_result = await self.bridge.optimize_audio_generation_bayesian(
                current_audio=audio_state,
                user_feedback=user_feedback,
                fire_context=fire_context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            result = OptimizedAudioResult(
                parameters=optimization_result.get("parameters", {}),
                confidence_intervals=optimization_result.get("confidence_intervals", []),
                metabolic_cost=optimization_result.get("metabolic_cost", 0.0),
                consciousness_alignment=optimization_result.get("consciousness_alignment", 0.0),
                uncertainty_quantification=optimization_result.get("uncertainty_quantification", {}),
                optimization_metadata={
                    "processing_time_ms": processing_time * 1000,
                    "optimization_goals": optimization_request.optimization_goals,
                    "convergence_iterations": optimization_result.get("iterations", 0),
                    "optimization_quality": optimization_result.get("quality_score", 0.0),
                    "bayesian_evidence": optimization_result.get("evidence", 0.0),
                }
            )
            
            logger.info(f"Audio optimization completed in {processing_time:.3f}s")
            return result
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"Audio optimization failed: {e}")
            raise
    
    async def update_user_model(
        self,
        user_id: str,
        interaction_history: List[Dict[str, Any]],
        physiological_data: Optional[Dict[str, Any]] = None,
        learning_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update user model using Autobahn's biological intelligence
        
        Delegates user modeling to Autobahn including:
        - Biological intelligence processing
        - ATP consumption pattern analysis
        - Consciousness profile updates
        - Learning metrics calculation
        """
        try:
            start_time = datetime.now()
            self.request_count += 1
            
            # Delegate to Autobahn through Rust bridge
            user_update = await self.bridge.update_user_model_biological(
                user_id=user_id,
                interaction_history=interaction_history,
                physiological_data=physiological_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            result = {
                "user_id": user_id,
                "updated_preferences": user_update.get("updated_preferences", []),
                "biological_profile": user_update.get("biological_profile", {}),
                "consciousness_profile": user_update.get("consciousness_profile", {}),
                "learning_metrics": user_update.get("learning_metrics", {}),
                "update_metadata": {
                    "processing_time_ms": processing_time * 1000,
                    "interactions_processed": len(interaction_history),
                    "model_version": user_update.get("model_version", "1.0"),
                    "adaptation_quality": user_update.get("adaptation_quality", 0.0),
                }
            }
            
            logger.info(f"User model update completed in {processing_time:.3f}s for user {user_id}")
            return result
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"User model update failed for user {user_id}: {e}")
            raise
    
    async def calculate_consciousness_emergence(
        self,
        integration_info: Dict[str, Any],
        system_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate consciousness emergence using Autobahn's IIT implementation
        
        Delegates consciousness calculation to Autobahn including:
        - Integrated Information Theory (IIT) Φ calculation
        - Emergence probability assessment
        - Critical transition analysis
        - Consciousness level determination
        """
        try:
            start_time = datetime.now()
            self.request_count += 1
            
            # Delegate to Autobahn through Rust bridge
            consciousness_result = await self.bridge.calculate_consciousness_emergence(
                integration_info=integration_info,
                current_state=system_state
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            result = {
                "phi_value": consciousness_result.get("phi_value", 0.0),
                "emergence_probability": consciousness_result.get("emergence_probability", 0.0),
                "critical_transitions": consciousness_result.get("critical_transitions", []),
                "consciousness_level": consciousness_result.get("consciousness_level", "Unconscious"),
                "calculation_metadata": {
                    "processing_time_ms": processing_time * 1000,
                    "iit_version": consciousness_result.get("iit_version", "3.0"),
                    "calculation_confidence": consciousness_result.get("confidence", 0.0),
                    "emergence_indicators": consciousness_result.get("emergence_indicators", []),
                }
            }
            
            logger.info(f"Consciousness calculation completed in {processing_time:.3f}s - Φ: {result['phi_value']:.3f}")
            return result
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"Consciousness calculation failed: {e}")
            raise
    
    async def quantify_uncertainty(
        self,
        model_parameters: List[float],
        input_data: Dict[str, Any],
        uncertainty_type: str = "Combined",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty using Autobahn's probabilistic methods
        
        Delegates uncertainty quantification to Autobahn including:
        - Aleatoric uncertainty (inherent randomness)
        - Epistemic uncertainty (model uncertainty)
        - Temporal uncertainty (time-varying)
        - Sensitivity analysis
        """
        try:
            start_time = datetime.now()
            self.request_count += 1
            
            # Delegate to Autobahn through Rust bridge
            uncertainty_result = await self.bridge.quantify_uncertainty(
                model_parameters=model_parameters,
                input_data=input_data,
                uncertainty_type=uncertainty_type
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            result = {
                "parameter_uncertainties": uncertainty_result.get("parameter_uncertainties", []),
                "model_confidence": uncertainty_result.get("model_confidence", 0.0),
                "prediction_intervals": uncertainty_result.get("prediction_intervals", []),
                "sensitivity_analysis": uncertainty_result.get("sensitivity_analysis", {}),
                "quantification_metadata": {
                    "processing_time_ms": processing_time * 1000,
                    "uncertainty_type": uncertainty_type,
                    "parameters_analyzed": len(model_parameters),
                    "uncertainty_method": uncertainty_result.get("method", "bayesian"),
                }
            }
            
            logger.info(f"Uncertainty quantification completed in {processing_time:.3f}s")
            return result
            
        except AutobahnError as e:
            self.error_count += 1
            logger.error(f"Uncertainty quantification failed: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the integration"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "request_count": self.request_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status.dict() if self.health_status else None,
        }
    
    async def reset_performance_metrics(self) -> None:
        """Reset performance tracking metrics"""
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info("Performance metrics reset")

class AutobahnPakatiIntegration:
    """
    Integration layer for Pakati Reference Understanding Engine with Autobahn
    
    This class specifically handles the validation of AI understanding through
    Autobahn's biological intelligence and consciousness modeling capabilities.
    """
    
    def __init__(self, autobahn_manager: AutobahnIntegrationManager):
        self.autobahn = autobahn_manager
        self.masking_strategies = [
            "partial_fire_masking",
            "temporal_fire_masking", 
            "spatial_fire_masking",
            "intensity_fire_masking"
        ]
    
    async def validate_fire_understanding(
        self,
        fire_pattern: Dict[str, Any],
        ai_interpretation: Dict[str, Any],
        validation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate AI understanding of fire patterns using Pakati method with Autobahn
        
        Uses Autobahn's biological intelligence to assess reconstruction quality
        and understanding depth through progressive masking strategies.
        """
        validation_results = []
        
        for strategy in self.masking_strategies:
            # Apply masking strategy to fire pattern
            masked_pattern = self._apply_masking_strategy(fire_pattern, strategy)
            
            # Use Autobahn to assess reconstruction quality
            reconstruction_assessment = await self.autobahn.analyze_fire_pattern(
                FirePatternAnalysis(
                    dynamics=masked_pattern.get("dynamics", {}),
                    emotional_context=masked_pattern.get("emotional_context", {}),
                    temporal_sequence=masked_pattern.get("temporal_sequence", []),
                    integration_info=masked_pattern.get("integration_info", {}),
                    uncertainty_context={"masking_strategy": strategy, "masking_level": 0.3}
                )
            )
            
            # Calculate understanding score using Autobahn's biological intelligence
            understanding_score = await self._calculate_understanding_score(
                original=fire_pattern,
                reconstruction=ai_interpretation,
                assessment=reconstruction_assessment,
                strategy=strategy
            )
            
            validation_results.append({
                "strategy": strategy,
                "understanding_score": understanding_score,
                "reconstruction_quality": reconstruction_assessment.processing_metadata.get("reconstruction_quality", 0.0),
                "biological_coherence": reconstruction_assessment.processing_metadata.get("biological_coherence", 0.0),
                "consciousness_alignment": reconstruction_assessment.consciousness_influence,
            })
        
        # Use Autobahn to synthesize overall understanding validation
        overall_understanding = await self._synthesize_understanding_validation(
            validation_results, validation_context
        )
        
        return {
            "is_valid": overall_understanding["meets_threshold"],
            "confidence": overall_understanding["confidence_level"],
            "understanding_depth": overall_understanding["depth_assessment"],
            "reconstruction_scores": validation_results,
            "autobahn_metadata": overall_understanding["autobahn_metadata"],
        }
    
    def _apply_masking_strategy(self, fire_pattern: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Apply specific masking strategy to fire pattern"""
        masked_pattern = fire_pattern.copy()
        
        if strategy == "partial_fire_masking":
            # Mask random portions of fire dynamics
            dynamics = masked_pattern.get("dynamics", {})
            if "intensity_variation" in dynamics:
                intensity = np.array(dynamics["intensity_variation"])
                mask_indices = np.random.choice(len(intensity), size=len(intensity)//3, replace=False)
                intensity[mask_indices] = 0.0
                dynamics["intensity_variation"] = intensity.tolist()
        
        elif strategy == "temporal_fire_masking":
            # Mask temporal sequences
            temporal_seq = masked_pattern.get("temporal_sequence", [])
            if temporal_seq:
                mask_count = len(temporal_seq) // 4
                mask_indices = np.random.choice(len(temporal_seq), size=mask_count, replace=False)
                for idx in mask_indices:
                    temporal_seq[idx] = {}
        
        elif strategy == "spatial_fire_masking":
            # Mask spatial information
            dynamics = masked_pattern.get("dynamics", {})
            if "movement_vectors" in dynamics:
                vectors = dynamics["movement_vectors"]
                mask_count = len(vectors) // 3
                mask_indices = np.random.choice(len(vectors), size=mask_count, replace=False)
                for idx in mask_indices:
                    vectors[idx] = (0.0, 0.0, 0.0)
        
        elif strategy == "intensity_fire_masking":
            # Mask intensity information
            dynamics = masked_pattern.get("dynamics", {})
            dynamics["flame_height"] = dynamics.get("flame_height", 1.0) * 0.5
            dynamics["color_temperature"] = 0.0
        
        return masked_pattern
    
    async def _calculate_understanding_score(
        self,
        original: Dict[str, Any],
        reconstruction: Dict[str, Any],
        assessment: ProbabilisticAnalysisResult,
        strategy: str
    ) -> float:
        """Calculate understanding score using Autobahn's assessment"""
        
        # Use Autobahn's uncertainty quantification for reconstruction quality
        uncertainty_result = await self.autobahn.quantify_uncertainty(
            model_parameters=[1.0],  # Placeholder
            input_data={
                "original": original,
                "reconstruction": reconstruction,
                "strategy": strategy
            },
            uncertainty_type="Epistemic"
        )
        
        # Calculate score based on Autobahn's biological intelligence metrics
        biological_score = assessment.consciousness_influence.get("phi_value", 0.0)
        uncertainty_penalty = uncertainty_result.get("model_confidence", 1.0)
        coherence_bonus = assessment.processing_metadata.get("biological_coherence", 0.0)
        
        understanding_score = (biological_score * uncertainty_penalty + coherence_bonus) / 2.0
        return min(max(understanding_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def _synthesize_understanding_validation(
        self,
        validation_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize overall understanding validation using Autobahn"""
        
        # Calculate weighted average of understanding scores
        scores = [result["understanding_score"] for result in validation_results]
        weights = [result["biological_coherence"].get("phi_value", 0.5) for result in validation_results]
        
        if sum(weights) > 0:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            weighted_score = sum(scores) / len(scores) if scores else 0.0
        
        # Use Autobahn consciousness calculation for validation threshold
        consciousness_data = await self.autobahn.calculate_consciousness_emergence(
            integration_info={
                "information_integration": weighted_score,
                "causal_connections": [],
                "workspace_activity": weighted_score,
                "metacognitive_signals": scores,
            },
            system_state={
                "current_activity": "understanding_validation",
                "resource_utilization": {"atp_consumption": 50.0},
                "network_connectivity": {"connection_density": 0.8},
                "temporal_dynamics": {"oscillation_frequency": 10.0},
            }
        )
        
        validation_threshold = 0.7  # Can be made configurable
        meets_threshold = weighted_score >= validation_threshold
        
        return {
            "meets_threshold": meets_threshold,
            "confidence_level": consciousness_data["phi_value"],
            "depth_assessment": weighted_score,
            "autobahn_metadata": {
                "consciousness_phi": consciousness_data["phi_value"],
                "emergence_probability": consciousness_data["emergence_probability"],
                "validation_scores": scores,
                "weighted_average": weighted_score,
            }
        }

# Utility functions for common operations

async def create_autobahn_manager(config: Optional[AutobahnConfig] = None) -> AutobahnIntegrationManager:
    """Create and initialize an Autobahn integration manager"""
    if config is None:
        config = AutobahnConfig()
    
    manager = AutobahnIntegrationManager(config)
    await manager.initialize()
    return manager

async def quick_fire_analysis(
    fire_pattern_data: Dict[str, Any],
    autobahn_url: str = "http://localhost:8080"
) -> ProbabilisticAnalysisResult:
    """Quick fire pattern analysis using default Autobahn configuration"""
    config = AutobahnConfig(base_url=autobahn_url)
    
    async with AutobahnIntegrationManager(config).session() as manager:
        fire_pattern = FirePatternAnalysis(
            dynamics=fire_pattern_data.get("dynamics", {}),
            emotional_context=fire_pattern_data.get("emotional_context", {}),
            temporal_sequence=fire_pattern_data.get("temporal_sequence", []),
            integration_info=fire_pattern_data.get("integration_info", {}),
            uncertainty_context=fire_pattern_data.get("uncertainty_context", {})
        )
        
        return await manager.analyze_fire_pattern(fire_pattern)

async def quick_audio_optimization(
    audio_request_data: Dict[str, Any],
    autobahn_url: str = "http://localhost:8080"
) -> OptimizedAudioResult:
    """Quick audio optimization using default Autobahn configuration"""
    config = AutobahnConfig(base_url=autobahn_url)
    
    async with AutobahnIntegrationManager(config).session() as manager:
        optimization_request = AudioOptimizationRequest(
            current_audio_state=audio_request_data.get("audio_state", {}),
            user_feedback=audio_request_data.get("user_feedback", {}),
            fire_context=audio_request_data.get("fire_context", {}),
            optimization_goals=audio_request_data.get("goals", [])
        )
        
        return await manager.optimize_audio_generation(optimization_request)

# Example usage and testing functions

async def test_autobahn_integration():
    """Test the Autobahn integration with sample data"""
    
    # Sample fire pattern data
    sample_fire_pattern = {
        "dynamics": {
            "flame_height": 1.5,
            "color_temperature": 2800.0,
            "intensity_variation": [0.8, 0.9, 0.7, 1.0, 0.85],
            "movement_vectors": [(0.1, 0.2, 0.0), (0.0, 0.1, 0.1)],
            "frequency_spectrum": [10.0, 20.0, 15.0, 25.0]
        },
        "emotional_context": {
            "primary_emotion": "joy",
            "emotion_intensity": 0.8,
            "emotion_stability": 0.6
        },
        "temporal_sequence": [
            {"timestamp": 1000, "flame_height": 1.4},
            {"timestamp": 2000, "flame_height": 1.6}
        ],
        "integration_info": {
            "information_integration": 0.75,
            "causal_connections": [],
            "workspace_activity": 0.8,
            "metacognitive_signals": [0.7, 0.8, 0.75]
        },
        "uncertainty_context": {
            "prior_uncertainty": 0.1,
            "environmental_noise": 0.05,
            "user_variability": 0.2,
            "measurement_precision": 0.95
        }
    }
    
    print("Testing Autobahn Integration...")
    
    # Test fire pattern analysis
    try:
        analysis_result = await quick_fire_analysis(sample_fire_pattern)
        print(f"✅ Fire analysis completed - Φ: {analysis_result.processing_metadata.get('consciousness_phi', 0.0):.3f}")
    except Exception as e:
        print(f"❌ Fire analysis failed: {e}")
    
    # Test audio optimization
    sample_audio_request = {
        "audio_state": {
            "current_parameters": {"frequency": 440.0, "amplitude": 0.8},
            "features": {"spectral_centroid": 1000.0, "energy": 0.7},
            "quality_metrics": {"signal_to_noise_ratio": 20.0},
            "user_satisfaction": 0.75
        },
        "user_feedback": {
            "preference_signals": [{"signal_type": "frequency", "value": 0.8, "confidence": 0.9}],
            "satisfaction_rating": 0.75,
            "engagement_metrics": {"interaction_duration": 120, "attention_score": 0.8}
        },
        "fire_context": {
            "current_fire_state": {"flame_height": 1.5, "color_temperature": 2800.0},
            "fire_history": [],
            "user_interaction_history": [],
            "neural_coupling": {"coupling_strength": 0.7, "coherence_measures": [0.8]}
        },
        "goals": ["maximize_engagement", "optimize_consciousness_alignment"]
    }
    
    try:
        optimization_result = await quick_audio_optimization(sample_audio_request)
        print(f"✅ Audio optimization completed - Cost: {optimization_result.metabolic_cost:.2f} ATP")
    except Exception as e:
        print(f"❌ Audio optimization failed: {e}")
    
    print("Autobahn integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_autobahn_integration()) 