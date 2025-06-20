// src/core/autobahn_bridge.rs
//! Autobahn Probabilistic Reasoning Bridge
//! 
//! This module provides the interface for Heihachi to delegate all probabilistic
//! reasoning, Bayesian inference, and biological intelligence tasks to the 
//! Autobahn oscillatory bio-metabolic RAG system.
//!
//! Architecture:
//! - Heihachi focuses on real-time audio processing and fire interface
//! - Autobahn handles all probabilistic reasoning and consciousness modeling
//! - Clean separation of concerns with high-performance bridge

use std::time::Duration;
use tokio::time::timeout;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use reqwest::Client;
use url::Url;

/// Main bridge interface to Autobahn probabilistic reasoning engine
#[derive(Debug, Clone)]
pub struct AutobahnBridge {
    client: Client,
    base_url: Url,
    timeout_duration: Duration,
    
    // Autobahn-specific configuration
    oscillatory_config: OscillatoryConfig,
    biological_config: BiologicalConfig,
    consciousness_config: ConsciousnessConfig,
}

/// Configuration for Autobahn oscillatory dynamics processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryConfig {
    pub max_frequency_hz: f64,
    pub hierarchy_levels: Vec<HierarchyLevel>,
    pub coupling_strength: f64,
    pub resonance_threshold: f64,
}

/// Configuration for Autobahn biological intelligence processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    pub atp_budget_per_query: f64,
    pub metabolic_mode: MetabolicMode,
    pub coherence_threshold: f64,
    pub membrane_optimization: bool,
}

/// Configuration for Autobahn consciousness emergence modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    pub phi_calculation_enabled: bool,
    pub emergence_threshold: f64,
    pub workspace_integration: bool,
    pub self_awareness_monitoring: bool,
}

/// Hierarchy levels for oscillatory processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchyLevel {
    Planck,      // 10^-44 s
    Quantum,     // 10^-24 s
    Atomic,      // 10^-12 s
    Molecular,   // 10^-9 s
    Cellular,    // 10^-3 s
    Neural,      // 10^-2 s
    Biological,  // 1 s
    Behavioral,  // 10^3 s
    Social,      // 10^6 s
    Cosmic,      // 10^13 s
}

/// Metabolic processing modes in Autobahn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetabolicMode {
    Flight,      // High-energy rapid processing
    ColdBlooded, // Energy-efficient sustained processing
    Mammalian,   // Balanced performance and efficiency
    Anaerobic,   // Emergency low-resource processing
}

/// Fire pattern data structure for Autobahn analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirePattern {
    pub dynamics: FireDynamics,
    pub emotional_context: EmotionalContext,
    pub temporal_sequence: Vec<FireState>,
    pub integration_info: IntegrationInfo,
}

/// Fire dynamics for oscillatory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireDynamics {
    pub flame_height: f64,
    pub color_temperature: f64,
    pub intensity_variation: Vec<f64>,
    pub movement_vectors: Vec<(f64, f64, f64)>,
    pub frequency_spectrum: Vec<f64>,
}

/// Emotional context for biological intelligence processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalContext {
    pub primary_emotion: String,
    pub emotion_intensity: f64,
    pub emotion_stability: f64,
    pub user_biometric_data: Option<BiometricData>,
}

/// Integration information for consciousness modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationInfo {
    pub information_integration: f64,
    pub causal_connections: Vec<CausalConnection>,
    pub workspace_activity: f64,
    pub metacognitive_signals: Vec<f64>,
}

/// Biometric data for enhanced biological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricData {
    pub heart_rate: Option<f64>,
    pub skin_conductance: Option<f64>,
    pub eye_tracking: Option<EyeTrackingData>,
    pub neural_activity: Option<Vec<f64>>,
}

/// Eye tracking data for attention modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeTrackingData {
    pub gaze_points: Vec<(f64, f64)>,
    pub fixation_duration: Vec<f64>,
    pub pupil_dilation: Vec<f64>,
}

/// Causal connections for consciousness modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConnection {
    pub source: String,
    pub target: String,
    pub strength: f64,
    pub causal_efficacy: f64,
}

/// Uncertainty context for probabilistic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyContext {
    pub prior_uncertainty: f64,
    pub environmental_noise: f64,
    pub user_variability: f64,
    pub measurement_precision: f64,
}

/// Probabilistic analysis result from Autobahn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticAnalysis {
    pub audio_feature_distributions: AudioFeatureDistributions,
    pub uncertainty_bounds: UncertaintyBounds,
    pub consciousness_influence: ConsciousnessInfluence,
    pub metabolic_recommendations: MetabolicRecommendations,
    pub oscillatory_patterns: OscillatoryPatterns,
}

/// Audio feature probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatureDistributions {
    pub frequency_distribution: ProbabilityDistribution,
    pub amplitude_distribution: ProbabilityDistribution,
    pub timbre_distribution: ProbabilityDistribution,
    pub temporal_distribution: ProbabilityDistribution,
}

/// Generic probability distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub confidence_interval: (f64, f64),
    pub distribution_type: DistributionType,
}

/// Types of probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    Beta,
    Gamma,
    Uniform,
    Exponential,
    Custom(String),
}

/// Uncertainty bounds for all parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    pub parameter_uncertainties: Vec<(String, f64)>,
    pub model_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub epistemic_uncertainty: f64,
}

/// Consciousness influence factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessInfluence {
    pub phi_value: f64,
    pub influence_factors: Vec<InfluenceFactor>,
    pub emergence_probability: f64,
    pub integration_strength: f64,
}

/// Individual influence factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluenceFactor {
    pub factor_name: String,
    pub influence_strength: f64,
    pub confidence: f64,
}

/// Metabolic recommendations from biological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicRecommendations {
    pub recommended_mode: MetabolicMode,
    pub atp_allocation: AtpAllocation,
    pub energy_efficiency: f64,
    pub resource_constraints: Vec<ResourceConstraint>,
}

/// ATP allocation across different processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpAllocation {
    pub processing_atp: f64,
    pub memory_atp: f64,
    pub communication_atp: f64,
    pub maintenance_atp: f64,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraint {
    pub resource_type: String,
    pub availability: f64,
    pub demand: f64,
    pub priority: i32,
}

/// Oscillatory patterns from multi-scale analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryPatterns {
    pub dominant_frequencies: Vec<f64>,
    pub cross_scale_coupling: Vec<ScaleCoupling>,
    pub resonance_patterns: Vec<ResonancePattern>,
    pub emergence_indicators: Vec<EmergenceIndicator>,
}

/// Cross-scale coupling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleCoupling {
    pub scale_1: HierarchyLevel,
    pub scale_2: HierarchyLevel,
    pub coupling_strength: f64,
    pub phase_relationship: f64,
}

/// Resonance patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonancePattern {
    pub frequency: f64,
    pub amplitude: f64,
    pub quality_factor: f64,
    pub resonance_type: String,
}

/// Emergence indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceIndicator {
    pub indicator_type: String,
    pub strength: f64,
    pub scale: HierarchyLevel,
    pub emergence_probability: f64,
}

/// Audio state for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioState {
    pub current_parameters: AudioParameters,
    pub features: AudioFeatures,
    pub quality_metrics: AudioQualityMetrics,
    pub user_satisfaction: f64,
}

/// Audio parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioParameters {
    pub frequency: f64,
    pub amplitude: f64,
    pub timbre_coefficients: Vec<f64>,
    pub spatial_positioning: (f64, f64, f64),
    pub temporal_envelope: Vec<f64>,
}

/// Audio features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub spectral_centroid: f64,
    pub spectral_rolloff: f64,
    pub mfcc_coefficients: Vec<f64>,
    pub zero_crossing_rate: f64,
    pub energy: f64,
}

/// Audio quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityMetrics {
    pub signal_to_noise_ratio: f64,
    pub total_harmonic_distortion: f64,
    pub dynamic_range: f64,
    pub perceptual_quality: f64,
}

/// User feedback for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub preference_signals: Vec<PreferenceSignal>,
    pub satisfaction_rating: f64,
    pub engagement_metrics: EngagementMetrics,
    pub physiological_response: Option<PhysiologicalResponse>,
}

/// Individual preference signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceSignal {
    pub signal_type: String,
    pub value: f64,
    pub confidence: f64,
    pub temporal_weight: f64,
}

/// Engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    pub interaction_duration: Duration,
    pub interaction_frequency: f64,
    pub attention_score: f64,
    pub emotional_engagement: f64,
}

/// Physiological response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologicalResponse {
    pub heart_rate_variability: f64,
    pub skin_conductance_response: f64,
    pub pupil_dilation_response: f64,
    pub neural_activity_change: Vec<f64>,
}

/// Fire context for audio generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireContext {
    pub current_fire_state: FireState,
    pub fire_history: Vec<FireState>,
    pub user_interaction_history: Vec<UserInteraction>,
    pub neural_coupling: NeuralCoupling,
}

/// Individual fire state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireState {
    pub timestamp: u64,
    pub flame_properties: FlameProperties,
    pub environmental_conditions: EnvironmentalConditions,
    pub user_emotional_state: EmotionalState,
}

/// Flame properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameProperties {
    pub height: f64,
    pub width: f64,
    pub color_temperature: f64,
    pub intensity: f64,
    pub movement_energy: f64,
}

/// Environmental conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalConditions {
    pub ambient_temperature: f64,
    pub humidity: f64,
    pub air_pressure: f64,
    pub wind_speed: f64,
}

/// User emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f64,    // Positive/negative
    pub arousal: f64,    // High/low energy
    pub dominance: f64,  // Control/submission
    pub emotion_labels: Vec<String>,
}

/// User interaction with fire
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    pub interaction_type: InteractionType,
    pub timestamp: u64,
    pub duration: Duration,
    pub intensity: f64,
    pub target_area: (f64, f64),
}

/// Types of user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Touch,
    Gesture,
    Breath,
    Movement,
    Voice,
    Gaze,
}

/// Neural coupling between fire and brain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCoupling {
    pub coupling_strength: f64,
    pub coherence_measures: Vec<f64>,
    pub phase_synchronization: f64,
    pub information_flow: Vec<InformationFlow>,
}

/// Information flow between systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlow {
    pub source: String,
    pub target: String,
    pub flow_rate: f64,
    pub information_content: f64,
}

/// Optimized audio parameters from Autobahn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedAudioParameters {
    pub parameters: AudioParameters,
    pub confidence_intervals: Vec<(String, f64, f64)>,
    pub metabolic_cost: f64,
    pub consciousness_alignment: f64,
    pub uncertainty_quantification: UncertaintyQuantification,
}

/// Detailed uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyQuantification {
    pub parameter_uncertainties: Vec<ParameterUncertainty>,
    pub model_confidence: f64,
    pub prediction_intervals: Vec<PredictionInterval>,
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Uncertainty for individual parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterUncertainty {
    pub parameter_name: String,
    pub mean: f64,
    pub variance: f64,
    pub confidence_interval: (f64, f64),
    pub distribution: ProbabilityDistribution,
}

/// Prediction intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    pub parameter: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub parameter_sensitivities: Vec<ParameterSensitivity>,
    pub global_sensitivity_indices: Vec<f64>,
    pub uncertainty_propagation: UncertaintyPropagation,
}

/// Sensitivity of individual parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivity {
    pub parameter_name: String,
    pub sensitivity_index: f64,
    pub confidence_interval: (f64, f64),
    pub interaction_effects: Vec<InteractionEffect>,
}

/// Interaction effects between parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEffect {
    pub parameter_1: String,
    pub parameter_2: String,
    pub interaction_strength: f64,
    pub significance: f64,
}

/// Uncertainty propagation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyPropagation {
    pub input_uncertainties: Vec<f64>,
    pub output_uncertainties: Vec<f64>,
    pub propagation_coefficients: Vec<f64>,
    pub uncertainty_amplification: f64,
}

/// Errors from Autobahn operations
#[derive(Debug, thiserror::Error)]
pub enum AutobahnError {
    #[error("Network communication error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("Timeout waiting for Autobahn response")]
    TimeoutError,
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Autobahn processing error: {message}")]
    ProcessingError { message: String },
    
    #[error("Insufficient ATP resources: required {required}, available {available}")]
    InsufficientATP { required: f64, available: f64 },
    
    #[error("Consciousness emergence threshold not met: {current} < {required}")]
    ConsciousnessThresholdError { current: f64, required: f64 },
    
    #[error("Biological coherence lost: {coherence_level}")]
    CoherenceLossError { coherence_level: f64 },
}

impl AutobahnBridge {
    /// Create a new Autobahn bridge with configuration
    pub fn new(
        base_url: Url,
        oscillatory_config: OscillatoryConfig,
        biological_config: BiologicalConfig,
        consciousness_config: ConsciousnessConfig,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        
        Ok(Self {
            client,
            base_url,
            timeout_duration: Duration::from_secs(30),
            oscillatory_config,
            biological_config,
            consciousness_config,
        })
    }
    
    /// Analyze fire pattern using Autobahn's probabilistic reasoning
    pub async fn analyze_fire_pattern_probabilistic(
        &self,
        fire_pattern: &FirePattern,
        uncertainty_context: &UncertaintyContext,
    ) -> Result<ProbabilisticAnalysis, AutobahnError> {
        let request_payload = serde_json::json!({
            "fire_pattern": fire_pattern,
            "uncertainty_context": uncertainty_context,
            "oscillatory_config": self.oscillatory_config,
            "biological_config": self.biological_config,
            "consciousness_config": self.consciousness_config,
        });
        
        let url = self.base_url.join("/api/v1/analyze_fire_pattern")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client
                .post(url)
                .json(&request_payload)
                .send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let analysis: ProbabilisticAnalysis = response.json().await?;
        Ok(analysis)
    }
    
    /// Optimize audio generation using Autobahn's Bayesian inference
    pub async fn optimize_audio_generation_bayesian(
        &self,
        current_audio: &AudioState,
        user_feedback: &UserFeedback,
        fire_context: &FireContext,
    ) -> Result<OptimizedAudioParameters, AutobahnError> {
        let request_payload = serde_json::json!({
            "current_audio": current_audio,
            "user_feedback": user_feedback,
            "fire_context": fire_context,
            "biological_config": self.biological_config,
            "consciousness_config": self.consciousness_config,
        });
        
        let url = self.base_url.join("/api/v1/optimize_audio_bayesian")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client
                .post(url)
                .json(&request_payload)
                .send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let optimized_params: OptimizedAudioParameters = response.json().await?;
        Ok(optimized_params)
    }
    
    /// Update user model using Autobahn's biological intelligence
    pub async fn update_user_model_biological(
        &self,
        user_id: &str,
        interaction_history: &[UserInteraction],
        physiological_data: &Option<PhysiologicalResponse>,
    ) -> Result<UserModelUpdate, AutobahnError> {
        let request_payload = serde_json::json!({
            "user_id": user_id,
            "interaction_history": interaction_history,
            "physiological_data": physiological_data,
            "biological_config": self.biological_config,
        });
        
        let url = self.base_url.join("/api/v1/update_user_model")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client
                .post(url)
                .json(&request_payload)
                .send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let user_update: UserModelUpdate = response.json().await?;
        Ok(user_update)
    }
    
    /// Calculate consciousness emergence using Autobahn's IIT implementation
    pub async fn calculate_consciousness_emergence(
        &self,
        integration_info: &IntegrationInfo,
        current_state: &SystemState,
    ) -> Result<ConsciousnessEmergence, AutobahnError> {
        let request_payload = serde_json::json!({
            "integration_info": integration_info,
            "current_state": current_state,
            "consciousness_config": self.consciousness_config,
        });
        
        let url = self.base_url.join("/api/v1/calculate_consciousness")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client
                .post(url)
                .json(&request_payload)
                .send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let consciousness: ConsciousnessEmergence = response.json().await?;
        Ok(consciousness)
    }
    
    /// Perform uncertainty quantification using Autobahn's probabilistic methods
    pub async fn quantify_uncertainty(
        &self,
        model_parameters: &[f64],
        input_data: &InputData,
        uncertainty_type: UncertaintyType,
    ) -> Result<UncertaintyQuantification, AutobahnError> {
        let request_payload = serde_json::json!({
            "model_parameters": model_parameters,
            "input_data": input_data,
            "uncertainty_type": uncertainty_type,
            "biological_config": self.biological_config,
        });
        
        let url = self.base_url.join("/api/v1/quantify_uncertainty")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client
                .post(url)
                .json(&request_payload)
                .send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let uncertainty: UncertaintyQuantification = response.json().await?;
        Ok(uncertainty)
    }
    
    /// Check Autobahn system health and capabilities
    pub async fn health_check(&self) -> Result<AutobahnHealth, AutobahnError> {
        let url = self.base_url.join("/api/v1/health")?;
        
        let response = timeout(
            self.timeout_duration,
            self.client.get(url).send()
        ).await
        .map_err(|_| AutobahnError::TimeoutError)?
        .map_err(AutobahnError::NetworkError)?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AutobahnError::ProcessingError {
                message: error_text,
            });
        }
        
        let health: AutobahnHealth = response.json().await?;
        Ok(health)
    }
}

/// User model update from biological intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserModelUpdate {
    pub user_id: String,
    pub updated_preferences: Vec<PreferenceUpdate>,
    pub biological_profile: BiologicalProfile,
    pub consciousness_profile: ConsciousnessProfile,
    pub learning_metrics: LearningMetrics,
}

/// Individual preference updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceUpdate {
    pub preference_type: String,
    pub old_value: f64,
    pub new_value: f64,
    pub confidence: f64,
    pub update_reason: String,
}

/// Biological profile of user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalProfile {
    pub metabolic_efficiency: f64,
    pub neural_coupling_strength: f64,
    pub membrane_coherence: f64,
    pub atp_consumption_patterns: Vec<AtpConsumptionPattern>,
}

/// ATP consumption patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpConsumptionPattern {
    pub activity_type: String,
    pub base_consumption: f64,
    pub efficiency_factor: f64,
    pub temporal_pattern: Vec<f64>,
}

/// Consciousness profile of user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessProfile {
    pub baseline_phi: f64,
    pub consciousness_variability: f64,
    pub emergence_patterns: Vec<EmergencePattern>,
    pub integration_capacity: f64,
}

/// Consciousness emergence patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencePattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub intensity: f64,
    pub duration: Duration,
    pub triggers: Vec<String>,
}

/// Learning metrics from biological intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetrics {
    pub adaptation_rate: f64,
    pub learning_efficiency: f64,
    pub memory_consolidation: f64,
    pub transfer_learning_capability: f64,
}

/// System state for consciousness calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub current_activity: String,
    pub resource_utilization: ResourceUtilization,
    pub network_connectivity: NetworkConnectivity,
    pub temporal_dynamics: TemporalDynamics,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_bandwidth: f64,
    pub atp_consumption: f64,
}

/// Network connectivity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnectivity {
    pub connection_density: f64,
    pub clustering_coefficient: f64,
    pub path_length: f64,
    pub small_world_index: f64,
}

/// Temporal dynamics information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDynamics {
    pub oscillation_frequency: f64,
    pub phase_coherence: f64,
    pub temporal_correlation: f64,
    pub predictability: f64,
}

/// Consciousness emergence calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEmergence {
    pub phi_value: f64,
    pub emergence_probability: f64,
    pub critical_transitions: Vec<CriticalTransition>,
    pub consciousness_level: ConsciousnessLevel,
}

/// Critical transitions in consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalTransition {
    pub transition_type: String,
    pub transition_probability: f64,
    pub transition_time: f64,
    pub stability_analysis: StabilityAnalysis,
}

/// Stability analysis of transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysis {
    pub eigenvalues: Vec<f64>,
    pub stability_index: f64,
    pub bifurcation_points: Vec<f64>,
    pub basin_of_attraction: f64,
}

/// Levels of consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessLevel {
    Unconscious,
    Subconscious,
    Preconscious,
    Conscious,
    SelfAware,
    Metacognitive,
    Transcendent,
}

/// Input data for uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputData {
    pub features: Vec<f64>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    pub temporal_sequence: Vec<Vec<f64>>,
    pub contextual_information: ContextualInformation,
}

/// Contextual information for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualInformation {
    pub environment_type: String,
    pub user_state: String,
    pub system_mode: String,
    pub external_conditions: Vec<ExternalCondition>,
}

/// External conditions affecting processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalCondition {
    pub condition_type: String,
    pub value: f64,
    pub uncertainty: f64,
    pub temporal_stability: f64,
}

/// Types of uncertainty to quantify
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyType {
    Aleatoric,   // Inherent randomness
    Epistemic,   // Model uncertainty
    Combined,    // Both types
    Temporal,    // Time-varying uncertainty
    Spatial,     // Spatial uncertainty
    Causal,      // Causal uncertainty
}

/// Autobahn system health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnHealth {
    pub status: SystemStatus,
    pub capabilities: Vec<Capability>,
    pub resource_usage: ResourceUsage,
    pub performance_metrics: PerformanceMetrics,
}

/// System status indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Healthy,
    Degraded,
    Critical,
    Offline,
}

/// Available capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub version: String,
    pub status: CapabilityStatus,
    pub performance_level: f64,
}

/// Status of individual capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapabilityStatus {
    Available,
    Limited,
    Unavailable,
    Maintenance,
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub atp_consumption: f64,
    pub memory_usage: f64,
    pub computational_load: f64,
    pub network_utilization: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub response_time: Duration,
    pub throughput: f64,
    pub accuracy: f64,
    pub reliability: f64,
} 