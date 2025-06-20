pub struct DigitalFire {
    // Visual fire properties
    pub flame_height: f32,      // Volume/energy
    pub flame_width: f32,       // Stereo width/spaciousness  
    pub color_temperature: f32, // Warmth/brightness (major/minor)
    pub flicker_rate: f32,      // Tempo/rhythm
    pub chaos_level: f32,       // Complexity/dissonance
    pub smoke_density: f32,     // Reverb/atmosphere
    pub ember_count: u32,       // Percussive elements
    pub fuel_type: FuelType,    // Genre/style characteristics
}

pub enum FuelType {
    Hardwood,    // Classical, jazz - long burning, steady
    Softwood,    // Pop, folk - quick ignition, bright flame
    Paper,       // Electronic - fast, intense, short-lived
    Oil,         // Metal, industrial - hot, aggressive
    Charcoal,    // Ambient, drone - slow, consistent heat
}

impl FireToAudioMapper {
    pub fn map_fire_to_audio(&self, fire: &DigitalFire) -> AudioSearchParams {
        AudioSearchParams {
            energy: fire.flame_height.powf(1.5),
            valence: self.color_temp_to_valence(fire.color_temperature),
            tempo: fire.flicker_rate * 60.0, // Convert to BPM
            complexity: fire.chaos_level,
            brightness: fire.color_temperature,
            spaciousness: fire.flame_width,
            percussiveness: fire.ember_count as f32 / 100.0,
            genre_weights: self.fuel_to_genre_weights(fire.fuel_type),
        }
    }
    
    fn color_temp_to_valence(&self, temp: f32) -> f32 {
        // Warm colors (red/orange) = positive valence
        // Cool colors (blue) = negative valence
        // Maps your fire consciousness theory perfectly
        (temp - 0.5) * 2.0 // Map 0-1 to -1 to 1
    }
}

class FireMaintenance {
    constructor(canvas) {
        this.fire = new DigitalFire();
        this.audioMapper = new FireToAudioMapper();
        this.llmReconstructor = new FireReconstructor();
    }
    
    // User actions
    addFuel(fuelType, amount) {
        this.fire.adjustForFuel(fuelType, amount);
        this.updateMusic();
    }
    
    pokeEmbers() {
        this.fire.chaos_level += 0.1;
        this.fire.flicker_rate *= 1.2;
        this.updateMusic();
    }
    
    adjustAirflow(direction) {
        if (direction === 'increase') {
            this.fire.flame_height *= 1.1;
            this.fire.flicker_rate += 0.05;
        }
        this.updateMusic();
    }
    
    async updateMusic() {
        // LLM tries to understand current fire state
        const fireDescription = await this.llmReconstructor.describe(this.fire);
        
        // Map fire state to audio parameters
        const audioParams = this.audioMapper.mapFireToAudio(this.fire);
        
        // Find matching music
        const matches = await this.findMusicMatches(audioParams);
        
        // Update playlist
        this.updatePlaylist(matches);
    }
}
//! # Fire-Music Interface: Complete Implementation
//! 
//! This is a comprehensive Rust implementation of the fire-music interface concept,
//! integrating audio analysis, fire simulation, and music discovery through
//! evolutionary fire-consciousness coupling.
//!
//! Based on the theory that fire control represents humanity's first abstraction
//! and the foundation of consciousness, this system allows users to discover
//! music by maintaining a digital fire rather than describing emotions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

// External crates that would be needed
// [dependencies]
// tokio = { version = "1.0", features = ["full"] }
// serde = { version = "1.0", features = ["derive"] }
// serde_json = "1.0"
// rustfft = "6.0"
// cpal = "0.15"
// rayon = "1.7"
// nalgebra = "0.32"
// rand = "0.8"

// ============================================================================
// CORE FIRE SIMULATION TYPES
// ============================================================================

/// Represents the state of a digital fire, mapping directly to musical qualities
/// through evolutionary fire-consciousness coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalFire {
    // Primary visual/physical properties
    pub flame_height: f32,      // 0.0-2.0: Energy/volume of music
    pub flame_width: f32,       // 0.0-2.0: Stereo width/spaciousness
    pub color_temperature: f32, // 0.0-1.0: Warmth (0=blue/sad, 1=red/happy)
    pub flicker_rate: f32,      // 0.1-10.0: Tempo/rhythm frequency
    pub chaos_level: f32,       // 0.0-1.0: Rhythmic/harmonic complexity
    pub smoke_density: f32,     // 0.0-1.0: Reverb/atmospheric effects
    pub ember_count: u32,       // 0-100: Percussive elements density
    pub heat_intensity: f32,    // 0.0-1.0: Overall energy/aggression
    
    // Fuel and maintenance state
    pub fuel_type: FuelType,
    pub fuel_amount: f32,       // 0.0-1.0: How much fuel remains
    pub airflow: f32,          // 0.0-2.0: Oxygen supply (affects all parameters)
    pub age: Duration,         // How long this fire has been burning
    
    // Internal simulation state
    pub particle_positions: Vec<FireParticle>,
    pub temperature_map: Vec<Vec<f32>>, // 2D temperature grid for realistic simulation
    pub wind_direction: f32,    // 0.0-360.0: Affects visual and audio spatialization
    
    // Historical state for evolution and learning
    pub state_history: Vec<FireSnapshot>,
    pub user_actions: Vec<UserAction>,
}

/// Different fuel types that affect fire behavior and map to musical genres
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FuelType {
    // Each fuel type has different burn characteristics that map to musical styles
    Hardwood,    // Classical, jazz - long burning, steady, complex harmonics
    Softwood,    // Pop, folk - quick ignition, bright flame, simple structure
    Paper,       // Electronic, breakbeat - fast, intense, short-lived bursts
    Oil,         // Metal, industrial - hot, aggressive, sustained intensity
    Charcoal,    // Ambient, drone - slow, consistent heat, minimal variation
    Kindling,    // Acoustic, singer-songwriter - delicate, needs attention
    Peat,        // Dark ambient, doom - smoldering, earthy, deep
    Alcohol,     // Experimental, glitch - unpredictable, blue flame, ethereal
}

/// Individual fire particle for realistic simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireParticle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub velocity_z: f32,
    pub temperature: f32,
    pub life_remaining: f32,
    pub size: f32,
}

/// Snapshot of fire state for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireSnapshot {
    pub timestamp: Instant,
    pub fire_state: DigitalFire,
    pub audio_match_quality: f32,
    pub user_satisfaction: Option<f32>, // If user provides feedback
}

/// User actions that affect the fire
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserAction {
    AddFuel { fuel_type: FuelType, amount: f32 },
    AdjustAirflow { delta: f32 },
    PokeEmbers { intensity: f32 },
    BankCoals,          // Save current state
    Rekindle,          // Return to banked state
    LetDieDown,        // Reduce all parameters gradually
    BuildUp,           // Increase energy gradually
    Stir { intensity: f32 },
    AddKindling,       // Quick energy boost
    Smother { amount: f32 }, // Reduce specific aspects
}

// ============================================================================
// AUDIO ANALYSIS AND MAPPING
// ============================================================================

/// Audio features extracted from music that correspond to fire parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    // Temporal features
    pub tempo: f32,              // BPM
    pub rhythm_complexity: f32,   // Syncopation, polyrhythm measure
    pub onset_density: f32,       // Percussive events per second
    
    // Spectral features  
    pub spectral_centroid: f32,   // Brightness
    pub spectral_rolloff: f32,    // High frequency content
    pub spectral_flux: f32,       // Rate of spectral change
    pub mfcc: Vec<f32>,          // Mel-frequency cepstral coefficients
    
    // Harmonic features
    pub harmonic_complexity: f32, // Dissonance, chord complexity
    pub tonal_stability: f32,     // Key stability
    pub pitch_range: f32,         // Frequency range used
    
    // Energy features
    pub rms_energy: f32,         // Overall energy
    pub dynamic_range: f32,       // Difference between loud/quiet parts
    pub attack_time: f32,        // How quickly sounds start
    
    // Spatial features
    pub stereo_width: f32,       // Stereo field usage
    pub reverb_amount: f32,      // Estimated reverb/space
    
    // Genre classification weights
    pub genre_probabilities: HashMap<String, f32>,
}

/// Parameters for searching/matching music based on fire state
#[derive(Debug, Clone)]
pub struct AudioSearchParams {
    pub energy_range: (f32, f32),
    pub tempo_range: (f32, f32),
    pub valence_range: (f32, f32),    // Emotional positivity
    pub complexity_range: (f32, f32),
    pub brightness_range: (f32, f32),
    pub genre_preferences: HashMap<String, f32>,
    pub similarity_threshold: f32,
}

/// Maps fire parameters to audio search parameters
pub struct FireToAudioMapper {
    // Calibration parameters learned from user feedback
    energy_scaling: f32,
    tempo_base: f32,
    complexity_sensitivity: f32,
    color_valence_mapping: fn(f32) -> f32,
}

impl FireToAudioMapper {
    pub fn new() -> Self {
        Self {
            energy_scaling: 1.5,
            tempo_base: 120.0, // Base BPM
            complexity_sensitivity: 2.0,
            color_valence_mapping: |temp| (temp - 0.5) * 2.0, // Map 0-1 to -1 to 1
        }
    }
    
    /// Core mapping function: converts fire state to audio search parameters
    pub fn map_fire_to_audio(&self, fire: &DigitalFire) -> AudioSearchParams {
        // Energy mapping: flame height + heat intensity
        let energy_base = (fire.flame_height * fire.heat_intensity).powf(self.energy_scaling);
        let energy_variance = fire.chaos_level * 0.3; // More chaos = wider energy range
        
        // Tempo mapping: flicker rate with fuel type modulation
        let tempo_base = self.tempo_base * fire.flicker_rate;
        let tempo_variance = match fire.fuel_type {
            FuelType::Paper | FuelType::Alcohol => 30.0, // High variance
            FuelType::Charcoal | FuelType::Peat => 10.0, // Low variance
            _ => 20.0,
        };
        
        // Valence (emotional positivity) from color temperature
        let valence_base = (self.color_valence_mapping)(fire.color_temperature);
        let valence_variance = fire.chaos_level * 0.4;
        
        // Complexity from chaos level and ember activity
        let complexity_base = fire.chaos_level * (1.0 + fire.ember_count as f32 / 100.0);
        let complexity_variance = 0.2;
        
        // Brightness from color temperature and flame height
        let brightness_base = fire.color_temperature * (0.5 + fire.flame_height * 0.5);
        let brightness_variance = fire.smoke_density * 0.3; // Smoke reduces brightness variance
        
        // Genre preferences based on fuel type
        let genre_preferences = self.fuel_to_genre_weights(fire.fuel_type);
        
        AudioSearchParams {
            energy_range: (
                (energy_base - energy_variance).max(0.0),
                (energy_base + energy_variance).min(1.0)
            ),
            tempo_range: (
                (tempo_base - tempo_variance).max(60.0),
                (tempo_base + tempo_variance).min(200.0)
            ),
            valence_range: (
                (valence_base - valence_variance).max(-1.0),
                (valence_base + valence_variance).min(1.0)
            ),
            complexity_range: (
                (complexity_base - complexity_variance).max(0.0),
                (complexity_base + complexity_variance).min(1.0)
            ),
            brightness_range: (
                (brightness_base - brightness_variance).max(0.0),
                (brightness_base + brightness_variance).min(1.0)
            ),
            genre_preferences,
            similarity_threshold: 0.7 - fire.chaos_level * 0.3, // More chaos = more diverse matches
        }
    }
    
    /// Convert fuel type to genre preference weights
    fn fuel_to_genre_weights(&self, fuel_type: FuelType) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        
        match fuel_type {
            FuelType::Hardwood => {
                weights.insert("classical".to_string(), 0.8);
                weights.insert("jazz".to_string(), 0.7);
                weights.insert("blues".to_string(), 0.6);
                weights.insert("folk".to_string(), 0.5);
            },
            FuelType::Softwood => {
                weights.insert("pop".to_string(), 0.8);
                weights.insert("folk".to_string(), 0.7);
                weights.insert("country".to_string(), 0.6);
                weights.insert("indie".to_string(), 0.6);
            },
            FuelType::Paper => {
                weights.insert("electronic".to_string(), 0.9);
                weights.insert("breakbeat".to_string(), 0.8);
                weights.insert("drum_and_bass".to_string(), 0.7);
                weights.insert("glitch".to_string(), 0.6);
            },
            FuelType::Oil => {
                weights.insert("metal".to_string(), 0.9);
                weights.insert("industrial".to_string(), 0.8);
                weights.insert("hardcore".to_string(), 0.7);
                weights.insert("punk".to_string(), 0.6);
            },
            FuelType::Charcoal => {
                weights.insert("ambient".to_string(), 0.9);
                weights.insert("drone".to_string(), 0.8);
                weights.insert("minimal".to_string(), 0.7);
                weights.insert("new_age".to_string(), 0.5);
            },
            FuelType::Kindling => {
                weights.insert("acoustic".to_string(), 0.8);
                weights.insert("singer_songwriter".to_string(), 0.7);
                weights.insert("indie_folk".to_string(), 0.6);
            },
            FuelType::Peat => {
                weights.insert("dark_ambient".to_string(), 0.9);
                weights.insert("doom".to_string(), 0.8);
                weights.insert("post_rock".to_string(), 0.6);
            },
            FuelType::Alcohol => {
                weights.insert("experimental".to_string(), 0.9);
                weights.insert("glitch".to_string(), 0.8);
                weights.insert("idm".to_string(), 0.7);
                weights.insert("noise".to_string(), 0.6);
            },
        }
        
        weights
    }
}

// ============================================================================
// AUDIO PROCESSING AND ANALYSIS
// ============================================================================
