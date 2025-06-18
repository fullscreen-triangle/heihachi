//! Heihachi Engine - Understanding Through Reconstruction
//! 
//! "What makes a tiger so strong is that it lacks humanity"
//! 
//! This module implements the core Heihachi audio understanding engine that proves
//! audio comprehension through perfect reconstruction.

use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};

/// Main Heihachi engine for audio understanding through reconstruction
#[derive(Debug, Clone)]
pub struct HeihachiEngine {
    /// Engine configuration
    pub config: HeihachiConfig,
    /// Spectral analyzer
    pub spectral_analyzer: SpectralAnalyzer,
    /// Temporal analyzer
    pub temporal_analyzer: TemporalAnalyzer,
    /// Reconstructor
    pub reconstructor: AudioReconstructor,
    /// Validator
    pub validator: ReconstructionValidator,
}

/// Configuration for Heihachi engine
#[derive(Debug, Clone)]
pub struct HeihachiConfig {
    /// FFT size for analysis
    pub fft_size: usize,
    /// Hop size for analysis
    pub hop_size: usize,
    /// Sample rate for processing
    pub sample_rate: u32,
    /// Target reconstruction fidelity
    pub target_fidelity: f64,
    /// Maximum reconstruction iterations
    pub max_iterations: u32,
}

/// Spectral analyzer
#[derive(Debug, Clone)]
pub struct SpectralAnalyzer {
    /// FFT size for analysis
    pub fft_size: usize,
    /// Window function
    pub window: Vec<f64>,
}

/// Temporal analyzer
#[derive(Debug, Clone)]
pub struct TemporalAnalyzer {
    /// Frame size for analysis
    pub frame_size: usize,
    /// Onset threshold for detection
    pub onset_threshold: f64,
}

/// Reconstructor
#[derive(Debug, Clone)]
pub struct AudioReconstructor {
    /// Reconstruction method
    pub method: ReconstructionMethod,
    /// Number of iterations
    pub iterations: u32,
}

/// Reconstruction methods
#[derive(Debug, Clone, PartialEq)]
pub enum ReconstructionMethod {
    /// Phase vocoder reconstruction
    PhaseVocoder,
    /// Griffin-Lim algorithm
    GriffinLim,
    /// Hybrid approach
    Hybrid,
}

/// Validator
#[derive(Debug, Clone)]
pub struct ReconstructionValidator {
    /// Metrics to compute
    pub metrics: Vec<ValidationMetric>,
}

/// Validation metrics
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationMetric {
    /// Signal-to-noise ratio
    SNR,
    /// Spectral distance
    SpectralDistance,
    /// Perceptual evaluation
    PerceptualEvaluation,
}

/// Audio understanding result from Heihachi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioUnderstandingResult {
    /// Overall understanding quality
    pub understanding_quality: f64,
    /// Reconstruction fidelity
    pub reconstruction_fidelity: f64,
    /// Confidence in understanding
    pub confidence: f64,
    /// Spectral features
    pub spectral_features: SpectralFeatures,
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    /// Detected patterns
    pub detected_patterns: Vec<AudioPattern>,
}

/// Spectral features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    /// Frequency bins
    pub frequency_bins: Vec<f64>,
    /// Magnitudes
    pub magnitudes: Vec<f64>,
    /// Phases
    pub phases: Vec<f64>,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Spectral bandwidth
    pub spectral_bandwidth: f64,
}

/// Temporal features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Onsets
    pub onsets: Vec<f64>,
    /// Envelope
    pub envelope: Vec<f64>,
    /// Zero crossing rate
    pub zero_crossing_rate: f64,
    /// Temporal centroid
    pub temporal_centroid: f64,
}

/// Audio pattern detected by understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Start time
    pub start_time: f64,
    /// Duration
    pub duration: f64,
    /// Confidence
    pub confidence: f64,
    /// Parameters
    pub parameters: HashMap<String, f64>,
}

impl HeihachiEngine {
    /// Create a new Heihachi engine
    pub fn new(config: HeihachiConfig) -> Result<Self> {
        let window = Self::create_hanning_window(config.fft_size);
        
        Ok(Self {
            spectral_analyzer: SpectralAnalyzer {
                fft_size: config.fft_size,
                window,
            },
            temporal_analyzer: TemporalAnalyzer {
                frame_size: config.hop_size,
                onset_threshold: 0.3,
            },
            reconstructor: AudioReconstructor {
                method: ReconstructionMethod::Hybrid,
                iterations: config.max_iterations,
            },
            validator: ReconstructionValidator {
                metrics: vec![
                    ValidationMetric::SNR,
                    ValidationMetric::SpectralDistance,
                    ValidationMetric::PerceptualEvaluation,
                ],
            },
            config,
        })
    }

    /// Understand audio through reconstruction
    pub fn understand_audio(&mut self, audio_path: &Path) -> Result<AudioUnderstandingResult> {
        // Load audio data
        let audio_data = self.load_audio(audio_path)?;
        
        // Perform spectral analysis
        let spectral_features = self.analyze_spectrum(&audio_data)?;
        
        // Perform temporal analysis
        let temporal_features = self.analyze_temporal(&audio_data)?;
        
        // Detect patterns
        let patterns = self.detect_patterns(&audio_data)?;
        
        // Attempt reconstruction
        let reconstructed = self.reconstruct_audio(&spectral_features)?;
        
        // Validate reconstruction
        let fidelity = self.calculate_fidelity(&audio_data, &reconstructed)?;
        
        // Calculate understanding quality
        let understanding_quality = self.calculate_understanding_quality(&spectral_features, &temporal_features, fidelity);
        
        Ok(AudioUnderstandingResult {
            understanding_quality,
            reconstruction_fidelity: fidelity,
            confidence: understanding_quality * 0.9, // Conservative confidence
            spectral_features,
            temporal_features,
            detected_patterns: patterns,
        })
    }

    /// Autonomous reconstruction for validation
    pub fn autonomous_reconstruction(&self, understanding: &AudioUnderstandingResult) -> Result<Vec<f32>> {
        self.reconstruct_from_features(&understanding.spectral_features)
    }

    /// Calculate reconstruction fidelity
    pub fn reconstruction_fidelity(&self, original: &[f32], reconstructed: &[f32]) -> Result<f64> {
        self.calculate_fidelity(original, reconstructed)
    }

    // Implementation methods
    fn load_audio(&self, audio_path: &Path) -> Result<Vec<f32>> {
        use symphonia::core::audio::{AudioBuffer, Signal};
        use symphonia::core::codecs::{Decoder, DecoderOptions};
        use symphonia::core::formats::{FormatOptions, FormatReader};
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;
        use std::fs::File;
        
        // Open the audio file
        let file = File::open(audio_path)
            .map_err(|e| Error::Processing(format!("Failed to open audio file: {}", e)))?;
        
        // Create a media source stream
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        
        // Create a probe hint using the file extension
        let mut hint = Hint::new();
        if let Some(extension) = audio_path.extension() {
            if let Some(extension_str) = extension.to_str() {
                hint.with_extension(extension_str);
            }
        }
        
        // Use the default options for metadata and format readers
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        
        // Probe the media source
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| Error::Processing(format!("Failed to probe audio format: {}", e)))?;
            
        // Get the instantiated format reader
        let mut format = probed.format;
        
        // Find the first audio track with a known codec
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| Error::Processing("No supported audio tracks found".to_string()))?;
            
        let track_id = track.id;
        
        // Use the default options for the decoder
        let dec_opts: DecoderOptions = Default::default();
        
        // Create a decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| Error::Processing(format!("Failed to create decoder: {}", e)))?;
            
        // Store decoded audio samples
        let mut audio_samples = Vec::new();
        let target_sample_rate = self.config.sample_rate;
        
        // Decode the packets
        loop {
            // Get the next packet from the media format
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    // The track list has been changed. Re-examine it and create a new set of decoders,
                    // then restart the decode loop. This is an advanced feature and we'll just break.
                    break;
                }
                Err(symphonia::core::errors::Error::IoError(err)) => {
                    // The packet reader has reached EOF
                    if err.kind() == std::io::ErrorKind::UnexpectedEof {
                        break;
                    }
                    return Err(Error::Processing(format!("IO error: {}", err)));
                }
                Err(err) => {
                    return Err(Error::Processing(format!("Decode error: {}", err)));
                }
            };
            
            // Only decode packets for the selected track
            if packet.track_id() != track_id {
                continue;
            }
            
            // Decode the packet into audio samples
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Convert audio buffer to f32 samples
                    self.convert_audio_buffer_to_samples(&decoded, &mut audio_samples)?;
                }
                Err(symphonia::core::errors::Error::IoError(_)) => {
                    // End of stream
                    break;
                }
                Err(symphonia::core::errors::Error::DecodeError(_)) => {
                    // Decode error, try to continue
                    continue;
                }
                Err(err) => {
                    return Err(Error::Processing(format!("Decoder error: {}", err)));
                }
            }
        }
        
        // Resample if necessary
        if let Some(source_sample_rate) = track.codec_params.sample_rate {
            if source_sample_rate != target_sample_rate {
                audio_samples = self.resample_audio(audio_samples, source_sample_rate, target_sample_rate)?;
            }
        }
        
        Ok(audio_samples)
    }
    
    /// Convert audio buffer to f32 samples
    fn convert_audio_buffer_to_samples(&self, decoded: &symphonia::core::audio::AudioBufferRef, samples: &mut Vec<f32>) -> Result<()> {
        use symphonia::core::audio::{AudioBufferRef, Signal};
        use symphonia::core::sample::{Sample, i24, u24};
        
        match decoded {
            AudioBufferRef::U8(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::U16(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::U24(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::U32(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::S8(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::S16(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::S24(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::S32(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample.to_f32());
                }
            }
            AudioBufferRef::F32(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample);
                }
            }
            AudioBufferRef::F64(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(sample as f32);
                }
            }
        }
        
        Ok(())
    }
    
    /// Resample audio to target sample rate
    fn resample_audio(&self, input: Vec<f32>, source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
        use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};
        
        if source_rate == target_rate {
            return Ok(input);
        }
        
        let ratio = target_rate as f64 / source_rate as f64;
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Linear,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        
        let mut resampler = SincFixedIn::<f32>::new(
            ratio,
            2.0, // max_resample_ratio_relative
            params,
            input.len(),
            1, // channels
        ).map_err(|e| Error::Processing(format!("Failed to create resampler: {}", e)))?;
        
        // Convert to the format expected by rubato (vector of channel vectors)
        let input_channels = vec![input];
        
        let output_channels = resampler.process(&input_channels, None)
            .map_err(|e| Error::Processing(format!("Resampling failed: {}", e)))?;
        
        Ok(output_channels[0].clone())
    }

    fn analyze_spectrum(&self, audio: &[f32]) -> Result<SpectralFeatures> {
        let frames = self.frame_audio(audio);
        let mut frequency_bins = Vec::new();
        let mut magnitudes = Vec::new();
        let mut phases = Vec::new();
        
        // Compute FFT for each frame and average
        let mut spectral_centroid_sum = 0.0;
        let mut spectral_bandwidth_sum = 0.0;
        let mut frame_count = 0;
        
        for frame in &frames {
            let (frame_freqs, frame_mags, frame_phases) = self.compute_fft(frame)?;
            
            // For the first frame, initialize bins
            if frequency_bins.is_empty() {
                frequency_bins = frame_freqs;
                magnitudes = vec![0.0; frame_mags.len()];
                phases = vec![0.0; frame_phases.len()];
            }
            
            // Accumulate magnitudes and phases
            for (i, (&mag, &phase)) in frame_mags.iter().zip(frame_phases.iter()).enumerate() {
                magnitudes[i] += mag;
                phases[i] += phase;
            }
            
            // Calculate spectral features for this frame
            spectral_centroid_sum += self.calculate_spectral_centroid(&frequency_bins, &frame_mags);
            spectral_bandwidth_sum += self.calculate_spectral_bandwidth(&frequency_bins, &frame_mags);
            frame_count += 1;
        }
        
        // Average the accumulated values
        for i in 0..magnitudes.len() {
            magnitudes[i] /= frame_count as f64;
            phases[i] /= frame_count as f64;
        }
        
        let spectral_centroid = spectral_centroid_sum / frame_count as f64;
        let spectral_bandwidth = spectral_bandwidth_sum / frame_count as f64;
        
        Ok(SpectralFeatures {
            frequency_bins,
            magnitudes,
            phases,
            spectral_centroid,
            spectral_bandwidth,
        })
    }
    
    /// Frame audio for analysis
    fn frame_audio(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();
        let frame_size = self.config.fft_size;
        let hop_size = self.config.hop_size;
        
        let mut start = 0;
        while start + frame_size <= audio.len() {
            let mut frame = audio[start..start + frame_size].to_vec();
            
            // Apply Hanning window
            for (i, sample) in frame.iter_mut().enumerate() {
                *sample *= self.spectral_analyzer.window[i] as f32;
            }
            
            frames.push(frame);
            start += hop_size;
        }
        
        frames
    }
    
    /// Compute FFT for a frame
    fn compute_fft(&self, frame: &[f32]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        use rustfft::{FftPlanner, num_complex::Complex};
        
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame.len());
        
        // Convert to complex numbers
        let mut buffer: Vec<Complex<f32>> = frame.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Perform FFT
        fft.process(&mut buffer);
        
        // Extract frequency bins, magnitudes, and phases
        let sample_rate = self.config.sample_rate as f64;
        let freq_bin_width = sample_rate / frame.len() as f64;
        
        let frequency_bins: Vec<f64> = (0..buffer.len() / 2)
            .map(|i| i as f64 * freq_bin_width)
            .collect();
            
        let magnitudes: Vec<f64> = buffer[..buffer.len() / 2].iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f64)
            .collect();
            
        let phases: Vec<f64> = buffer[..buffer.len() / 2].iter()
            .map(|c| c.im.atan2(c.re) as f64)
            .collect();
        
        Ok((frequency_bins, magnitudes, phases))
    }
    
    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, frequency_bins: &[f64], magnitudes: &[f64]) -> f64 {
        let weighted_sum: f64 = frequency_bins.iter()
            .zip(magnitudes.iter())
            .map(|(freq, mag)| freq * mag)
            .sum();
            
        let magnitude_sum: f64 = magnitudes.iter().sum();
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    /// Calculate spectral bandwidth
    fn calculate_spectral_bandwidth(&self, frequency_bins: &[f64], magnitudes: &[f64]) -> f64 {
        let centroid = self.calculate_spectral_centroid(frequency_bins, magnitudes);
        let magnitude_sum: f64 = magnitudes.iter().sum();
        
        if magnitude_sum > 0.0 {
            let variance: f64 = frequency_bins.iter()
                .zip(magnitudes.iter())
                .map(|(freq, mag)| {
                    let diff = freq - centroid;
                    diff * diff * mag
                })
                .sum::<f64>() / magnitude_sum;
            variance.sqrt()
        } else {
            0.0
        }
    }

    fn analyze_temporal(&self, audio: &[f32]) -> Result<TemporalFeatures> {
        // Calculate envelope
        let envelope = self.calculate_envelope(audio);
        
        // Detect onsets
        let onsets = self.detect_onsets(&envelope);
        
        // Calculate zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(audio);
        
        // Calculate temporal centroid
        let temporal_centroid = self.calculate_temporal_centroid(&envelope);
        
        Ok(TemporalFeatures {
            onsets,
            envelope,
            zero_crossing_rate: zcr,
            temporal_centroid,
        })
    }

    fn detect_patterns(&self, audio: &[f32]) -> Result<Vec<AudioPattern>> {
        let mut patterns = Vec::new();
        
        // Simple pattern detection - detect periodic structures
        let frame_size = 1024;
        let num_frames = audio.len() / frame_size;
        
        for i in 0..(num_frames - 1) {
            let frame1_start = i * frame_size;
            let frame2_start = (i + 1) * frame_size;
            
            if frame2_start + frame_size <= audio.len() {
                let frame1 = &audio[frame1_start..frame1_start + frame_size];
                let frame2 = &audio[frame2_start..frame2_start + frame_size];
                
                // Calculate correlation
                let correlation = self.calculate_correlation(frame1, frame2);
                
                if correlation > 0.8 {
                    patterns.push(AudioPattern {
                        pattern_type: "repetitive".to_string(),
                        start_time: frame1_start as f64 / self.config.sample_rate as f64,
                        duration: frame_size as f64 / self.config.sample_rate as f64,
                        confidence: correlation,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("correlation".to_string(), correlation);
                            params
                        },
                    });
                }
            }
        }
        
        Ok(patterns)
    }

    fn reconstruct_audio(&self, spectral_features: &SpectralFeatures) -> Result<Vec<f32>> {
        self.reconstruct_from_features(spectral_features)
    }

    fn reconstruct_from_features(&self, spectral_features: &SpectralFeatures) -> Result<Vec<f32>> {
        match self.reconstructor.method {
            ReconstructionMethod::PhaseVocoder => {
                self.phase_vocoder_reconstruction(spectral_features)
            },
            ReconstructionMethod::GriffinLim => {
                self.griffin_lim_reconstruction(spectral_features)
            },
            ReconstructionMethod::Hybrid => {
                // Try Griffin-Lim first, fallback to phase vocoder
                match self.griffin_lim_reconstruction(spectral_features) {
                    Ok(audio) => Ok(audio),
                    Err(_) => self.phase_vocoder_reconstruction(spectral_features),
                }
            },
        }
    }
    
    /// Phase vocoder reconstruction
    fn phase_vocoder_reconstruction(&self, spectral_features: &SpectralFeatures) -> Result<Vec<f32>> {
        use rustfft::{FftPlanner, num_complex::Complex};
        
        let frame_size = self.config.fft_size;
        let hop_size = self.config.hop_size;
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(frame_size);
        
        // Reconstruct complex spectrum
        let mut complex_spectrum = Vec::new();
        for (i, (&magnitude, &phase)) in spectral_features.magnitudes.iter()
            .zip(spectral_features.phases.iter()).enumerate() {
            let real = magnitude * phase.cos();
            let imag = magnitude * phase.sin();
            complex_spectrum.push(Complex::new(real as f32, imag as f32));
        }
        
        // Mirror for negative frequencies
        let mut full_spectrum = complex_spectrum.clone();
        for i in (1..complex_spectrum.len() - 1).rev() {
            full_spectrum.push(Complex::new(
                complex_spectrum[i].re,
                -complex_spectrum[i].im,
            ));
        }
        
        // Ensure correct length
        full_spectrum.resize(frame_size, Complex::new(0.0, 0.0));
        
        // Perform IFFT
        let mut buffer = full_spectrum;
        ifft.process(&mut buffer);
        
        // Extract real part
        let reconstructed: Vec<f32> = buffer.iter()
            .map(|c| c.re / frame_size as f32)
            .collect();
        
        Ok(reconstructed)
    }
    
    /// Griffin-Lim reconstruction algorithm
    fn griffin_lim_reconstruction(&self, spectral_features: &SpectralFeatures) -> Result<Vec<f32>> {
        use rustfft::{FftPlanner, num_complex::Complex};
        
        let frame_size = self.config.fft_size;
        let hop_size = self.config.hop_size;
        let iterations = self.reconstructor.iterations.min(50); // Cap iterations
        
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);
        
        // Initialize with random phases
        let mut phases = spectral_features.phases.clone();
        
        // Estimate audio length
        let estimated_length = frame_size + hop_size * 10; // Rough estimate
        let mut reconstructed = vec![0.0f32; estimated_length];
        
        for _ in 0..iterations {
            // Reconstruct frames
            for (frame_idx, (&magnitude, &phase)) in spectral_features.magnitudes.iter()
                .zip(phases.iter()).enumerate() {
                
                // Create complex spectrum for this "frame" (simplified)
                let mut complex_spectrum = vec![Complex::new(0.0, 0.0); frame_size];
                
                // Only use first component for simplification
                if frame_idx < frame_size / 2 {
                    let real = magnitude * phase.cos();
                    let imag = magnitude * phase.sin();
                    complex_spectrum[frame_idx] = Complex::new(real as f32, imag as f32);
                    
                    // Mirror for negative frequencies
                    if frame_idx > 0 && frame_idx < frame_size / 2 {
                        complex_spectrum[frame_size - frame_idx] = Complex::new(
                            real as f32,
                            -(imag as f32),
                        );
                    }
                }
                
                // IFFT to time domain
                let mut frame_buffer = complex_spectrum;
                ifft.process(&mut frame_buffer);
                
                // Overlap-add to output
                let start_idx = frame_idx * hop_size;
                for (i, sample) in frame_buffer.iter().enumerate() {
                    if start_idx + i < reconstructed.len() {
                        reconstructed[start_idx + i] += sample.re / frame_size as f32;
                    }
                }
            }
            
            // Update phases based on reconstructed audio
            self.update_phases_from_audio(&reconstructed, &mut phases, &fft)?;
        }
        
        Ok(reconstructed)
    }
    
    /// Update phases from current audio estimate
    fn update_phases_from_audio(&self, audio: &[f32], phases: &mut [f64], fft: &std::sync::Arc<dyn rustfft::Fft<f32>>) -> Result<()> {
        use rustfft::num_complex::Complex;
        
        let frame_size = self.config.fft_size;
        let hop_size = self.config.hop_size;
        
        // Process overlapping frames
        let mut frame_idx = 0;
        let mut start = 0;
        
        while start + frame_size <= audio.len() && frame_idx < phases.len() {
            // Extract frame
            let mut frame: Vec<Complex<f32>> = audio[start..start + frame_size].iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let windowed = sample * self.spectral_analyzer.window[i] as f32;
                    Complex::new(windowed, 0.0)
                })
                .collect();
            
            // Forward FFT
            let mut frame_copy = frame.clone();
            fft.process(&mut frame_copy);
            
            // Update phase for this frequency bin
            if frame_copy.len() > frame_idx {
                phases[frame_idx] = frame_copy[frame_idx].im.atan2(frame_copy[frame_idx].re) as f64;
            }
            
            frame_idx += 1;
            start += hop_size;
        }
        
        Ok(())
    }

    fn calculate_fidelity(&self, original: &[f32], reconstructed: &[f32]) -> Result<f64> {
        // Ensure same length for comparison
        let min_len = original.len().min(reconstructed.len());
        let original = &original[..min_len];
        let reconstructed = &reconstructed[..min_len];
        
        // Calculate multiple fidelity metrics
        let snr = self.calculate_snr(original, reconstructed)?;
        let correlation = self.calculate_correlation(original, reconstructed);
        let spectral_distance = self.calculate_spectral_distance(original, reconstructed)?;
        
        // Composite fidelity score (0.0 to 1.0)
        let snr_normalized = (snr / 30.0).min(1.0).max(0.0); // 30 dB is excellent
        let correlation_normalized = correlation.abs();
        let spectral_distance_normalized = (1.0 - spectral_distance.min(1.0)).max(0.0);
        
        // Weighted combination
        let fidelity = 0.4 * snr_normalized + 0.4 * correlation_normalized + 0.2 * spectral_distance_normalized;
        
        Ok(fidelity)
    }
    
    /// Calculate Signal-to-Noise Ratio in dB
    fn calculate_snr(&self, original: &[f32], reconstructed: &[f32]) -> Result<f64> {
        let signal_power: f64 = original.iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>() / original.len() as f64;
            
        let noise_power: f64 = original.iter()
            .zip(reconstructed.iter())
            .map(|(&orig, &recon)| ((orig - recon) as f64).powi(2))
            .sum::<f64>() / original.len() as f64;
        
        if noise_power > 0.0 && signal_power > 0.0 {
            let snr = 10.0 * (signal_power / noise_power).log10();
            Ok(snr)
        } else if noise_power == 0.0 {
            Ok(f64::INFINITY) // Perfect reconstruction
        } else {
            Ok(0.0) // No signal
        }
    }
    
    /// Calculate spectral distance between original and reconstructed
    fn calculate_spectral_distance(&self, original: &[f32], reconstructed: &[f32]) -> Result<f64> {
        let orig_spectrum = self.compute_spectrum_for_comparison(original)?;
        let recon_spectrum = self.compute_spectrum_for_comparison(reconstructed)?;
        
        // Calculate spectral difference
        let spectral_distance = orig_spectrum.iter()
            .zip(recon_spectrum.iter())
            .map(|(&orig_mag, &recon_mag)| (orig_mag - recon_mag).powi(2))
            .sum::<f64>().sqrt();
            
        // Normalize by original spectrum magnitude
        let original_magnitude = orig_spectrum.iter()
            .map(|&x| x.powi(2))
            .sum::<f64>().sqrt();
            
        if original_magnitude > 0.0 {
            Ok(spectral_distance / original_magnitude)
        } else {
            Ok(1.0)
        }
    }
    
    /// Compute spectrum for comparison
    fn compute_spectrum_for_comparison(&self, audio: &[f32]) -> Result<Vec<f64>> {
        use rustfft::{FftPlanner, num_complex::Complex};
        
        let fft_size = self.config.fft_size.min(audio.len());
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        
        // Take the first frame for comparison
        let mut frame: Vec<Complex<f32>> = audio[..fft_size].iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window_val = if i < self.spectral_analyzer.window.len() {
                    self.spectral_analyzer.window[i] as f32
                } else {
                    1.0
                };
                Complex::new(sample * window_val, 0.0)
            })
            .collect();
        
        fft.process(&mut frame);
        
        // Extract magnitude spectrum
        let spectrum: Vec<f64> = frame[..fft_size / 2].iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f64)
            .collect();
            
        Ok(spectrum)
    }

    fn calculate_understanding_quality(&self, spectral: &SpectralFeatures, temporal: &TemporalFeatures, fidelity: f64) -> f64 {
        // Combine multiple factors for understanding quality
        let spectral_quality = if spectral.spectral_centroid > 0.0 { 0.8 } else { 0.3 };
        let temporal_quality = if temporal.onsets.len() > 0 { 0.9 } else { 0.4 };
        let reconstruction_quality = fidelity;
        
        (spectral_quality + temporal_quality + reconstruction_quality) / 3.0
    }

    // Helper methods
    fn create_hanning_window(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64;
                0.5 * (1.0 - phase.cos())
            })
            .collect()
    }

    fn calculate_envelope(&self, audio: &[f32]) -> Vec<f64> {
        let frame_size = 512;
        let mut envelope = Vec::new();
        
        for chunk in audio.chunks(frame_size) {
            let rms = (chunk.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / chunk.len() as f64).sqrt();
            envelope.push(rms);
        }
        
        envelope
    }

    fn detect_onsets(&self, envelope: &[f64]) -> Vec<f64> {
        let mut onsets = Vec::new();
        let frame_time = 512.0 / self.config.sample_rate as f64;
        
        for i in 1..envelope.len() {
            let diff = envelope[i] - envelope[i - 1];
            if diff > self.temporal_analyzer.onset_threshold {
                onsets.push(i as f64 * frame_time);
            }
        }
        
        onsets
    }

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f64 {
        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f64 / audio.len() as f64
    }

    fn calculate_temporal_centroid(&self, envelope: &[f64]) -> f64 {
        let total_energy: f64 = envelope.iter().sum();
        if total_energy == 0.0 {
            return 0.0;
        }
        
        let weighted_sum: f64 = envelope.iter()
            .enumerate()
            .map(|(i, &energy)| i as f64 * energy)
            .sum();
        
        weighted_sum / total_energy
    }

    fn calculate_correlation(&self, frame1: &[f32], frame2: &[f32]) -> f64 {
        let len = frame1.len().min(frame2.len());
        if len == 0 {
            return 0.0;
        }
        
        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for i in 0..len {
            let x = frame1[i] as f64;
            let y = frame2[i] as f64;
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        let n = len as f64;
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).abs()
        }
    }
}

impl Default for HeihachiConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            sample_rate: 44100,
            target_fidelity: 0.9,
            max_iterations: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_heihachi_creation() {
        let config = HeihachiConfig::default();
        let engine = HeihachiEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_fidelity_calculation() {
        let config = HeihachiConfig::default();
        let engine = HeihachiEngine::new(config).unwrap();
        
        let original = vec![1.0, 0.5, -0.5, -1.0];
        let perfect_copy = original.clone();
        
        let fidelity = engine.calculate_fidelity(&original, &perfect_copy).unwrap();
        assert_eq!(fidelity, 1.0);
    }

    #[test]
    fn test_window_creation() {
        let window = HeihachiEngine::create_hanning_window(512);
        assert_eq!(window.len(), 512);
        assert!((window[0] - 0.0).abs() < 1e-10);
        assert!((window[256] - 1.0).abs() < 1e-10);
    }
} 