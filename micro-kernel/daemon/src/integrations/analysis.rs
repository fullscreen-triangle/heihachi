//! Measurement rungs over rendered audio.
//!
//! Each measurement carries the resolution at which it was obtained, and
//! that resolution is derived from the instrument rather than asserted: the
//! spectral floor is the bin width `sample_rate / window`, and the level
//! floor is the quantisation step of the source format. A measurement whose
//! floor could not be established is not emitted.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// One measured quantity, with the resolution at which it was obtained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub channel: String,
    pub value: f64,
    pub floor: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSummary {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
    pub duration_seconds: f64,
    pub measurements: Vec<Measurement>,
    /// Set when the file could not be decoded.
    pub note: Option<String>,
}

const WINDOW: usize = 4096;

/// Decode a WAV render and measure it.
///
/// Only WAV is decoded here. Other formats yield a summary carrying a note,
/// which is a legal value: the rung ran and drew no measurements.
pub fn analyse(path: &Path) -> AudioSummary {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext != "wav" {
        return AudioSummary {
            sample_rate: 0,
            channels: 0,
            frames: 0,
            duration_seconds: 0.0,
            measurements: Vec::new(),
            note: Some(format!(
                "{ext} is not decoded by this build; render WAV for measurement"
            )),
        };
    }

    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            return AudioSummary {
                sample_rate: 0,
                channels: 0,
                frames: 0,
                duration_seconds: 0.0,
                measurements: Vec::new(),
                note: Some(format!("render unreadable: {e}")),
            }
        }
    };

    let spec = reader.spec();
    let bits = spec.bits_per_sample.max(1);
    // the level floor is the quantisation step of the source format
    let level_floor = match spec.sample_format {
        hound::SampleFormat::Float => 1e-7,
        hound::SampleFormat::Int => 1.0 / (1i64 << (bits - 1)) as f64,
    };

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .map(f64::from)
            .collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| f64::from(s) / scale)
                .collect()
        }
    };

    if samples.is_empty() {
        return AudioSummary {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            frames: 0,
            duration_seconds: 0.0,
            measurements: Vec::new(),
            note: Some("render decoded to no samples".into()),
        };
    }

    let channels = spec.channels.max(1) as usize;
    let frames = samples.len() / channels;
    let mono: Vec<f64> = samples
        .chunks(channels)
        .map(|f| f.iter().sum::<f64>() / channels as f64)
        .collect();

    let mut measurements = Vec::new();

    // peak and rms, at the format's quantisation step
    let peak = mono.iter().fold(0.0_f64, |a, s| a.max(s.abs()));
    let rms = (mono.iter().map(|s| s * s).sum::<f64>() / mono.len() as f64).sqrt();
    let db = |x: f64| if x > 0.0 { 20.0 * x.log10() } else { -144.0 };
    // a floor in dB at the quantisation step, evaluated near the signal
    let level_floor_db = (20.0 * (1.0 + level_floor / peak.max(level_floor)).log10())
        .max(1e-4);

    measurements.push(Measurement {
        channel: "level.peak".into(),
        value: db(peak),
        floor: level_floor_db,
        unit: "dBFS".into(),
    });
    measurements.push(Measurement {
        channel: "level.rms".into(),
        value: db(rms),
        floor: level_floor_db,
        unit: "dBFS".into(),
    });
    if rms > 0.0 {
        measurements.push(Measurement {
            channel: "level.crest".into(),
            value: db(peak) - db(rms),
            floor: 2.0 * level_floor_db,
            unit: "dB".into(),
        });
    }

    // spectral centroid, at the bin width of the analysis window
    if mono.len() >= WINDOW {
        let bin_width = spec.sample_rate as f64 / WINDOW as f64;
        if let Some(centroid) = spectral_centroid(&mono, spec.sample_rate) {
            measurements.push(Measurement {
                channel: "spectrum.centroid".into(),
                value: centroid,
                floor: bin_width,
                unit: "Hz".into(),
            });
        }
        if let Some(ratio) = low_band_ratio(&mono, spec.sample_rate) {
            measurements.push(Measurement {
                channel: "spectrum.low_ratio".into(),
                value: ratio,
                // one bin out of the summed band, as a proportion
                floor: 1.0 / (WINDOW as f64 / 2.0),
                unit: "".into(),
            });
        }
    }

    // stereo width, when there are two channels to compare
    if channels == 2 {
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for f in samples.chunks(2) {
            l.push(f[0]);
            r.push(f.get(1).copied().unwrap_or(0.0));
        }
        if let Some(corr) = correlation(&l, &r) {
            measurements.push(Measurement {
                channel: "stereo.correlation".into(),
                value: corr,
                floor: 1.0 / (mono.len() as f64).sqrt(),
                unit: "".into(),
            });
        }
    }

    AudioSummary {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        frames,
        duration_seconds: frames as f64 / spec.sample_rate.max(1) as f64,
        measurements,
        note: None,
    }
}

fn hann(n: usize, len: usize) -> f64 {
    let x = std::f64::consts::PI * n as f64 / (len - 1) as f64;
    x.sin().powi(2)
}

/// Average magnitude spectrum over non-overlapping windows.
fn spectrum(samples: &[f64]) -> Option<Vec<f64>> {
    use rustfft::{num_complex::Complex, FftPlanner};
    if samples.len() < WINDOW {
        return None;
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW);

    let mut acc = vec![0.0_f64; WINDOW / 2];
    let mut windows = 0usize;
    // cap the work so a long render does not stall the daemon
    for chunk in samples.chunks_exact(WINDOW).take(64) {
        let mut buf: Vec<Complex<f64>> = chunk
            .iter()
            .enumerate()
            .map(|(n, s)| Complex::new(s * hann(n, WINDOW), 0.0))
            .collect();
        fft.process(&mut buf);
        for (i, slot) in acc.iter_mut().enumerate() {
            *slot += buf[i].norm();
        }
        windows += 1;
    }
    if windows == 0 {
        return None;
    }
    for slot in acc.iter_mut() {
        *slot /= windows as f64;
    }
    Some(acc)
}

fn spectral_centroid(samples: &[f64], sample_rate: u32) -> Option<f64> {
    let spec = spectrum(samples)?;
    let bin_width = sample_rate as f64 / WINDOW as f64;
    let total: f64 = spec.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let weighted: f64 = spec
        .iter()
        .enumerate()
        .map(|(i, m)| i as f64 * bin_width * m)
        .sum();
    Some(weighted / total)
}

/// Proportion of spectral energy below 200 Hz -- the band a bass lives in.
fn low_band_ratio(samples: &[f64], sample_rate: u32) -> Option<f64> {
    let spec = spectrum(samples)?;
    let bin_width = sample_rate as f64 / WINDOW as f64;
    let cutoff = (200.0 / bin_width).round() as usize;
    let total: f64 = spec.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let low: f64 = spec.iter().take(cutoff.min(spec.len())).sum();
    Some(low / total)
}

fn correlation(l: &[f64], r: &[f64]) -> Option<f64> {
    let n = l.len().min(r.len());
    if n == 0 {
        return None;
    }
    let (lm, rm) = (
        l[..n].iter().sum::<f64>() / n as f64,
        r[..n].iter().sum::<f64>() / n as f64,
    );
    let mut num = 0.0;
    let (mut dl, mut dr) = (0.0, 0.0);
    for i in 0..n {
        let (a, b) = (l[i] - lm, r[i] - rm);
        num += a * b;
        dl += a * a;
        dr += b * b;
    }
    let den = (dl * dr).sqrt();
    if den <= 0.0 {
        return None;
    }
    Some(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sine(path: &Path, freq: f64, seconds: f64, rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let n = (seconds * rate as f64) as usize;
        for i in 0..n {
            let t = i as f64 / rate as f64;
            let s = (2.0 * std::f64::consts::PI * freq * t).sin() * 0.5;
            w.write_sample((s * f64::from(i16::MAX)) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn a_sine_is_measured_with_instrument_derived_floors() {
        let dir = std::env::temp_dir().join("hk_analysis_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("sine.wav");
        write_sine(&p, 100.0, 0.5, 44100);

        let s = analyse(&p);
        assert!(s.note.is_none(), "note was {:?}", s.note);
        assert_eq!(s.sample_rate, 44100);
        assert!(s.measurements.iter().all(|m| m.floor > 0.0));

        let centroid = s
            .measurements
            .iter()
            .find(|m| m.channel == "spectrum.centroid")
            .expect("centroid");
        // a 100 Hz sine should sit low; the floor is the bin width
        assert!(centroid.value < 2000.0, "centroid {}", centroid.value);
        assert!((centroid.floor - 44100.0 / WINDOW as f64).abs() < 1e-9);

        let low = s
            .measurements
            .iter()
            .find(|m| m.channel == "spectrum.low_ratio")
            .expect("low ratio");
        assert!(low.value > 0.5, "low ratio {}", low.value);

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_undecodable_file_yields_a_note_not_a_panic() {
        let s = analyse(Path::new("nothing_here.wav"));
        assert!(s.note.is_some());
        assert!(s.measurements.is_empty());

        let s = analyse(Path::new("bounce.mp3"));
        assert!(s.note.unwrap().contains("not decoded"));
    }
}
