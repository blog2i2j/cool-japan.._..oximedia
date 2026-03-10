//! Python bindings for audio loudness normalization and dynamics processing.
//!
//! Wraps `oximedia-normalize` for EBU R128 compliant loudness normalization,
//! and provides pure-Rust implementations of a feed-forward compressor and a
//! peak limiter.

use pyo3::prelude::*;

use oximedia_metering::Standard;
use oximedia_normalize::{Normalizer, NormalizerConfig, ProcessingMode};

// ---------------------------------------------------------------------------
// Loudness normalization (EBU R128 / ATSC)
// ---------------------------------------------------------------------------

/// Normalize audio loudness using two-pass EBU R128 normalization.
///
/// Pass 1 analyses the integrated loudness; Pass 2 applies the computed gain
/// to bring the signal to the standard's target level.
///
/// # Arguments
/// * `samples`     - Interleaved float32 audio samples
/// * `sample_rate` - Sample rate in Hz (e.g., 48000.0)
/// * `channels`    - Number of interleaved channels (default: 2)
/// * `standard`    - Loudness standard: `"ebu-r128"` (default), `"atsc"`,
///                   `"spotify"`, `"youtube"`, `"apple-music"`, `"netflix"`
/// * `max_gain_db` - Maximum gain cap in dB (default: 20.0)
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, channels=2, standard="ebu-r128", max_gain_db=20.0))]
pub fn normalize_loudness(
    samples: Vec<f32>,
    sample_rate: f64,
    channels: usize,
    standard: &str,
    max_gain_db: f64,
) -> PyResult<Vec<f32>> {
    let std = parse_standard(standard)?;

    let mut config = NormalizerConfig::new(std, sample_rate, channels);
    config.processing_mode = ProcessingMode::TwoPass;
    config.max_gain_db = max_gain_db;

    let mut normalizer = Normalizer::new(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create normalizer: {}", e))
    })?;

    // Pass 1: analysis
    normalizer.analyze_f32(&samples);

    // Pass 2: apply gain
    let mut output = vec![0.0f32; samples.len()];
    normalizer.process_f32(&samples, &mut output).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Normalization processing failed: {}", e))
    })?;

    Ok(output)
}

/// Parse a standard string into the `oximedia_metering::Standard` enum.
fn parse_standard(s: &str) -> PyResult<Standard> {
    match s.to_lowercase().as_str() {
        "ebu-r128" | "ebu_r128" | "ebur128" => Ok(Standard::EbuR128),
        "atsc" | "atsc-a85" => Ok(Standard::AtscA85),
        "spotify" => Ok(Standard::Spotify),
        "youtube" => Ok(Standard::YouTube),
        "apple-music" | "apple_music" | "apple" => Ok(Standard::AppleMusic),
        "netflix" => Ok(Standard::Netflix),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown loudness standard '{}'. Use: ebu-r128, atsc, spotify, youtube, apple-music, netflix",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dynamic range compressor
// ---------------------------------------------------------------------------

/// Apply dynamic range compression (feed-forward, RMS-based).
///
/// Reduces the dynamic range of the signal by attenuating samples that
/// exceed `threshold_db` by `ratio`:1.
///
/// # Arguments
/// * `samples`      - Interleaved float32 samples
/// * `sample_rate`  - Sample rate in Hz
/// * `threshold_db` - Gain reduction threshold (default: -20.0 dBFS)
/// * `ratio`        - Compression ratio (default: 4.0)
/// * `attack_ms`    - Attack time in milliseconds (default: 5.0)
/// * `release_ms`   - Release time in milliseconds (default: 50.0)
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=50.0))]
pub fn apply_compressor(
    samples: Vec<f32>,
    sample_rate: f32,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
) -> PyResult<Vec<f32>> {
    if ratio < 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Compression ratio must be >= 1.0",
        ));
    }
    if sample_rate <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sample rate must be positive",
        ));
    }

    let threshold_linear = db_to_linear(threshold_db);
    let attack_coeff = time_to_coeff(attack_ms * 0.001, sample_rate);
    let release_coeff = time_to_coeff(release_ms * 0.001, sample_rate);

    let mut envelope = 0.0f32;
    let mut output = Vec::with_capacity(samples.len());

    for &sample in &samples {
        let abs_sample = sample.abs();

        // Envelope follower
        if abs_sample > envelope {
            envelope = attack_coeff * envelope + (1.0 - attack_coeff) * abs_sample;
        } else {
            envelope = release_coeff * envelope + (1.0 - release_coeff) * abs_sample;
        }

        // Gain computation
        let gain = if envelope > threshold_linear && envelope > 1e-10 {
            let overshoot_db = linear_to_db(envelope) - threshold_db;
            let gain_reduction_db = overshoot_db * (1.0 - 1.0 / ratio);
            db_to_linear(-gain_reduction_db)
        } else {
            1.0
        };

        output.push(sample * gain);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Peak limiter
// ---------------------------------------------------------------------------

/// Apply a true-peak limiter (brick-wall ceiling) to audio samples.
///
/// Clamps all samples so that their absolute value does not exceed the
/// linear equivalent of `ceiling_db`.
///
/// # Arguments
/// * `samples`     - Float32 audio samples (mono or interleaved)
/// * `ceiling_db`  - Peak ceiling in dBFS (default: -1.0)
#[pyfunction]
#[pyo3(signature = (samples, ceiling_db=-1.0))]
pub fn apply_limiter(samples: Vec<f32>, ceiling_db: f32) -> PyResult<Vec<f32>> {
    let ceiling_linear = db_to_linear(ceiling_db);
    let output = samples
        .into_iter()
        .map(|s| s.clamp(-ceiling_linear, ceiling_linear))
        .collect();
    Ok(output)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert dBFS to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear amplitude to dBFS.
#[inline]
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        return -120.0;
    }
    20.0 * linear.log10()
}

/// Convert a time constant (seconds) to an IIR coefficient for sample-by-sample
/// envelope following: `y[n] = coeff * y[n-1] + (1 - coeff) * x[n]`.
#[inline]
fn time_to_coeff(time_s: f32, sample_rate: f32) -> f32 {
    if time_s <= 0.0 {
        return 0.0;
    }
    (-1.0 / (time_s * sample_rate)).exp()
}
