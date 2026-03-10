/// Utility functions for noise analysis.
use super::noise_stats::estimate_plane_noise;

/// Estimate noise level in a frame.
#[must_use]
pub fn estimate_noise_level(data: &[u8], width: u32, height: u32) -> f32 {
    estimate_plane_noise(data, width, height)
}

/// Compute local variance map.
#[must_use]
pub fn variance_map(data: &[u8], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let mut variance = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let (_, var) = compute_local_stats(data, x, y, width, height, radius);
            variance[(y * width + x) as usize] = var;
        }
    }

    variance
}

fn compute_local_stats(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    radius: u32,
) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut count = 0;

    let y_min = y.saturating_sub(radius);
    let y_max = (y + radius + 1).min(height);
    let x_min = x.saturating_sub(radius);
    let x_max = (x + radius + 1).min(width);

    for ny in y_min..y_max {
        for nx in x_min..x_max {
            let nidx = (ny * width + nx) as usize;
            let val = data.get(nidx).copied().unwrap_or(128) as f32;
            sum += val;
            sq_sum += val * val;
            count += 1;
        }
    }

    if count > 0 {
        let mean = sum / count as f32;
        let variance = (sq_sum / count as f32) - (mean * mean);
        (mean, variance)
    } else {
        (128.0, 0.0)
    }
}

/// Classify noise type based on characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    /// Gaussian white noise.
    Gaussian,
    /// Salt and pepper (impulse) noise.
    Impulse,
    /// Film grain noise.
    FilmGrain,
    /// Compression artifacts.
    Compression,
    /// Unknown or mixed noise.
    Unknown,
}

/// Classify the type of noise in a frame.
#[must_use]
pub fn classify_noise(data: &[u8], width: u32, height: u32) -> NoiseType {
    let noise_sigma = estimate_noise_level(data, width, height);

    if noise_sigma < 5.0 {
        return NoiseType::Compression;
    }

    let mut histogram = [0u32; 256];
    for &val in data {
        histogram[val as usize] += 1;
    }

    let total = data.len() as f32;
    let extreme_ratio = (histogram[0] + histogram[255]) as f32 / total;

    if extreme_ratio > 0.05 {
        NoiseType::Impulse
    } else if noise_sigma > 20.0 {
        NoiseType::FilmGrain
    } else if noise_sigma > 5.0 && noise_sigma < 20.0 {
        NoiseType::Gaussian
    } else {
        NoiseType::Unknown
    }
}

/// Get recommended config for detected noise type.
#[must_use]
pub fn recommend_config(noise_type: NoiseType) -> super::DenoiseConfig {
    match noise_type {
        NoiseType::Gaussian => super::presets::medium(),
        NoiseType::Impulse => super::DenoiseConfig::new()
            .with_method(super::DenoiseMethod::Median)
            .with_strength(0.8)
            .with_spatial_radius(3),
        NoiseType::FilmGrain => super::presets::strong(),
        NoiseType::Compression => super::DenoiseConfig::new()
            .with_method(super::DenoiseMethod::Bilateral)
            .with_strength(0.5)
            .with_spatial_radius(3),
        NoiseType::Unknown => super::presets::medium(),
    }
}
