//! Color correction utilities for better blending in delogo filter.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

/// Adjust brightness and contrast.
#[must_use]
pub fn adjust_brightness_contrast(value: u8, brightness: f32, contrast: f32) -> u8 {
    let v = value as f32;
    let adjusted = (v - 128.0) * contrast + 128.0 + brightness;
    adjusted.round().clamp(0.0, 255.0) as u8
}

/// Match histogram between two regions.
pub fn match_histogram(target: &mut [u8], reference: &[u8]) {
    if target.is_empty() || reference.is_empty() {
        return;
    }

    // Compute CDFs
    let target_cdf = compute_cdf(target);
    let reference_cdf = compute_cdf(reference);

    // Create lookup table
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let target_val = target_cdf[i];
        let mut best_match = 0;
        let mut best_diff = f32::INFINITY;

        for j in 0..256 {
            let diff = (reference_cdf[j] - target_val).abs();
            if diff < best_diff {
                best_diff = diff;
                best_match = j;
            }
        }

        lut[i] = best_match as u8;
    }

    // Apply lookup table
    for pixel in target.iter_mut() {
        *pixel = lut[*pixel as usize];
    }
}

/// Compute cumulative distribution function.
fn compute_cdf(data: &[u8]) -> [f32; 256] {
    let mut histogram = [0u32; 256];
    for &pixel in data {
        histogram[pixel as usize] += 1;
    }

    let mut cdf = [0.0f32; 256];
    let total = data.len() as f32;
    let mut sum = 0u32;

    for i in 0..256 {
        sum += histogram[i];
        cdf[i] = sum as f32 / total;
    }

    cdf
}

/// Compute mean color value.
#[must_use]
pub fn mean(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 128.0;
    }

    let sum: u32 = data.iter().map(|&x| x as u32).sum();
    sum as f32 / data.len() as f32
}

/// Compute standard deviation.
#[must_use]
pub fn std_dev(data: &[u8], mean_val: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let variance: f32 = data
        .iter()
        .map(|&x| {
            let diff = x as f32 - mean_val;
            diff * diff
        })
        .sum::<f32>()
        / data.len() as f32;

    variance.sqrt()
}
