//! Film grain pattern analysis.
//!
//! Analyzes video frames to identify and characterize film grain patterns,
//! distinguishing them from digital noise.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Grain characteristics map.
#[derive(Clone, Debug)]
pub struct GrainMap {
    /// Width of the grain map.
    pub width: usize,
    /// Height of the grain map.
    pub height: usize,
    /// Grain strength at each pixel (0.0 = no grain, 1.0 = strong grain).
    pub strength: Vec<f32>,
    /// Grain size characteristic.
    pub average_grain_size: f32,
    /// Grain distribution pattern.
    pub pattern_type: GrainPattern,
}

/// Type of grain pattern detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrainPattern {
    /// No significant grain detected.
    None,
    /// Fine grain (typical of high-speed film).
    Fine,
    /// Medium grain (typical of standard film).
    Medium,
    /// Coarse grain (typical of low-speed or pushed film).
    Coarse,
    /// Digital noise (not film grain).
    DigitalNoise,
}

/// Analyze film grain in a video frame.
///
/// Detects and characterizes grain patterns to enable grain-preserving
/// denoising.
///
/// # Arguments
/// * `frame` - Input video frame
///
/// # Returns
/// Grain characteristics map
pub fn analyze_grain(frame: &VideoFrame) -> DenoiseResult<GrainMap> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0]; // Use luma plane
    let (width, height) = frame.plane_dimensions(0);

    // Compute high-frequency component (potential grain)
    let high_freq = extract_high_frequency(
        plane.data.as_ref(),
        width as usize,
        height as usize,
        plane.stride,
    );

    // Analyze grain characteristics
    let average_grain_size = estimate_grain_size(&high_freq, width as usize, height as usize);
    let pattern_type = classify_grain_pattern(&high_freq, average_grain_size);

    // Create grain strength map
    let strength =
        compute_grain_strength_map(&high_freq, width as usize, height as usize, pattern_type);

    Ok(GrainMap {
        width: width as usize,
        height: height as usize,
        strength,
        average_grain_size,
        pattern_type,
    })
}

/// Extract high-frequency component using high-pass filter.
fn extract_high_frequency(data: &[u8], width: usize, height: usize, stride: usize) -> Vec<f32> {
    let mut high_freq = vec![0.0f32; width * height];

    // Apply Laplacian high-pass filter
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = y * stride + x;
            let center = f32::from(data[idx]);

            let laplacian = 4.0 * center
                - f32::from(data[idx - 1])
                - f32::from(data[idx + 1])
                - f32::from(data[idx - stride])
                - f32::from(data[idx + stride]);

            high_freq[y * width + x] = laplacian.abs();
        }
    }

    high_freq
}

/// Estimate average grain size from high-frequency component.
fn estimate_grain_size(high_freq: &[f32], width: usize, height: usize) -> f32 {
    // Compute autocorrelation to estimate grain size
    let mut autocorr_sum = 0.0f32;
    let mut count = 0;
    let max_lag = 5;

    for y in max_lag..(height - max_lag) {
        for x in max_lag..(width - max_lag) {
            let center = high_freq[y * width + x];

            for lag in 1..=max_lag {
                let neighbor = high_freq[y * width + (x + lag)];
                autocorr_sum += center * neighbor;
                count += 1;
            }
        }
    }

    let avg_autocorr = if count > 0 {
        autocorr_sum / count as f32
    } else {
        0.0
    };

    // Grain size inversely related to autocorrelation
    (1.0 / (avg_autocorr + 0.1)).clamp(1.0, 10.0)
}

/// Classify grain pattern type.
fn classify_grain_pattern(high_freq: &[f32], grain_size: f32) -> GrainPattern {
    // Compute statistics
    let sum: f32 = high_freq.iter().sum();
    let avg = sum / high_freq.len() as f32;

    let variance: f32 = high_freq
        .iter()
        .map(|&x| {
            let diff = x - avg;
            diff * diff
        })
        .sum::<f32>()
        / high_freq.len() as f32;

    let std_dev = variance.sqrt();

    // Classify based on grain size and variance
    if avg < 2.0 && std_dev < 3.0 {
        GrainPattern::None
    } else if grain_size < 2.0 {
        GrainPattern::Fine
    } else if grain_size < 5.0 {
        GrainPattern::Medium
    } else if std_dev > 10.0 {
        GrainPattern::DigitalNoise
    } else {
        GrainPattern::Coarse
    }
}

/// Compute grain strength map.
fn compute_grain_strength_map(
    high_freq: &[f32],
    width: usize,
    height: usize,
    pattern_type: GrainPattern,
) -> Vec<f32> {
    let mut strength_map = vec![0.0f32; width * height];

    // Normalize high-frequency component to strength values
    let max_hf = high_freq.iter().copied().fold(0.0f32, f32::max);

    if max_hf > 0.0 {
        for i in 0..strength_map.len() {
            let normalized = high_freq[i] / max_hf;

            // Adjust based on pattern type
            strength_map[i] = match pattern_type {
                GrainPattern::None => 0.0,
                GrainPattern::Fine => normalized * 0.3,
                GrainPattern::Medium => normalized * 0.5,
                GrainPattern::Coarse => normalized * 0.7,
                GrainPattern::DigitalNoise => 0.0, // Don't preserve digital noise
            };
        }
    }

    strength_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_analyze_grain() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = analyze_grain(&frame);
        assert!(result.is_ok());

        let grain_map = result.expect("grain_map should be valid");
        assert_eq!(grain_map.width, 64);
        assert_eq!(grain_map.height, 64);
        assert_eq!(grain_map.strength.len(), 64 * 64);
    }

    #[test]
    fn test_grain_pattern_classification() {
        let high_freq = vec![0.5f32; 100];
        let pattern = classify_grain_pattern(&high_freq, 3.0);
        assert!(matches!(
            pattern,
            GrainPattern::None
                | GrainPattern::Fine
                | GrainPattern::Medium
                | GrainPattern::Coarse
                | GrainPattern::DigitalNoise
        ));
    }

    #[test]
    fn test_grain_size_estimation() {
        let high_freq = vec![1.0f32; 64 * 64];
        let size = estimate_grain_size(&high_freq, 64, 64);
        assert!(size > 0.0);
    }

    #[test]
    fn test_high_frequency_extraction() {
        let data = vec![128u8; 64 * 64];
        let high_freq = extract_high_frequency(&data, 64, 64, 64);
        assert_eq!(high_freq.len(), 64 * 64);
    }
}
