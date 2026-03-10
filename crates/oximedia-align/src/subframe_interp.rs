#![allow(dead_code)]
//! Sub-frame interpolation for precision alignment.
//!
//! When alignment accuracy needs to be finer than a single frame,
//! this module provides interpolation techniques to estimate sub-frame
//! offsets from cross-correlation peaks and feature match residuals.

/// Interpolation method for sub-frame refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpMethod {
    /// Parabolic (quadratic) peak interpolation.
    Parabolic,
    /// Gaussian peak fitting.
    Gaussian,
    /// Sinc interpolation (band-limited).
    Sinc,
    /// Linear interpolation (simplest).
    Linear,
}

/// Configuration for sub-frame interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpConfig {
    /// Interpolation method to use.
    pub method: InterpMethod,
    /// Number of neighboring samples to consider on each side.
    pub radius: usize,
    /// Frame rate of the source material (fps).
    pub frame_rate: f64,
    /// Minimum peak height for valid interpolation.
    pub min_peak_height: f64,
}

impl Default for InterpConfig {
    fn default() -> Self {
        Self {
            method: InterpMethod::Parabolic,
            radius: 1,
            frame_rate: 30.0,
            min_peak_height: 0.1,
        }
    }
}

/// Result of sub-frame interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubFrameOffset {
    /// Integer frame offset (from peak location).
    pub frame_offset: i64,
    /// Fractional offset within the frame (-0.5..+0.5).
    pub fractional: f64,
    /// Interpolated peak value at the refined position.
    pub peak_value: f64,
    /// Confidence of the interpolation (0.0..1.0).
    pub confidence: f64,
}

impl SubFrameOffset {
    /// Create a new sub-frame offset.
    #[must_use]
    pub fn new(frame_offset: i64, fractional: f64, peak_value: f64, confidence: f64) -> Self {
        Self {
            frame_offset,
            fractional,
            peak_value,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return the total offset in frames (integer + fractional).
    #[must_use]
    pub fn total_frames(&self) -> f64 {
        self.frame_offset as f64 + self.fractional
    }

    /// Convert the total offset to seconds given a frame rate.
    #[must_use]
    pub fn to_seconds(&self, frame_rate: f64) -> f64 {
        if frame_rate > 0.0 {
            self.total_frames() / frame_rate
        } else {
            0.0
        }
    }

    /// Convert the total offset to milliseconds given a frame rate.
    #[must_use]
    pub fn to_millis(&self, frame_rate: f64) -> f64 {
        self.to_seconds(frame_rate) * 1000.0
    }
}

/// Perform parabolic (quadratic) interpolation around a peak in a correlation signal.
///
/// Given three consecutive samples y_{-1}, `y_0`, y_{+1} where `y_0` is the peak,
/// the fractional offset is: delta = (y_{-1} - y_{+1}) / (2 * (y_{-1} - 2*`y_0` + y_{+1}))
#[must_use]
pub fn parabolic_interpolation(y_minus: f64, y_center: f64, y_plus: f64) -> (f64, f64) {
    let denom = y_minus - 2.0 * y_center + y_plus;
    if denom.abs() < 1e-15 {
        return (0.0, y_center);
    }
    let delta = 0.5 * (y_minus - y_plus) / denom;
    let peak = y_center - 0.25 * (y_minus - y_plus) * delta;
    (delta.clamp(-0.5, 0.5), peak)
}

/// Perform Gaussian peak interpolation around a peak.
///
/// Uses log-domain fitting: delta = ln(y_{-1}/y_{+1}) / (2 * ln(y_{-1}*y_{+`1}/y_0^2`))
#[must_use]
pub fn gaussian_interpolation(y_minus: f64, y_center: f64, y_plus: f64) -> (f64, f64) {
    // All values must be positive for log-domain fitting
    if y_minus <= 0.0 || y_center <= 0.0 || y_plus <= 0.0 {
        return parabolic_interpolation(y_minus, y_center, y_plus);
    }
    let ln_minus = y_minus.ln();
    let ln_center = y_center.ln();
    let ln_plus = y_plus.ln();

    let denom = 2.0 * (ln_minus - 2.0 * ln_center + ln_plus);
    if denom.abs() < 1e-15 {
        return (0.0, y_center);
    }
    let delta = (ln_minus - ln_plus) / denom;
    let clamped = delta.clamp(-0.5, 0.5);
    let peak = (ln_center
        - 0.125 * (ln_minus - ln_plus).powi(2) / (ln_minus - 2.0 * ln_center + ln_plus))
        .exp();
    (clamped, peak)
}

/// Perform windowed sinc interpolation at a fractional position.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn sinc_interpolation(
    samples: &[f64],
    center_idx: usize,
    fractional: f64,
    radius: usize,
) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;

    let start = center_idx.saturating_sub(radius);
    let end = (center_idx + radius + 1).min(samples.len());

    for i in start..end {
        let x = i as f64 - center_idx as f64 - fractional;
        let w = sinc(x) * hann_window(x, radius as f64);
        sum += samples[i] * w;
        weight_sum += w;
    }

    if weight_sum.abs() > 1e-15 {
        sum / weight_sum
    } else {
        samples.get(center_idx).copied().unwrap_or(0.0)
    }
}

/// Normalized sinc function: sin(pi*x) / (pi*x).
#[must_use]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-15 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Hann window function for windowed sinc.
#[must_use]
fn hann_window(x: f64, radius: f64) -> f64 {
    if x.abs() > radius {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * x / radius).cos())
    }
}

/// Find the peak in a correlation signal and refine with sub-frame interpolation.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn find_subframe_peak(correlation: &[f64], config: &InterpConfig) -> Option<SubFrameOffset> {
    if correlation.len() < 3 {
        return None;
    }

    // Find the integer peak
    let (peak_idx, &peak_val) = correlation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    if peak_val < config.min_peak_height {
        return None;
    }

    // Cannot interpolate at boundaries
    if peak_idx == 0 || peak_idx == correlation.len() - 1 {
        let conf = compute_peak_confidence(correlation, peak_idx);
        return Some(SubFrameOffset::new(peak_idx as i64, 0.0, peak_val, conf));
    }

    let y_minus = correlation[peak_idx - 1];
    let y_center = correlation[peak_idx];
    let y_plus = correlation[peak_idx + 1];

    let (delta, interp_peak) = match config.method {
        InterpMethod::Parabolic => parabolic_interpolation(y_minus, y_center, y_plus),
        InterpMethod::Gaussian => gaussian_interpolation(y_minus, y_center, y_plus),
        InterpMethod::Sinc => {
            let (para_delta, _) = parabolic_interpolation(y_minus, y_center, y_plus);
            let val = sinc_interpolation(correlation, peak_idx, para_delta, config.radius);
            (para_delta, val)
        }
        InterpMethod::Linear => {
            // Simple linear between the two highest neighbors
            if y_minus > y_plus {
                let d = -0.5 * (y_center - y_minus) / (y_center - y_minus + f64::EPSILON);
                (d.clamp(-0.5, 0.0), y_center)
            } else {
                let d = 0.5 * (y_center - y_plus) / (y_center - y_plus + f64::EPSILON);
                (d.clamp(0.0, 0.5), y_center)
            }
        }
    };

    let confidence = compute_peak_confidence(correlation, peak_idx);
    Some(SubFrameOffset::new(
        peak_idx as i64,
        delta,
        interp_peak,
        confidence,
    ))
}

/// Compute a confidence measure for the peak (ratio of peak to second highest).
fn compute_peak_confidence(correlation: &[f64], peak_idx: usize) -> f64 {
    let peak_val = correlation[peak_idx];
    if peak_val.abs() < 1e-15 {
        return 0.0;
    }

    // Find the second highest local maximum
    let mut second_max = 0.0_f64;
    for (i, &val) in correlation.iter().enumerate() {
        if i != peak_idx && val > second_max {
            // Only consider if it's a local peak or boundary
            let is_local = i == 0
                || i == correlation.len() - 1
                || (val >= correlation[i.saturating_sub(1)]
                    && val >= correlation[(i + 1).min(correlation.len() - 1)]);
            if is_local {
                second_max = val;
            }
        }
    }

    if second_max.abs() < 1e-15 {
        return 1.0;
    }
    let ratio = 1.0 - (second_max / peak_val);
    ratio.clamp(0.0, 1.0)
}

/// Linear interpolation between two values.
#[must_use]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Cubic Hermite interpolation for smooth sub-frame values.
#[must_use]
pub fn cubic_hermite(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c = -0.5 * y0 + 0.5 * y2;
    let d = y1;
    ((a * t + b) * t + c) * t + d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parabolic_symmetric_peak() {
        // Symmetric peak => delta should be ~0
        let (delta, peak) = parabolic_interpolation(0.5, 1.0, 0.5);
        assert!(delta.abs() < 1e-10);
        assert!((peak - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parabolic_asymmetric_peak() {
        // y_minus > y_plus => peak shifts toward minus direction (negative delta)
        let (delta, _) = parabolic_interpolation(0.8, 1.0, 0.4);
        assert!(delta < 0.0); // peak is between center and y_minus
    }

    #[test]
    fn test_parabolic_flat() {
        let (delta, peak) = parabolic_interpolation(1.0, 1.0, 1.0);
        assert!((delta).abs() < 1e-10);
        assert!((peak - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_symmetric() {
        let (delta, _) = gaussian_interpolation(0.5, 1.0, 0.5);
        assert!(delta.abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_fallback_on_negative() {
        let (delta, _) = gaussian_interpolation(-0.5, 1.0, 0.5);
        // Should fall back to parabolic
        assert!(delta.abs() < 1.0);
    }

    #[test]
    fn test_sinc_at_integer() {
        let samples = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let val = sinc_interpolation(&samples, 2, 0.0, 2);
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sinc_empty() {
        let val = sinc_interpolation(&[], 0, 0.0, 2);
        assert!((val).abs() < f64::EPSILON);
    }

    #[test]
    fn test_subframe_offset_total() {
        let offset = SubFrameOffset::new(10, 0.25, 0.9, 0.95);
        assert!((offset.total_frames() - 10.25).abs() < 1e-10);
    }

    #[test]
    fn test_subframe_offset_to_seconds() {
        let offset = SubFrameOffset::new(30, 0.0, 1.0, 0.9);
        assert!((offset.to_seconds(30.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_subframe_offset_to_millis() {
        let offset = SubFrameOffset::new(30, 0.0, 1.0, 0.9);
        assert!((offset.to_millis(30.0) - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_subframe_offset_zero_fps() {
        let offset = SubFrameOffset::new(10, 0.5, 1.0, 0.9);
        assert!((offset.to_seconds(0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_subframe_peak_parabolic() {
        let correlation = vec![0.1, 0.3, 0.5, 0.9, 0.5, 0.3, 0.1];
        let config = InterpConfig {
            method: InterpMethod::Parabolic,
            ..InterpConfig::default()
        };
        let result = find_subframe_peak(&correlation, &config);
        assert!(result.is_some());
        let r = result.expect("r should be valid");
        assert_eq!(r.frame_offset, 3);
        assert!(r.fractional.abs() < 0.5);
    }

    #[test]
    fn test_find_subframe_peak_gaussian() {
        let correlation = vec![0.2, 0.5, 1.0, 0.5, 0.2];
        let config = InterpConfig {
            method: InterpMethod::Gaussian,
            ..InterpConfig::default()
        };
        let result = find_subframe_peak(&correlation, &config);
        assert!(result.is_some());
        let r = result.expect("r should be valid");
        assert_eq!(r.frame_offset, 2);
    }

    #[test]
    fn test_find_subframe_peak_too_short() {
        let correlation = vec![0.5, 0.9];
        let config = InterpConfig::default();
        assert!(find_subframe_peak(&correlation, &config).is_none());
    }

    #[test]
    fn test_find_subframe_peak_below_threshold() {
        let correlation = vec![0.01, 0.02, 0.01];
        let config = InterpConfig {
            min_peak_height: 0.5,
            ..InterpConfig::default()
        };
        assert!(find_subframe_peak(&correlation, &config).is_none());
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((lerp(2.0, 4.0, 0.25) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_hermite_at_endpoints() {
        let v = cubic_hermite(0.0, 1.0, 2.0, 3.0, 0.0);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_subframe_confidence_clamped() {
        let offset = SubFrameOffset::new(0, 0.0, 1.0, 1.5);
        assert!((offset.confidence - 1.0).abs() < f64::EPSILON);

        let offset2 = SubFrameOffset::new(0, 0.0, 1.0, -0.5);
        assert!((offset2.confidence).abs() < f64::EPSILON);
    }
}
