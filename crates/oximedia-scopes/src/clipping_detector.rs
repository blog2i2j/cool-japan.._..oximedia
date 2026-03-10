#![allow(dead_code)]
//! Video clipping and crushing detection.
//!
//! Detects pixels that are clipped (at maximum value) or crushed
//! (at minimum value) in each RGB channel. This is essential for
//! exposure evaluation and broadcast-safe compliance checking.

/// Clipping report for a single color channel.
#[derive(Debug, Clone, Copy)]
pub struct ChannelClipInfo {
    /// Number of pixels at the maximum value (clipped highlights).
    pub clipped_count: u64,
    /// Number of pixels at the minimum value (crushed shadows).
    pub crushed_count: u64,
    /// Fraction of pixels that are clipped (0.0..=1.0).
    pub clip_ratio: f64,
    /// Fraction of pixels that are crushed (0.0..=1.0).
    pub crush_ratio: f64,
}

/// Full clipping report for an RGB frame.
#[derive(Debug, Clone)]
pub struct ClippingReport {
    /// Red channel clipping info.
    pub red: ChannelClipInfo,
    /// Green channel clipping info.
    pub green: ChannelClipInfo,
    /// Blue channel clipping info.
    pub blue: ChannelClipInfo,
    /// Number of pixels where ALL channels are clipped (pure white).
    pub white_clip_count: u64,
    /// Number of pixels where ALL channels are crushed (pure black).
    pub black_crush_count: u64,
    /// Total number of pixels analyzed.
    pub total_pixels: u64,
    /// Overall severity: fraction of pixels with any clipping or crushing.
    pub overall_severity: f64,
}

/// Configuration for clipping detection.
#[derive(Debug, Clone)]
pub struct ClippingConfig {
    /// Upper threshold for clipping detection (inclusive). Default: 255.
    pub clip_threshold: u8,
    /// Lower threshold for crushing detection (inclusive). Default: 0.
    pub crush_threshold: u8,
    /// Whether to use broadcast-safe range (16-235) instead of full range (0-255).
    pub broadcast_safe: bool,
}

impl Default for ClippingConfig {
    fn default() -> Self {
        Self {
            clip_threshold: 255,
            crush_threshold: 0,
            broadcast_safe: false,
        }
    }
}

impl ClippingConfig {
    /// Returns a configuration for broadcast-safe (IRE 0-100) detection.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            clip_threshold: 235,
            crush_threshold: 16,
            broadcast_safe: true,
        }
    }

    /// Returns the effective clip threshold.
    #[must_use]
    pub fn effective_clip(&self) -> u8 {
        self.clip_threshold
    }

    /// Returns the effective crush threshold.
    #[must_use]
    pub fn effective_crush(&self) -> u8 {
        self.crush_threshold
    }
}

/// Detects clipping and crushing in an RGB24 video frame.
///
/// # Arguments
///
/// * `frame` - RGB24 pixel data (3 bytes per pixel, row-major).
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `config` - Detection configuration.
///
/// # Returns
///
/// A `ClippingReport`, or `None` if the frame data is invalid.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn detect_clipping(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ClippingConfig,
) -> Option<ClippingReport> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 || frame.len() < pixel_count * 3 {
        return None;
    }

    let clip_hi = config.effective_clip();
    let clip_lo = config.effective_crush();

    let mut r_clip: u64 = 0;
    let mut r_crush: u64 = 0;
    let mut g_clip: u64 = 0;
    let mut g_crush: u64 = 0;
    let mut b_clip: u64 = 0;
    let mut b_crush: u64 = 0;
    let mut white_clip: u64 = 0;
    let mut black_crush: u64 = 0;

    for i in 0..pixel_count {
        let base = i * 3;
        let r = frame[base];
        let g = frame[base + 1];
        let b = frame[base + 2];

        let r_is_clip = r >= clip_hi;
        let r_is_crush = r <= clip_lo;
        let g_is_clip = g >= clip_hi;
        let g_is_crush = g <= clip_lo;
        let b_is_clip = b >= clip_hi;
        let b_is_crush = b <= clip_lo;

        if r_is_clip {
            r_clip += 1;
        }
        if r_is_crush {
            r_crush += 1;
        }
        if g_is_clip {
            g_clip += 1;
        }
        if g_is_crush {
            g_crush += 1;
        }
        if b_is_clip {
            b_clip += 1;
        }
        if b_is_crush {
            b_crush += 1;
        }

        if r_is_clip && g_is_clip && b_is_clip {
            white_clip += 1;
        }
        if r_is_crush && g_is_crush && b_is_crush {
            black_crush += 1;
        }
    }

    let total = pixel_count as u64;
    let total_f = total as f64;

    let make_info = |clip: u64, crush: u64| -> ChannelClipInfo {
        ChannelClipInfo {
            clipped_count: clip,
            crushed_count: crush,
            clip_ratio: clip as f64 / total_f,
            crush_ratio: crush as f64 / total_f,
        }
    };

    let red = make_info(r_clip, r_crush);
    let green = make_info(g_clip, g_crush);
    let blue = make_info(b_clip, b_crush);

    // Count pixels with any clipping or crushing in any channel
    let mut any_affected: u64 = 0;
    for i in 0..pixel_count {
        let base = i * 3;
        let r = frame[base];
        let g = frame[base + 1];
        let b = frame[base + 2];
        if r >= clip_hi
            || r <= clip_lo
            || g >= clip_hi
            || g <= clip_lo
            || b >= clip_hi
            || b <= clip_lo
        {
            any_affected += 1;
        }
    }

    Some(ClippingReport {
        red,
        green,
        blue,
        white_clip_count: white_clip,
        black_crush_count: black_crush,
        total_pixels: total,
        overall_severity: any_affected as f64 / total_f,
    })
}

/// Generates a clipping mask for a frame.
///
/// Returns a single-channel buffer where:
/// - 0 = normal pixel
/// - 1 = clipped (highlight)
/// - 2 = crushed (shadow)
/// - 3 = both clipped and crushed in different channels
#[must_use]
pub fn generate_clip_mask(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ClippingConfig,
) -> Option<Vec<u8>> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 || frame.len() < pixel_count * 3 {
        return None;
    }

    let clip_hi = config.effective_clip();
    let clip_lo = config.effective_crush();
    let mut mask = vec![0u8; pixel_count];

    for i in 0..pixel_count {
        let base = i * 3;
        let r = frame[base];
        let g = frame[base + 1];
        let b = frame[base + 2];

        let any_clip = r >= clip_hi || g >= clip_hi || b >= clip_hi;
        let any_crush = r <= clip_lo || g <= clip_lo || b <= clip_lo;

        mask[i] = match (any_clip, any_crush) {
            (true, true) => 3,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 0,
        };
    }

    Some(mask)
}

/// Checks whether a frame is broadcast-safe (all samples within 16-235 range).
#[must_use]
pub fn is_broadcast_safe(frame: &[u8], width: u32, height: u32) -> Option<bool> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 || frame.len() < pixel_count * 3 {
        return None;
    }

    for i in 0..pixel_count {
        let base = i * 3;
        for ch in 0..3 {
            let v = frame[base + ch];
            if v < 16 || v > 235 {
                return Some(false);
            }
        }
    }
    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let pc = (width as usize) * (height as usize);
        let mut data = vec![0u8; pc * 3];
        for i in 0..pc {
            data[i * 3] = r;
            data[i * 3 + 1] = g;
            data[i * 3 + 2] = b;
        }
        data
    }

    #[test]
    fn test_no_clipping() {
        let frame = make_frame(4, 4, 128, 128, 128);
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 4, 4, &config).expect("should succeed in test");
        assert_eq!(report.red.clipped_count, 0);
        assert_eq!(report.red.crushed_count, 0);
        assert!(report.overall_severity < f64::EPSILON);
    }

    #[test]
    fn test_all_white_clipped() {
        let frame = make_frame(4, 4, 255, 255, 255);
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 4, 4, &config).expect("should succeed in test");
        assert_eq!(report.red.clipped_count, 16);
        assert_eq!(report.white_clip_count, 16);
        assert!((report.overall_severity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_black_crushed() {
        let frame = make_frame(4, 4, 0, 0, 0);
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 4, 4, &config).expect("should succeed in test");
        assert_eq!(report.red.crushed_count, 16);
        assert_eq!(report.black_crush_count, 16);
    }

    #[test]
    fn test_red_only_clipped() {
        let frame = make_frame(4, 4, 255, 128, 128);
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 4, 4, &config).expect("should succeed in test");
        assert_eq!(report.red.clipped_count, 16);
        assert_eq!(report.green.clipped_count, 0);
        assert_eq!(report.blue.clipped_count, 0);
        assert_eq!(report.white_clip_count, 0);
    }

    #[test]
    fn test_broadcast_safe_config() {
        let config = ClippingConfig::broadcast();
        assert_eq!(config.effective_clip(), 235);
        assert_eq!(config.effective_crush(), 16);
    }

    #[test]
    fn test_broadcast_clipping() {
        let frame = make_frame(2, 2, 240, 128, 128);
        let config = ClippingConfig::broadcast();
        let report = detect_clipping(&frame, 2, 2, &config).expect("should succeed in test");
        assert_eq!(report.red.clipped_count, 4); // 240 >= 235
    }

    #[test]
    fn test_invalid_frame() {
        let config = ClippingConfig::default();
        assert!(detect_clipping(&[0u8; 5], 10, 10, &config).is_none());
    }

    #[test]
    fn test_zero_dimensions() {
        let config = ClippingConfig::default();
        assert!(detect_clipping(&[], 0, 0, &config).is_none());
    }

    #[test]
    fn test_clip_ratio() {
        // 2 of 4 pixels clipped
        let mut frame = make_frame(4, 1, 128, 128, 128);
        frame[0] = 255; // pixel 0 red clipped
        frame[3] = 255; // pixel 1 red clipped
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 4, 1, &config).expect("should succeed in test");
        assert!((report.red.clip_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_mask_normal() {
        let frame = make_frame(2, 2, 128, 128, 128);
        let config = ClippingConfig::default();
        let mask = generate_clip_mask(&frame, 2, 2, &config).expect("should succeed in test");
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_clip_mask_clipped() {
        let frame = make_frame(2, 2, 255, 128, 128);
        let config = ClippingConfig::default();
        let mask = generate_clip_mask(&frame, 2, 2, &config).expect("should succeed in test");
        assert!(mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_clip_mask_crushed() {
        let frame = make_frame(2, 2, 0, 128, 128);
        let config = ClippingConfig::default();
        let mask = generate_clip_mask(&frame, 2, 2, &config).expect("should succeed in test");
        assert!(mask.iter().all(|&v| v == 2));
    }

    #[test]
    fn test_clip_mask_invalid() {
        let config = ClippingConfig::default();
        assert!(generate_clip_mask(&[], 0, 0, &config).is_none());
    }

    #[test]
    fn test_is_broadcast_safe_yes() {
        let frame = make_frame(2, 2, 100, 100, 100);
        assert_eq!(is_broadcast_safe(&frame, 2, 2), Some(true));
    }

    #[test]
    fn test_is_broadcast_safe_no() {
        let frame = make_frame(2, 2, 10, 100, 100); // 10 < 16
        assert_eq!(is_broadcast_safe(&frame, 2, 2), Some(false));
    }

    #[test]
    fn test_is_broadcast_safe_invalid() {
        assert!(is_broadcast_safe(&[], 0, 0).is_none());
    }

    #[test]
    fn test_total_pixels() {
        let frame = make_frame(10, 5, 128, 128, 128);
        let config = ClippingConfig::default();
        let report = detect_clipping(&frame, 10, 5, &config).expect("should succeed in test");
        assert_eq!(report.total_pixels, 50);
    }
}
