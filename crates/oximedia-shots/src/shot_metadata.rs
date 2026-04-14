//! Shot metadata enrichment.
//!
//! Given a sequence of frames belonging to a detected shot, this module
//! computes rich photometric and motion-based metadata tags:
//!
//! | Field                  | Description                                          |
//! |------------------------|------------------------------------------------------|
//! | `duration_frames`      | Number of frames in the shot                         |
//! | `avg_brightness`       | Mean luminance (0–255) across all frames             |
//! | `brightness_variance`  | Inter-frame luminance variance                       |
//! | `dominant_color`       | Most representative RGB colour (k=1 centroid)        |
//! | `secondary_color`      | Second representative RGB colour (k=2 centroid)      |
//! | `motion_magnitude`     | Mean per-frame optical-flow proxy (0–1)              |
//! | `motion_category`      | `Static`, `Slow`, `Medium`, or `Fast` motion         |
//! | `contrast`             | RMS contrast of the representative frame             |
//! | `scene_change_score`   | Maximum IFD across the shot (cut-like spike = high)  |
//!
//! # Algorithm
//!
//! - **Dominant colour** is estimated by a fast 2-centroid k-means pass on
//!   a 5%-sampled pixel set (luminance-weighted), then selecting the centroid
//!   with the largest cluster.
//! - **Motion magnitude** uses the mean-absolute-difference of successive
//!   luminance planes as a low-cost proxy for optical flow energy.
//! - All computations are pure Rust with no external crate dependencies.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Motion intensity classification derived from `motion_magnitude`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionCategory {
    /// Near-zero camera and subject movement (tripod dialogue, locked-off).
    Static,
    /// Subtle motion — slow pan, gentle movement (< 2% mean pixel change).
    Slow,
    /// Moderate motion — walking subject, steady cam (2–8%).
    Medium,
    /// High motion — action, handheld chase, rapid pan (> 8%).
    Fast,
}

impl MotionCategory {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Static => "Static",
            Self::Slow => "Slow",
            Self::Medium => "Medium",
            Self::Fast => "Fast",
        }
    }
}

/// An RGB colour represented as three `u8` components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbColor {
    /// Red component (0–255).
    pub r: u8,
    /// Green component (0–255).
    pub g: u8,
    /// Blue component (0–255).
    pub b: u8,
}

impl RgbColor {
    /// Construct from components.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Euclidean distance² in RGB space.
    #[must_use]
    pub fn dist_sq(self, other: Self) -> u32 {
        let dr = i32::from(self.r) - i32::from(other.r);
        let dg = i32::from(self.g) - i32::from(other.g);
        let db = i32::from(self.b) - i32::from(other.b);
        (dr * dr + dg * dg + db * db) as u32
    }
}

/// Computed metadata for a single shot.
#[derive(Debug, Clone)]
pub struct ShotMetadata {
    /// Number of frames in the shot.
    pub duration_frames: usize,
    /// Mean luminance across all frames (BT.601, 0–255).
    pub avg_brightness: f32,
    /// Inter-frame luminance variance (0–255²).
    pub brightness_variance: f32,
    /// Most dominant colour (k-means centroid with largest cluster).
    pub dominant_color: RgbColor,
    /// Second most dominant colour (`None` if only one frame was given).
    pub secondary_color: Option<RgbColor>,
    /// Mean inter-frame difference as a motion proxy (0.0–1.0).
    pub motion_magnitude: f32,
    /// Categorical motion intensity.
    pub motion_category: MotionCategory,
    /// RMS contrast of the median/representative frame (0.0–1.0).
    pub contrast: f32,
    /// Maximum inter-frame difference across the shot (0.0–1.0).
    /// A high value near a frame boundary indicates a potential cut.
    pub scene_change_score: f32,
}

/// Configuration for [`ShotMetadataExtractor`].
#[derive(Debug, Clone)]
pub struct MetadataConfig {
    /// Fraction of pixels sampled for colour analysis (0 < ratio ≤ 1.0).
    pub color_sample_ratio: f32,
    /// Number of k-means iterations for colour clustering.
    pub kmeans_iterations: usize,
    /// IFD threshold below which motion is `Static`.
    pub static_ifd_threshold: f32,
    /// IFD threshold below which motion is `Slow` (must be > static).
    pub slow_ifd_threshold: f32,
    /// IFD threshold below which motion is `Medium` (must be > slow).
    pub medium_ifd_threshold: f32,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            color_sample_ratio: 0.05,
            kmeans_iterations: 10,
            static_ifd_threshold: 0.005,
            slow_ifd_threshold: 0.02,
            medium_ifd_threshold: 0.08,
        }
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Extracts rich photometric and motion metadata from a shot's frame sequence.
pub struct ShotMetadataExtractor {
    config: MetadataConfig,
}

impl Default for ShotMetadataExtractor {
    fn default() -> Self {
        Self::new(MetadataConfig::default())
    }
}

impl ShotMetadataExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: MetadataConfig) -> Self {
        Self { config }
    }

    /// Extract metadata from a sequence of frames belonging to one shot.
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] if `frames` is empty or any frame
    /// has fewer than 3 channels.
    pub fn extract(&self, frames: &[FrameBuffer]) -> ShotResult<ShotMetadata> {
        if frames.is_empty() {
            return Err(ShotError::InvalidFrame(
                "Cannot extract metadata from an empty frame sequence".to_string(),
            ));
        }
        for (idx, frame) in frames.iter().enumerate() {
            let (_, _, ch) = frame.dim();
            if ch < 3 {
                return Err(ShotError::InvalidFrame(format!(
                    "Frame {idx} has only {ch} channel(s); at least 3 required"
                )));
            }
        }

        let duration_frames = frames.len();

        // Per-frame mean luminance
        let frame_lumas: Vec<f32> = frames.iter().map(mean_luma).collect();
        let avg_brightness = frame_lumas.iter().sum::<f32>() / frame_lumas.len() as f32;
        let brightness_variance = variance(&frame_lumas);

        // Inter-frame differences (IFD)
        let ifds: Vec<f32> = if frames.len() > 1 {
            frames
                .windows(2)
                .map(|w| mean_abs_luma_diff_internal(&w[0], &w[1]))
                .collect()
        } else {
            Vec::new()
        };

        let motion_magnitude = if ifds.is_empty() {
            0.0
        } else {
            ifds.iter().sum::<f32>() / ifds.len() as f32
        };

        let scene_change_score = ifds.iter().copied().fold(0.0_f32, f32::max);

        let motion_category = self.classify_motion(motion_magnitude);

        // Use the middle frame as representative for colour and contrast.
        let rep_frame = &frames[frames.len() / 2];

        let (dominant_color, secondary_color) = self.extract_colors(rep_frame)?;
        let contrast = rms_contrast(rep_frame);

        Ok(ShotMetadata {
            duration_frames,
            avg_brightness,
            brightness_variance,
            dominant_color,
            secondary_color,
            motion_magnitude,
            motion_category,
            contrast,
            scene_change_score,
        })
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &MetadataConfig {
        &self.config
    }

    // ---- Private helpers ----

    fn classify_motion(&self, ifd: f32) -> MotionCategory {
        if ifd <= self.config.static_ifd_threshold {
            MotionCategory::Static
        } else if ifd <= self.config.slow_ifd_threshold {
            MotionCategory::Slow
        } else if ifd <= self.config.medium_ifd_threshold {
            MotionCategory::Medium
        } else {
            MotionCategory::Fast
        }
    }

    /// Fast 2-centroid k-means on a sampled pixel set.
    ///
    /// Returns `(dominant, secondary)` where `secondary` is `None` when the
    /// frame is so uniform that both centroids collapse.
    fn extract_colors(&self, frame: &FrameBuffer) -> ShotResult<(RgbColor, Option<RgbColor>)> {
        let (h, w, _) = frame.dim();
        let total = h * w;
        if total == 0 {
            return Err(ShotError::InvalidFrame(
                "Empty frame for colour extraction".to_string(),
            ));
        }

        // Sample pixels
        let step = ((1.0 / self.config.color_sample_ratio).round() as usize).max(1);
        let mut samples: Vec<(u8, u8, u8)> = Vec::new();
        let mut i = 0_usize;
        'outer: for y in 0..h {
            for x in 0..w {
                if i % step == 0 {
                    let r = frame.get(y, x, 0);
                    let g = frame.get(y, x, 1);
                    let b = frame.get(y, x, 2);
                    samples.push((r, g, b));
                }
                i += 1;
                if samples.len() > 4096 {
                    break 'outer;
                }
            }
        }

        if samples.is_empty() {
            // Fallback: just use the first pixel
            let r = frame.get(0, 0, 0);
            let g = frame.get(0, 0, 1);
            let b = frame.get(0, 0, 2);
            return Ok((RgbColor::new(r, g, b), None));
        }

        // Initialise 2 centroids: first and last sample
        let n = samples.len();
        let mut c0 = samples[0];
        let mut c1 = samples[n - 1];

        for _ in 0..self.config.kmeans_iterations {
            let mut sum0 = (0u64, 0u64, 0u64);
            let mut cnt0 = 0u64;
            let mut sum1 = (0u64, 0u64, 0u64);
            let mut cnt1 = 0u64;

            for &(r, g, b) in &samples {
                let d0 = color_dist_sq(r, g, b, c0.0, c0.1, c0.2);
                let d1 = color_dist_sq(r, g, b, c1.0, c1.1, c1.2);
                if d0 <= d1 {
                    sum0.0 += u64::from(r);
                    sum0.1 += u64::from(g);
                    sum0.2 += u64::from(b);
                    cnt0 += 1;
                } else {
                    sum1.0 += u64::from(r);
                    sum1.1 += u64::from(g);
                    sum1.2 += u64::from(b);
                    cnt1 += 1;
                }
            }

            if let Some(c0r) = sum0.0.checked_div(cnt0) {
                c0 = (c0r as u8, (sum0.1 / cnt0) as u8, (sum0.2 / cnt0) as u8);
            }
            if let Some(c1r) = sum1.0.checked_div(cnt1) {
                c1 = (c1r as u8, (sum1.1 / cnt1) as u8, (sum1.2 / cnt1) as u8);
            }
        }

        // Final assignment to count cluster sizes
        let mut cnt0 = 0u64;
        let mut cnt1 = 0u64;
        for &(r, g, b) in &samples {
            let d0 = color_dist_sq(r, g, b, c0.0, c0.1, c0.2);
            let d1 = color_dist_sq(r, g, b, c1.0, c1.1, c1.2);
            if d0 <= d1 {
                cnt0 += 1;
            } else {
                cnt1 += 1;
            }
        }

        let (dominant, secondary) = if cnt0 >= cnt1 {
            (
                RgbColor::new(c0.0, c0.1, c0.2),
                RgbColor::new(c1.0, c1.1, c1.2),
            )
        } else {
            (
                RgbColor::new(c1.0, c1.1, c1.2),
                RgbColor::new(c0.0, c0.1, c0.2),
            )
        };

        // If centroids are very similar, suppress secondary
        let secondary_opt = if dominant.dist_sq(secondary) < 100 {
            None
        } else {
            Some(secondary)
        };

        Ok((dominant, secondary_opt))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Mean luminance (BT.601) for a frame (0–255).
fn mean_luma(frame: &FrameBuffer) -> f32 {
    let (h, w, _) = frame.dim();
    let n = (h * w) as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let r = f64::from(frame.get(y, x, 0));
            let g = f64::from(frame.get(y, x, 1));
            let b = f64::from(frame.get(y, x, 2));
            sum += r * 0.299 + g * 0.587 + b * 0.114;
        }
    }
    (sum / n) as f32
}

/// Mean absolute luminance difference between two frames (normalised 0–1).
fn mean_abs_luma_diff_internal(a: &FrameBuffer, b: &FrameBuffer) -> f32 {
    let (h, w, _) = a.dim();
    let (h2, w2, _) = b.dim();
    if h != h2 || w != w2 || h == 0 || w == 0 {
        return 0.0;
    }
    let n = (h * w) as f64;
    let mut sum = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let la = f64::from(a.get(y, x, 0)) * 0.299
                + f64::from(a.get(y, x, 1)) * 0.587
                + f64::from(a.get(y, x, 2)) * 0.114;
            let lb = f64::from(b.get(y, x, 0)) * 0.299
                + f64::from(b.get(y, x, 1)) * 0.587
                + f64::from(b.get(y, x, 2)) * 0.114;
            sum += (la - lb).abs();
        }
    }
    (sum / (n * 255.0)) as f32
}

/// Population variance of a slice.
fn variance(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / n;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n
}

/// RMS contrast of a frame, defined as the standard deviation of per-pixel
/// luminance, normalised to [0, 1] by dividing by 255.
fn rms_contrast(frame: &FrameBuffer) -> f32 {
    let (h, w, _) = frame.dim();
    let n = (h * w) as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let l = f64::from(frame.get(y, x, 0)) * 0.299
                + f64::from(frame.get(y, x, 1)) * 0.587
                + f64::from(frame.get(y, x, 2)) * 0.114;
            sum += l;
            sum_sq += l * l;
        }
    }
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    (variance.sqrt() / 255.0) as f32
}

/// Squared Euclidean distance in RGB space.
#[inline]
fn color_dist_sq(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> u32 {
    let dr = i32::from(r1) - i32::from(r2);
    let dg = i32::from(g1) - i32::from(g2);
    let db = i32::from(b1) - i32::from(b2);
    (dr * dr + dg * dg + db * db) as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_buffer::FrameBuffer;

    // ---- Helpers ----

    fn flat_frame(h: usize, w: usize, r: u8, g: u8, b: u8) -> FrameBuffer {
        let mut f = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                f.set(y, x, 0, r);
                f.set(y, x, 1, g);
                f.set(y, x, 2, b);
            }
        }
        f
    }

    // ---- MotionCategory ----

    #[test]
    fn test_motion_category_names() {
        assert_eq!(MotionCategory::Static.name(), "Static");
        assert_eq!(MotionCategory::Slow.name(), "Slow");
        assert_eq!(MotionCategory::Medium.name(), "Medium");
        assert_eq!(MotionCategory::Fast.name(), "Fast");
    }

    // ---- RgbColor ----

    #[test]
    fn test_rgb_color_dist_sq_zero() {
        let c = RgbColor::new(100, 150, 200);
        assert_eq!(c.dist_sq(c), 0);
    }

    #[test]
    fn test_rgb_color_dist_sq_known() {
        let a = RgbColor::new(0, 0, 0);
        let b = RgbColor::new(3, 4, 0);
        // sqrt(9+16+0) = 5; dist_sq = 25
        assert_eq!(a.dist_sq(b), 25);
    }

    // ---- extract returns error on empty frames ----

    #[test]
    fn test_extract_empty_frames_error() {
        let extractor = ShotMetadataExtractor::default();
        assert!(extractor.extract(&[]).is_err());
    }

    // ---- extract error on wrong channel count ----

    #[test]
    fn test_extract_wrong_channels_error() {
        let extractor = ShotMetadataExtractor::default();
        let f = FrameBuffer::zeros(32, 32, 1);
        assert!(extractor.extract(&[f]).is_err());
    }

    // ---- single frame — static motion ----

    #[test]
    fn test_single_frame_static_motion() {
        let extractor = ShotMetadataExtractor::default();
        let f = flat_frame(32, 32, 128, 64, 32);
        let meta = extractor
            .extract(&[f])
            .expect("single frame should succeed");
        assert_eq!(meta.duration_frames, 1);
        assert_eq!(meta.motion_category, MotionCategory::Static);
        assert!((meta.motion_magnitude - 0.0).abs() < 1e-5);
        assert_eq!(meta.scene_change_score, 0.0);
    }

    // ---- identical frames have zero motion ----

    #[test]
    fn test_identical_frames_zero_motion() {
        let extractor = ShotMetadataExtractor::default();
        let frames: Vec<FrameBuffer> = (0..5).map(|_| flat_frame(32, 32, 100, 100, 100)).collect();
        let meta = extractor
            .extract(&frames)
            .expect("identical frames should succeed");
        assert!(
            meta.motion_magnitude < 1e-4,
            "motion should be ~0: {}",
            meta.motion_magnitude
        );
        assert_eq!(meta.motion_category, MotionCategory::Static);
    }

    // ---- max-contrast frames → fast motion ----

    #[test]
    fn test_alternating_frames_fast_motion() {
        let extractor = ShotMetadataExtractor::default();
        let frames: Vec<FrameBuffer> = (0..6)
            .map(|i| {
                if i % 2 == 0 {
                    flat_frame(32, 32, 0, 0, 0)
                } else {
                    flat_frame(32, 32, 255, 255, 255)
                }
            })
            .collect();
        let meta = extractor
            .extract(&frames)
            .expect("alternating frames should succeed");
        assert_eq!(meta.motion_category, MotionCategory::Fast);
        assert!(meta.motion_magnitude > 0.5);
    }

    // ---- dominant colour of uniform red frame ----

    #[test]
    fn test_dominant_color_uniform_red() {
        let extractor = ShotMetadataExtractor::default();
        let f = flat_frame(64, 64, 200, 30, 30);
        let meta = extractor.extract(&[f]).expect("red frame should succeed");
        // Dominant colour should be close to the uniform red
        assert!(meta.dominant_color.r > 150, "dominant R should be ~200");
        assert!(meta.dominant_color.g < 100, "dominant G should be ~30");
    }

    // ---- avg_brightness of pure white ----

    #[test]
    fn test_avg_brightness_white_frame() {
        let extractor = ShotMetadataExtractor::default();
        let f = flat_frame(32, 32, 255, 255, 255);
        let meta = extractor.extract(&[f]).expect("white frame");
        // BT.601: 255*0.299 + 255*0.587 + 255*0.114 ≈ 255
        assert!(
            meta.avg_brightness > 250.0,
            "white frame brightness should be ~255"
        );
    }

    // ---- avg_brightness of pure black ----

    #[test]
    fn test_avg_brightness_black_frame() {
        let extractor = ShotMetadataExtractor::default();
        let f = flat_frame(32, 32, 0, 0, 0);
        let meta = extractor.extract(&[f]).expect("black frame");
        assert!(
            meta.avg_brightness < 5.0,
            "black frame brightness should be ~0"
        );
    }

    // ---- contrast of uniform frame is zero ----

    #[test]
    fn test_contrast_uniform_frame_is_zero() {
        let extractor = ShotMetadataExtractor::default();
        let f = flat_frame(32, 32, 128, 128, 128);
        let meta = extractor.extract(&[f]).expect("uniform frame");
        assert!(meta.contrast < 1e-4, "uniform frame contrast should be ~0");
    }

    // ---- scene_change_score peaks near cut ----

    #[test]
    fn test_scene_change_score_near_cut() {
        let extractor = ShotMetadataExtractor::default();
        // 5 similar frames followed by 5 very different frames
        let mut frames: Vec<FrameBuffer> = (0..5).map(|_| flat_frame(32, 32, 50, 50, 50)).collect();
        frames.extend((0..5).map(|_| flat_frame(32, 32, 200, 200, 200)));
        let meta = extractor.extract(&frames).expect("mixed frames");
        assert!(
            meta.scene_change_score > 0.3,
            "scene_change_score should be high near a cut: {}",
            meta.scene_change_score
        );
    }

    // ---- duration_frames ----

    #[test]
    fn test_duration_frames_count() {
        let extractor = ShotMetadataExtractor::default();
        let frames: Vec<FrameBuffer> = (0..7).map(|_| flat_frame(32, 32, 128, 128, 128)).collect();
        let meta = extractor.extract(&frames).expect("7 frames");
        assert_eq!(meta.duration_frames, 7);
    }
}
