//! Efficient thumbnail generation with multi-size output, quality presets,
//! format hints, and time-based frame sampling.
//!
//! Unlike the simpler [`crate::thumbnail`] module, `thumbnail_generator` supports:
//! - **Multi-size batch generation** — produce multiple thumbnails in one pass.
//! - **Quality presets** — `Draft`, `Standard`, `High`, `Lossless` with sensible defaults.
//! - **Format hints** — carry format metadata alongside the pixel data.
//! - **Time-based sampling** — calculate frame numbers from a video timestamp.
//! - **Configurable interpolation** — nearest-neighbor or bilinear.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use thiserror::Error;

/// Errors that arise during thumbnail generation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ThumbnailError {
    /// Source image has at least one zero dimension.
    #[error("source dimensions {width}x{height} are invalid (must be > 0)")]
    InvalidSourceDimensions {
        /// Declared source width.
        width: u32,
        /// Declared source height.
        height: u32,
    },
    /// The requested thumbnail target size has at least one zero dimension.
    #[error("thumbnail target size {width}x{height} is invalid (must be > 0)")]
    InvalidTargetSize {
        /// Requested target width.
        width: u32,
        /// Requested target height.
        height: u32,
    },
    /// Pixel buffer length does not match the declared dimensions × channels.
    #[error("buffer length {actual} does not match {width}x{height}x{channels}")]
    BufferMismatch {
        /// Actual buffer length.
        actual: usize,
        /// Declared source width.
        width: u32,
        /// Declared source height.
        height: u32,
        /// Channel count.
        channels: u32,
    },
    /// The video duration is zero, making time-based sampling undefined.
    #[error("video duration must be positive for time-based sampling")]
    ZeroDuration,
}

/// Quality preset for thumbnail encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityPreset {
    /// Fast, low-quality thumbnail suitable for server-side previews.
    Draft,
    /// Balanced quality for web display (JPEG ~75).
    Standard,
    /// High quality for editorial display (JPEG ~90).
    High,
    /// Lossless / maximum quality.
    Lossless,
}

impl QualityPreset {
    /// Numeric quality value in the range 0–100.
    pub fn quality_value(self) -> u8 {
        match self {
            Self::Draft => 50,
            Self::Standard => 75,
            Self::High => 90,
            Self::Lossless => 100,
        }
    }
}

impl Default for QualityPreset {
    fn default() -> Self {
        Self::Standard
    }
}

/// Hint for the thumbnail container/codec format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatHint {
    /// JPEG encoding hint.
    Jpeg,
    /// WebP encoding hint.
    WebP,
    /// PNG lossless hint.
    Png,
    /// Raw uncompressed pixel data.
    Raw,
}

impl Default for FormatHint {
    fn default() -> Self {
        Self::Jpeg
    }
}

/// Scaling algorithm for thumbnail generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbScaleAlgorithm {
    /// Nearest-neighbor — fast, no smoothing.
    NearestNeighbor,
    /// Bilinear — smooth, slightly slower.
    Bilinear,
}

impl Default for ThumbScaleAlgorithm {
    fn default() -> Self {
        Self::Bilinear
    }
}

/// Specification for a single thumbnail size in a multi-size request.
#[derive(Debug, Clone)]
pub struct ThumbSize {
    /// Maximum target width (long edge may be clamped; aspect ratio is preserved).
    pub max_width: u32,
    /// Maximum target height.
    pub max_height: u32,
    /// Quality preset for this size.
    pub preset: QualityPreset,
    /// Optional human-readable label (e.g. `"small"`, `"medium"`, `"large"`).
    pub label: Option<String>,
}

impl ThumbSize {
    /// Create a new size specification.
    pub fn new(max_width: u32, max_height: u32) -> Self {
        Self {
            max_width,
            max_height,
            preset: QualityPreset::default(),
            label: None,
        }
    }

    /// Builder: set the quality preset.
    pub fn with_preset(mut self, preset: QualityPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Builder: set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Compute output dimensions that fit within this spec while preserving
    /// the source aspect ratio.  The image is never upscaled.
    pub fn compute_dimensions(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        if src_w == 0 || src_h == 0 || self.max_width == 0 || self.max_height == 0 {
            return (0, 0);
        }
        let scale_w = self.max_width as f64 / src_w as f64;
        let scale_h = self.max_height as f64 / src_h as f64;
        let scale = scale_w.min(scale_h).min(1.0); // Never upscale.
        let out_w = ((src_w as f64 * scale).round() as u32).max(1);
        let out_h = ((src_h as f64 * scale).round() as u32).max(1);
        (out_w, out_h)
    }
}

/// A generated thumbnail.
#[derive(Debug, Clone)]
pub struct GeneratedThumbnail {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Raw pixel data in the same channel format as the source input.
    pub data: Vec<u8>,
    /// Quality preset used.
    pub preset: QualityPreset,
    /// Format hint.
    pub format: FormatHint,
    /// Optional label (from the [`ThumbSize`] spec).
    pub label: Option<String>,
}

impl GeneratedThumbnail {
    /// Total pixel count.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Aspect ratio (width / height).  Returns 0.0 if height is zero.
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f64 / self.height as f64
        }
    }
}

/// Configuration for the [`ThumbnailGenerator`].
#[derive(Debug, Clone)]
pub struct ThumbnailGeneratorConfig {
    /// Scaling algorithm.
    pub algorithm: ThumbScaleAlgorithm,
    /// Default format hint applied to all thumbnails (can be overridden per-size in future).
    pub format: FormatHint,
    /// Number of pixel channels in the source data (1 = grey, 3 = RGB, 4 = RGBA).
    pub channels: u32,
}

impl Default for ThumbnailGeneratorConfig {
    fn default() -> Self {
        Self {
            algorithm: ThumbScaleAlgorithm::Bilinear,
            format: FormatHint::Jpeg,
            channels: 3,
        }
    }
}

impl ThumbnailGeneratorConfig {
    /// Create a new config.
    pub fn new(channels: u32) -> Self {
        Self {
            channels,
            ..Default::default()
        }
    }

    /// Builder: set the scaling algorithm.
    pub fn with_algorithm(mut self, algorithm: ThumbScaleAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Builder: set the default format hint.
    pub fn with_format(mut self, format: FormatHint) -> Self {
        self.format = format;
        self
    }
}

/// High-level thumbnail generator.
///
/// ```
/// use oximedia_scaling::thumbnail_generator::{
///     ThumbnailGenerator, ThumbnailGeneratorConfig, ThumbSize, QualityPreset,
/// };
///
/// let pixels = vec![128u8; 1920 * 1080 * 3];
/// let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
///
/// let sizes = vec![
///     ThumbSize::new(640, 360).with_label("medium"),
///     ThumbSize::new(320, 180).with_preset(QualityPreset::Draft).with_label("small"),
/// ];
///
/// let thumbs = gen.generate_multi(&pixels, 1920, 1080, &sizes).unwrap();
/// assert_eq!(thumbs.len(), 2);
/// ```
pub struct ThumbnailGenerator {
    config: ThumbnailGeneratorConfig,
}

impl ThumbnailGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ThumbnailGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a single thumbnail.
    pub fn generate(
        &self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        size: &ThumbSize,
    ) -> Result<GeneratedThumbnail, ThumbnailError> {
        self.validate_source(pixels, src_w, src_h)?;

        if size.max_width == 0 || size.max_height == 0 {
            return Err(ThumbnailError::InvalidTargetSize {
                width: size.max_width,
                height: size.max_height,
            });
        }

        let (dst_w, dst_h) = size.compute_dimensions(src_w, src_h);
        let data = self.scale_pixels(pixels, src_w, src_h, dst_w, dst_h);

        Ok(GeneratedThumbnail {
            width: dst_w,
            height: dst_h,
            data,
            preset: size.preset,
            format: self.config.format,
            label: size.label.clone(),
        })
    }

    /// Generate multiple thumbnails from the same source in a single call.
    pub fn generate_multi(
        &self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        sizes: &[ThumbSize],
    ) -> Result<Vec<GeneratedThumbnail>, ThumbnailError> {
        self.validate_source(pixels, src_w, src_h)?;

        sizes
            .iter()
            .map(|s| self.generate(pixels, src_w, src_h, s))
            .collect()
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn validate_source(&self, pixels: &[u8], src_w: u32, src_h: u32) -> Result<(), ThumbnailError> {
        if src_w == 0 || src_h == 0 {
            return Err(ThumbnailError::InvalidSourceDimensions {
                width: src_w,
                height: src_h,
            });
        }
        let expected = (src_w * src_h * self.config.channels) as usize;
        if pixels.len() != expected {
            return Err(ThumbnailError::BufferMismatch {
                actual: pixels.len(),
                width: src_w,
                height: src_h,
                channels: self.config.channels,
            });
        }
        Ok(())
    }

    fn scale_pixels(
        &self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        let ch = self.config.channels as usize;
        let mut out = vec![0u8; (dst_w * dst_h) as usize * ch];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let dst_idx = (dy * dst_w + dx) as usize * ch;

                match self.config.algorithm {
                    ThumbScaleAlgorithm::NearestNeighbor => {
                        let sx = (dx * src_w / dst_w).min(src_w - 1);
                        let sy = (dy * src_h / dst_h).min(src_h - 1);
                        let src_idx = (sy * src_w + sx) as usize * ch;
                        out[dst_idx..dst_idx + ch].copy_from_slice(&pixels[src_idx..src_idx + ch]);
                    }

                    ThumbScaleAlgorithm::Bilinear => {
                        let src_x_f = (dx as f64 + 0.5) * (src_w as f64 / dst_w as f64) - 0.5;
                        let src_y_f = (dy as f64 + 0.5) * (src_h as f64 / dst_h as f64) - 0.5;

                        let sx0 = src_x_f.floor().clamp(0.0, (src_w - 1) as f64) as u32;
                        let sy0 = src_y_f.floor().clamp(0.0, (src_h - 1) as f64) as u32;
                        let sx1 = (sx0 + 1).min(src_w - 1);
                        let sy1 = (sy0 + 1).min(src_h - 1);
                        let tx = src_x_f - src_x_f.floor();
                        let ty = src_y_f - src_y_f.floor();

                        for c in 0..ch {
                            let p00 = pixels[((sy0 * src_w + sx0) as usize) * ch + c] as f64;
                            let p10 = pixels[((sy0 * src_w + sx1) as usize) * ch + c] as f64;
                            let p01 = pixels[((sy1 * src_w + sx0) as usize) * ch + c] as f64;
                            let p11 = pixels[((sy1 * src_w + sx1) as usize) * ch + c] as f64;

                            let v = p00 * (1.0 - tx) * (1.0 - ty)
                                + p10 * tx * (1.0 - ty)
                                + p01 * (1.0 - tx) * ty
                                + p11 * tx * ty;

                            out[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }

        out
    }
}

// ── Time-based sampling utilities ─────────────────────────────────────────────

/// Compute which frame numbers to sample to generate `count` thumbnails evenly
/// distributed across a video of `total_frames` frames, avoiding exact endpoints
/// to reduce the chance of capturing blank frames.
///
/// Returns a sorted `Vec<u64>` of frame indices in the range `[0, total_frames)`.
pub fn sample_frame_numbers(total_frames: u64, count: usize) -> Vec<u64> {
    if count == 0 || total_frames == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![total_frames / 2];
    }

    (0..count)
        .map(|i| {
            // Distribute evenly within (0, total_frames) avoiding exact 0 and total_frames.
            let fraction = (i as f64 + 0.5) / count as f64;
            ((fraction * total_frames as f64) as u64).min(total_frames - 1)
        })
        .collect()
}

/// Convert a wall-clock timestamp (in seconds) to a frame number given the
/// video's frame rate as a rational `(numerator, denominator)`.
///
/// # Errors
/// Returns `ThumbnailError::ZeroDuration` if either fps component is zero.
pub fn timestamp_to_frame(
    timestamp_secs: f64,
    fps_num: u32,
    fps_den: u32,
) -> Result<u64, ThumbnailError> {
    if fps_num == 0 || fps_den == 0 {
        return Err(ThumbnailError::ZeroDuration);
    }
    let fps = fps_num as f64 / fps_den as f64;
    Ok((timestamp_secs * fps).floor() as u64)
}

/// Evenly distributed timestamps (seconds) for sampling `count` frames from a
/// video of `duration_secs` duration.
///
/// # Errors
/// Returns `ThumbnailError::ZeroDuration` if `duration_secs` is zero or negative.
pub fn sample_timestamps(duration_secs: f64, count: usize) -> Result<Vec<f64>, ThumbnailError> {
    if duration_secs <= 0.0 {
        return Err(ThumbnailError::ZeroDuration);
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    Ok((0..count)
        .map(|i| {
            let fraction = (i as f64 + 0.5) / count as f64;
            fraction * duration_secs
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixels(w: u32, h: u32, channels: u32) -> Vec<u8> {
        (0..(w * h * channels) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    // ── ThumbSize tests ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_dimensions_no_upscale() {
        let s = ThumbSize::new(512, 512);
        let (w, h) = s.compute_dimensions(100, 100);
        assert_eq!((w, h), (100, 100));
    }

    #[test]
    fn test_compute_dimensions_landscape() {
        let s = ThumbSize::new(128, 128);
        let (w, h) = s.compute_dimensions(320, 240);
        assert_eq!(w, 128);
        assert_eq!(h, 96); // 128 * 240/320 = 96
    }

    #[test]
    fn test_compute_dimensions_zero_source() {
        let s = ThumbSize::new(128, 128);
        let (w, h) = s.compute_dimensions(0, 100);
        assert_eq!((w, h), (0, 0));
    }

    // ── QualityPreset tests ───────────────────────────────────────────────────

    #[test]
    fn test_quality_preset_values() {
        assert_eq!(QualityPreset::Draft.quality_value(), 50);
        assert_eq!(QualityPreset::Standard.quality_value(), 75);
        assert_eq!(QualityPreset::High.quality_value(), 90);
        assert_eq!(QualityPreset::Lossless.quality_value(), 100);
    }

    #[test]
    fn test_quality_preset_ordering() {
        assert!(QualityPreset::Draft < QualityPreset::Standard);
        assert!(QualityPreset::High > QualityPreset::Standard);
        assert!(QualityPreset::Lossless > QualityPreset::High);
    }

    // ── ThumbnailGenerator single tests ──────────────────────────────────────

    #[test]
    fn test_generate_basic() {
        let pixels = make_pixels(640, 480, 3);
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let size = ThumbSize::new(160, 120).with_label("small");
        let thumb = gen.generate(&pixels, 640, 480, &size).unwrap();
        assert_eq!(thumb.width, 160);
        assert_eq!(thumb.height, 120);
        assert_eq!(thumb.data.len(), 160 * 120 * 3);
        assert_eq!(thumb.label.as_deref(), Some("small"));
    }

    #[test]
    fn test_generate_zero_source_error() {
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let size = ThumbSize::new(160, 120);
        let err = gen.generate(&[], 0, 480, &size).unwrap_err();
        assert!(matches!(
            err,
            ThumbnailError::InvalidSourceDimensions { .. }
        ));
    }

    #[test]
    fn test_generate_buffer_mismatch_error() {
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let size = ThumbSize::new(160, 120);
        // Wrong length buffer.
        let err = gen.generate(&[0u8; 100], 640, 480, &size).unwrap_err();
        assert!(matches!(err, ThumbnailError::BufferMismatch { .. }));
    }

    #[test]
    fn test_generate_zero_target_error() {
        let pixels = make_pixels(640, 480, 3);
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let size = ThumbSize::new(0, 120);
        let err = gen.generate(&pixels, 640, 480, &size).unwrap_err();
        assert!(matches!(err, ThumbnailError::InvalidTargetSize { .. }));
    }

    // ── Multi-size tests ──────────────────────────────────────────────────────

    #[test]
    fn test_generate_multi_count() {
        let pixels = make_pixels(1920, 1080, 3);
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let sizes = vec![
            ThumbSize::new(1280, 720),
            ThumbSize::new(640, 360),
            ThumbSize::new(320, 180),
        ];
        let thumbs = gen.generate_multi(&pixels, 1920, 1080, &sizes).unwrap();
        assert_eq!(thumbs.len(), 3);
    }

    #[test]
    fn test_generate_multi_aspect_ratio_preserved() {
        let pixels = make_pixels(1920, 1080, 3);
        let gen = ThumbnailGenerator::new(ThumbnailGeneratorConfig::new(3));
        let sizes = vec![ThumbSize::new(640, 640)]; // Square box for non-square source.
        let thumbs = gen.generate_multi(&pixels, 1920, 1080, &sizes).unwrap();
        let t = &thumbs[0];
        let ratio = t.aspect_ratio();
        // 1920/1080 ≈ 1.778
        assert!(
            (ratio - 1920.0 / 1080.0).abs() < 0.01,
            "aspect_ratio={ratio}"
        );
    }

    // ── Nearest-neighbor algorithm test ──────────────────────────────────────

    #[test]
    fn test_nearest_neighbor_solid_color() {
        let pixels = vec![200u8, 100u8, 50u8].repeat(64 * 64);
        let cfg =
            ThumbnailGeneratorConfig::new(3).with_algorithm(ThumbScaleAlgorithm::NearestNeighbor);
        let gen = ThumbnailGenerator::new(cfg);
        let size = ThumbSize::new(32, 32);
        let thumb = gen.generate(&pixels, 64, 64, &size).unwrap();
        for i in 0..32 * 32usize {
            assert_eq!(thumb.data[i * 3], 200);
            assert_eq!(thumb.data[i * 3 + 1], 100);
            assert_eq!(thumb.data[i * 3 + 2], 50);
        }
    }

    // ── Time-based sampling tests ─────────────────────────────────────────────

    #[test]
    fn test_sample_frame_numbers_basic() {
        let frames = sample_frame_numbers(100, 4);
        assert_eq!(frames.len(), 4);
        // All indices must be in [0, 100).
        for &f in &frames {
            assert!(f < 100, "frame {f} out of bounds");
        }
        // Must be sorted.
        for w in frames.windows(2) {
            assert!(w[0] <= w[1], "frames not sorted: {:?}", frames);
        }
    }

    #[test]
    fn test_sample_frame_numbers_zero_count() {
        assert!(sample_frame_numbers(100, 0).is_empty());
    }

    #[test]
    fn test_sample_frame_numbers_single() {
        let frames = sample_frame_numbers(100, 1);
        assert_eq!(frames, vec![50]);
    }

    #[test]
    fn test_timestamp_to_frame_basic() {
        // 30 fps, 2.0 s → frame 60.
        let f = timestamp_to_frame(2.0, 30, 1).unwrap();
        assert_eq!(f, 60);
    }

    #[test]
    fn test_timestamp_to_frame_zero_fps_error() {
        let err = timestamp_to_frame(1.0, 0, 1).unwrap_err();
        assert!(matches!(err, ThumbnailError::ZeroDuration));
    }

    #[test]
    fn test_sample_timestamps_basic() {
        let ts = sample_timestamps(10.0, 4).unwrap();
        assert_eq!(ts.len(), 4);
        for &t in &ts {
            assert!(t >= 0.0 && t < 10.0, "timestamp {t} out of range");
        }
    }

    #[test]
    fn test_sample_timestamps_zero_duration_error() {
        let err = sample_timestamps(0.0, 4).unwrap_err();
        assert!(matches!(err, ThumbnailError::ZeroDuration));
    }

    #[test]
    fn test_sample_timestamps_zero_count() {
        let ts = sample_timestamps(60.0, 0).unwrap();
        assert!(ts.is_empty());
    }
}
