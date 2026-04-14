//! Frame blending for motion blur and time-average compositing.
//!
//! Frame blending is used in post-production and broadcast to:
//!
//! * Simulate **motion blur** by blending multiple frames within a shutter
//!   window (as a film camera would capture).
//! * Create **time-average composites** where a sliding or expanding window
//!   of frames is averaged to expose temporal patterns.
//! * Perform **ghost / long-exposure** effects for creative output.
//! * Smooth frame-rate conversions when combined with a motion estimator.
//!
//! ## Supported modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | [`BlendMode::Equal`] | All frames weighted equally. |
//! | [`BlendMode::Linear`] | Linearly-ramped weights (most recent frame has highest weight). |
//! | [`BlendMode::Gaussian`] | Bell-curve weights centred on the middle of the window. |
//! | [`BlendMode::Custom`] | User-supplied per-frame weights. |
//!
//! All modes use the same normalised blending pipeline so that output pixels
//! are always in the valid `[0, 255]` range regardless of the number of frames.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during frame blending.
#[derive(Debug, Error)]
pub enum FrameBlendError {
    /// No frames were supplied to the blender.
    #[error("no frames supplied for blending")]
    NoFrames,
    /// The weight vector length does not match the frame count.
    #[error("weight count {weights} does not match frame count {frames}")]
    WeightMismatch {
        /// Number of supplied weights.
        weights: usize,
        /// Number of supplied frames.
        frames: usize,
    },
    /// A supplied frame is shorter than `width × height`.
    #[error("frame {index} has length {got} which is shorter than required {expected}")]
    FrameTooShort {
        /// 0-based index of the offending frame.
        index: usize,
        /// Actual frame buffer length.
        got: usize,
        /// Required length (`width × height`).
        expected: usize,
    },
    /// Frame dimensions are degenerate.
    #[error("degenerate dimensions {width}×{height}")]
    DegenerateDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
    /// All supplied weights sum to zero.
    #[error("weights sum to zero — cannot normalise")]
    ZeroWeightSum,
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Weighting strategy for frame blending.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendMode {
    /// Each frame contributes equally to the output.
    Equal,
    /// Frames are weighted linearly: `w[i] = i + 1` (latest frame = highest weight).
    Linear,
    /// Frames are weighted by a Gaussian centred on the middle of the window.
    /// `sigma_factor` scales the Gaussian standard deviation relative to the
    /// half-width of the window.  A value of `1.0` is a good default.
    Gaussian {
        /// Scaling factor for the Gaussian standard deviation.
        sigma_factor: f64,
    },
    /// User-supplied per-frame weights.  The `Vec` must have the same length
    /// as the frame slice passed to the blending functions.
    Custom(Vec<f64>),
}

/// Configuration for the [`FrameBlender`] stateful blender.
#[derive(Debug, Clone)]
pub struct BlendConfig {
    /// Number of frames to keep in the blending window.
    pub window_size: usize,
    /// Weighting strategy.
    pub mode: BlendMode,
}

impl BlendConfig {
    /// Create a new `BlendConfig` with equal weighting.
    pub fn new_equal(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            mode: BlendMode::Equal,
        }
    }

    /// Create a new `BlendConfig` with linear weighting.
    pub fn new_linear(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            mode: BlendMode::Linear,
        }
    }

    /// Create a new `BlendConfig` with Gaussian weighting.
    pub fn new_gaussian(window_size: usize, sigma_factor: f64) -> Self {
        Self {
            window_size: window_size.max(1),
            mode: BlendMode::Gaussian { sigma_factor },
        }
    }
}

/// Stateful frame blender that accumulates a sliding window of frames.
///
/// New frames are added via [`push`][`FrameBlender::push`] and the blended
/// result is obtained via [`blend`][`FrameBlender::blend`].
#[derive(Debug, Clone)]
pub struct FrameBlender {
    config: BlendConfig,
    /// Sliding window of raw frame data (oldest first).
    window: std::collections::VecDeque<Vec<u8>>,
    /// Cached frame dimensions from the first push.
    frame_width: u32,
    frame_height: u32,
}

impl FrameBlender {
    /// Create a new `FrameBlender`.
    pub fn new(config: BlendConfig) -> Self {
        let cap = config.window_size;
        Self {
            config,
            window: std::collections::VecDeque::with_capacity(cap),
            frame_width: 0,
            frame_height: 0,
        }
    }

    /// Push a new frame into the blending window.
    ///
    /// The first call sets the expected `width` and `height`; subsequent calls
    /// must use the same dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`FrameBlendError`] if dimensions are degenerate or the buffer
    /// is too short.
    pub fn push(&mut self, frame: Vec<u8>, width: u32, height: u32) -> Result<(), FrameBlendError> {
        validate_dimensions(width, height)?;
        let expected = (width as usize) * (height as usize);
        if frame.len() < expected {
            return Err(FrameBlendError::FrameTooShort {
                index: self.window.len(),
                got: frame.len(),
                expected,
            });
        }

        self.frame_width = width;
        self.frame_height = height;

        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(frame);
        Ok(())
    }

    /// Blend all frames currently in the window and return the result.
    ///
    /// Returns `None` if the window is empty.
    ///
    /// # Errors
    ///
    /// Returns [`FrameBlendError`] if blending fails.
    pub fn blend(&self) -> Result<Option<Vec<u8>>, FrameBlendError> {
        if self.window.is_empty() {
            return Ok(None);
        }

        let frames: Vec<&[u8]> = self.window.iter().map(|v| v.as_slice()).collect();
        let weights = build_weights(&self.config.mode, frames.len())?;
        let out = blend_frames_weighted(&frames, &weights, self.frame_width, self.frame_height)?;
        Ok(Some(out))
    }

    /// Number of frames currently in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Clear all accumulated frames.
    pub fn clear(&mut self) {
        self.window.clear();
    }
}

// ---------------------------------------------------------------------------
// Public free functions
// ---------------------------------------------------------------------------

/// Blend multiple frames with equal weighting.
///
/// This is a convenience wrapper around [`blend_frames_weighted`] with
/// automatically-generated equal weights.
///
/// # Parameters
///
/// * `frames` — slice of frame buffer references, all `width × height` bytes.
/// * `width` / `height` — frame dimensions.
///
/// # Errors
///
/// Returns [`FrameBlendError`] on validation failure.
pub fn blend_equal(frames: &[&[u8]], width: u32, height: u32) -> Result<Vec<u8>, FrameBlendError> {
    validate_dimensions(width, height)?;
    if frames.is_empty() {
        return Err(FrameBlendError::NoFrames);
    }
    let weights = vec![1.0f64; frames.len()];
    blend_frames_weighted(frames, &weights, width, height)
}

/// Blend multiple frames with linearly increasing weights.
///
/// Frame 0 has weight 1, frame `n-1` has weight `n`.  This biases the output
/// towards the most recent (last) frame.
///
/// # Errors
///
/// Returns [`FrameBlendError`] on validation failure.
pub fn blend_linear(frames: &[&[u8]], width: u32, height: u32) -> Result<Vec<u8>, FrameBlendError> {
    validate_dimensions(width, height)?;
    if frames.is_empty() {
        return Err(FrameBlendError::NoFrames);
    }
    let weights: Vec<f64> = (1..=frames.len()).map(|i| i as f64).collect();
    blend_frames_weighted(frames, &weights, width, height)
}

/// Blend multiple frames with Gaussian weights centred on the middle frame.
///
/// `sigma_factor` scales the standard deviation relative to the half-width of
/// the frame array.  A value of `1.0` is a sensible default giving a smooth
/// roll-off.
///
/// # Errors
///
/// Returns [`FrameBlendError`] on validation failure.
pub fn blend_gaussian(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    sigma_factor: f64,
) -> Result<Vec<u8>, FrameBlendError> {
    validate_dimensions(width, height)?;
    if frames.is_empty() {
        return Err(FrameBlendError::NoFrames);
    }
    let n = frames.len();
    let weights = gaussian_weights(n, sigma_factor);
    blend_frames_weighted(frames, &weights, width, height)
}

/// Blend frames with explicit per-frame weights.
///
/// `weights` must have the same length as `frames`.  Weights need not be
/// normalised — they are normalised internally.
///
/// # Errors
///
/// Returns [`FrameBlendError`] on validation failure.
pub fn blend_frames_weighted(
    frames: &[&[u8]],
    weights: &[f64],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, FrameBlendError> {
    validate_dimensions(width, height)?;
    if frames.is_empty() {
        return Err(FrameBlendError::NoFrames);
    }
    if weights.len() != frames.len() {
        return Err(FrameBlendError::WeightMismatch {
            weights: weights.len(),
            frames: frames.len(),
        });
    }

    let pixel_count = (width as usize) * (height as usize);

    // Validate each frame.
    for (i, frame) in frames.iter().enumerate() {
        if frame.len() < pixel_count {
            return Err(FrameBlendError::FrameTooShort {
                index: i,
                got: frame.len(),
                expected: pixel_count,
            });
        }
    }

    let weight_sum: f64 = weights.iter().sum();
    if weight_sum.abs() < f64::EPSILON {
        return Err(FrameBlendError::ZeroWeightSum);
    }

    let mut acc = vec![0.0f64; pixel_count];
    for (frame, &w) in frames.iter().zip(weights.iter()) {
        let norm_w = w / weight_sum;
        for (acc_px, &px) in acc.iter_mut().zip(frame.iter()).take(pixel_count) {
            *acc_px += px as f64 * norm_w;
        }
    }

    Ok(acc
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect())
}

/// Compute an exposure-style long-exposure frame by accumulating `n` frames
/// from an iterator and blending them with equal weights.
///
/// This mirrors a physical long-exposure photograph where the shutter is open
/// for an extended period.
///
/// # Errors
///
/// Returns [`FrameBlendError`] if validation fails or the iterator is empty.
pub fn long_exposure_blend<'a>(
    frames: impl Iterator<Item = &'a [u8]>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, FrameBlendError> {
    let collected: Vec<&'a [u8]> = frames.collect();
    if collected.is_empty() {
        return Err(FrameBlendError::NoFrames);
    }
    blend_equal(&collected, width, height)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate that dimensions are non-zero.
fn validate_dimensions(width: u32, height: u32) -> Result<(), FrameBlendError> {
    if width == 0 || height == 0 {
        return Err(FrameBlendError::DegenerateDimensions { width, height });
    }
    Ok(())
}

/// Build weights according to [`BlendMode`].
fn build_weights(mode: &BlendMode, n: usize) -> Result<Vec<f64>, FrameBlendError> {
    match mode {
        BlendMode::Equal => Ok(vec![1.0f64; n]),
        BlendMode::Linear => Ok((1..=n).map(|i| i as f64).collect()),
        BlendMode::Gaussian { sigma_factor } => Ok(gaussian_weights(n, *sigma_factor)),
        BlendMode::Custom(w) => {
            if w.len() != n {
                Err(FrameBlendError::WeightMismatch {
                    weights: w.len(),
                    frames: n,
                })
            } else {
                let sum: f64 = w.iter().sum();
                if sum.abs() < f64::EPSILON {
                    Err(FrameBlendError::ZeroWeightSum)
                } else {
                    Ok(w.clone())
                }
            }
        }
    }
}

/// Generate Gaussian weights for `n` samples.
///
/// The Gaussian is centred on `(n - 1) / 2.0` with sigma = `sigma_factor *
/// (n - 1) / 2.0` (i.e. sigma covers `sigma_factor` half-widths).
fn gaussian_weights(n: usize, sigma_factor: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let centre = (n - 1) as f64 / 2.0;
    let sigma = sigma_factor * centre;
    let sigma_sq = if sigma < f64::EPSILON {
        f64::EPSILON
    } else {
        sigma * sigma
    };

    (0..n)
        .map(|i| {
            let d = i as f64 - centre;
            (-(d * d) / (2.0 * sigma_sq)).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w * h) as usize]
    }

    fn ramp(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    // 1. blend_equal: single frame → identical output
    #[test]
    fn test_blend_equal_single_frame_passthrough() {
        let frame = ramp(8, 8);
        let out = blend_equal(&[frame.as_slice()], 8, 8).unwrap();
        assert_eq!(out, frame);
    }

    // 2. blend_equal: two identical frames → same frame
    #[test]
    fn test_blend_equal_two_identical_frames() {
        let frame = flat(8, 8, 100);
        let out = blend_equal(&[frame.as_slice(), frame.as_slice()], 8, 8).unwrap();
        assert_eq!(out, frame);
    }

    // 3. blend_equal: 0 and 200 → 100
    #[test]
    fn test_blend_equal_average_two() {
        let a = flat(4, 4, 0);
        let b = flat(4, 4, 200);
        let out = blend_equal(&[a.as_slice(), b.as_slice()], 4, 4).unwrap();
        for px in out {
            assert_eq!(px, 100u8);
        }
    }

    // 4. blend_equal: no frames → error
    #[test]
    fn test_blend_equal_no_frames_error() {
        let err = blend_equal(&[], 8, 8);
        assert!(matches!(err, Err(FrameBlendError::NoFrames)));
    }

    // 5. blend_linear: later frames have more weight
    #[test]
    fn test_blend_linear_weights_increase() {
        // 3 frames: 0, 100, 200. Weights 1, 2, 3. Sum = 6.
        // Expected = (0*1 + 100*2 + 200*3) / 6 = 800/6 ≈ 133
        let a = flat(4, 4, 0);
        let b = flat(4, 4, 100);
        let c = flat(4, 4, 200);
        let out = blend_linear(&[a.as_slice(), b.as_slice(), c.as_slice()], 4, 4).unwrap();
        for px in &out {
            let expected = ((0u32 + 200 + 600) / 6) as u8; // 133
            assert!(
                (*px as i32 - expected as i32).abs() <= 1,
                "linear blend expected ≈{expected}, got {px}"
            );
        }
    }

    // 6. blend_gaussian: output has same size as input frame
    #[test]
    fn test_blend_gaussian_output_size() {
        let w = 8u32;
        let h = 8u32;
        let frames: Vec<Vec<u8>> = (0..5).map(|_| flat(w, h, 128)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let out = blend_gaussian(&refs, w, h, 1.0).unwrap();
        assert_eq!(out.len(), (w * h) as usize);
    }

    // 7. blend_frames_weighted: weight mismatch → error
    #[test]
    fn test_blend_weighted_weight_mismatch() {
        let frame = flat(4, 4, 100);
        let err = blend_frames_weighted(&[frame.as_slice()], &[1.0, 1.0], 4, 4);
        assert!(matches!(err, Err(FrameBlendError::WeightMismatch { .. })));
    }

    // 8. blend_frames_weighted: zero weight sum → error
    #[test]
    fn test_blend_weighted_zero_weight_sum() {
        let frame = flat(4, 4, 100);
        let err = blend_frames_weighted(&[frame.as_slice()], &[0.0], 4, 4);
        assert!(matches!(err, Err(FrameBlendError::ZeroWeightSum)));
    }

    // 9. FrameBlender: push and blend
    #[test]
    fn test_frame_blender_push_and_blend() {
        let mut blender = FrameBlender::new(BlendConfig::new_equal(3));
        blender.push(flat(8, 8, 100), 8, 8).unwrap();
        blender.push(flat(8, 8, 200), 8, 8).unwrap();
        let out = blender.blend().unwrap().unwrap();
        for px in &out {
            assert_eq!(*px, 150u8, "expected average of 100 and 200");
        }
    }

    // 10. FrameBlender: window slides (oldest frame dropped)
    #[test]
    fn test_frame_blender_window_slides() {
        let mut blender = FrameBlender::new(BlendConfig::new_equal(2));
        blender.push(flat(4, 4, 0), 4, 4).unwrap();
        blender.push(flat(4, 4, 100), 4, 4).unwrap();
        blender.push(flat(4, 4, 200), 4, 4).unwrap();
        // Only last 2 frames retained: 100 and 200 → average 150
        assert_eq!(blender.len(), 2);
        let out = blender.blend().unwrap().unwrap();
        for px in &out {
            assert_eq!(*px, 150u8);
        }
    }

    // 11. FrameBlender: empty blender returns None
    #[test]
    fn test_frame_blender_empty_returns_none() {
        let blender = FrameBlender::new(BlendConfig::new_equal(4));
        let result = blender.blend().unwrap();
        assert!(result.is_none());
    }

    // 12. long_exposure_blend: averages correctly
    #[test]
    fn test_long_exposure_blend() {
        let a = flat(4, 4, 0u8);
        let b = flat(4, 4, 100u8);
        let c = flat(4, 4, 200u8);
        let frames: Vec<&[u8]> = vec![a.as_slice(), b.as_slice(), c.as_slice()];
        let out = long_exposure_blend(frames.into_iter(), 4, 4).unwrap();
        // (0 + 100 + 200) / 3 = 100
        for px in &out {
            assert_eq!(*px, 100u8);
        }
    }

    // 13. Degenerate dimensions → error
    #[test]
    fn test_degenerate_dimensions_error() {
        let err = blend_equal(&[&[0u8; 0]], 0, 8);
        assert!(matches!(
            err,
            Err(FrameBlendError::DegenerateDimensions { .. })
        ));
    }

    // 14. BlendMode::Custom: correct weights applied
    #[test]
    fn test_custom_mode_weights() {
        let mut blender = FrameBlender::new(BlendConfig {
            window_size: 2,
            mode: BlendMode::Custom(vec![1.0, 3.0]),
        });
        blender.push(flat(4, 4, 0), 4, 4).unwrap();
        blender.push(flat(4, 4, 100), 4, 4).unwrap();
        // weights 1, 3 → (0*1 + 100*3) / 4 = 75
        let out = blender.blend().unwrap().unwrap();
        for px in &out {
            assert_eq!(*px, 75u8, "custom-weight blend expected 75, got {px}");
        }
    }

    // 15. Gaussian weights sum to > 0 and are symmetric
    #[test]
    fn test_gaussian_weights_symmetric() {
        let w = gaussian_weights(5, 1.0);
        assert_eq!(w.len(), 5);
        let sum: f64 = w.iter().sum();
        assert!(sum > 0.0, "gaussian weights should sum to positive value");
        // Check symmetry around centre
        for i in 0..2 {
            assert!(
                (w[i] - w[4 - i]).abs() < 1e-10,
                "gaussian weights should be symmetric: w[{i}]={} != w[{}]={}",
                w[i],
                4 - i,
                w[4 - i]
            );
        }
    }
}
