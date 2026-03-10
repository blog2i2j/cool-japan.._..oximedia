//! Temporal quality metrics for video sequences.
//!
//! This module provides metrics for analyzing temporal artifacts and
//! consistency across video frames.
//!
//! # Metrics
//!
//! - **TI** (Temporal Information): Measures temporal complexity
//! - **Temporal Coherence**: Measures frame-to-frame consistency
//! - **Flicker**: Detects flickering artifacts
//! - **Judder**: Detects motion judder
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::quality::temporal::calculate_temporal_info;
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! frame.allocate();
//!
//! let info = calculate_temporal_info(&frame)?;
//! println!("TI: {:.2}", info.ti);
//! ```

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

/// Temporal information metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalInfo {
    /// Temporal Information (TI) score.
    /// Measures motion and temporal complexity.
    pub ti: f64,

    /// Temporal perceptual information.
    pub tpi: f64,

    /// Frame difference standard deviation.
    pub std_dev: f64,

    /// Maximum temporal difference.
    pub max_diff: f64,

    /// Mean temporal difference.
    pub mean_diff: f64,
}

impl TemporalInfo {
    /// Create new temporal info with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ti: 0.0,
            tpi: 0.0,
            std_dev: 0.0,
            max_diff: 0.0,
            mean_diff: 0.0,
        }
    }

    /// Check if content has high temporal complexity.
    #[must_use]
    pub fn is_high_motion(&self) -> bool {
        self.ti > 50.0
    }

    /// Check if content is static or near-static.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.ti < 10.0
    }
}

impl Default for TemporalInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate temporal information for a single frame.
///
/// For a single frame, this estimates temporal activity by analyzing
/// spatial gradients. For true temporal metrics, use `calculate_temporal_metrics`.
///
/// # Errors
///
/// Returns an error if frame data is invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::temporal::calculate_temporal_info;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// frame.allocate();
///
/// let info = calculate_temporal_info(&frame)?;
/// assert!(info.ti >= 0.0);
/// ```
pub fn calculate_temporal_info(frame: &VideoFrame) -> CvResult<TemporalInfo> {
    if frame.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let plane = &frame.planes[0];
    let width = frame.width as usize;
    let height = frame.height as usize;
    let stride = plane.stride;

    // Calculate temporal differences (approximate using spatial differences)
    let mut differences = Vec::new();
    let mut sum_diff = 0.0;
    let mut max_diff: f64 = 0.0;

    for y in 1..height {
        for x in 0..width {
            let idx = y * stride + x;
            let prev_idx = (y - 1) * stride + x;

            if idx < plane.data.len() && prev_idx < plane.data.len() {
                let diff = (i32::from(plane.data[idx]) - i32::from(plane.data[prev_idx])).abs();
                let diff_f = diff as f64;

                differences.push(diff_f);
                sum_diff += diff_f;
                max_diff = max_diff.max(diff_f);
            }
        }
    }

    let count = differences.len() as f64;
    if count == 0.0 {
        return Ok(TemporalInfo::new());
    }

    let mean_diff = sum_diff / count;

    // Calculate standard deviation
    let variance = differences
        .iter()
        .map(|&d| (d - mean_diff) * (d - mean_diff))
        .sum::<f64>()
        / count;
    let std_dev = variance.sqrt();

    // TI is the standard deviation of temporal differences
    let ti = std_dev;

    // Temporal Perceptual Information (weighted by spatial activity)
    let spatial_activity = calculate_spatial_activity(frame)?;
    let tpi = ti * (1.0 + spatial_activity / 100.0);

    Ok(TemporalInfo {
        ti,
        tpi,
        std_dev,
        max_diff,
        mean_diff,
    })
}

/// Calculate temporal metrics between two consecutive frames.
///
/// Provides detailed temporal quality analysis including coherence and artifacts.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::temporal::calculate_temporal_metrics;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut frame1 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// frame1.allocate();
/// let mut frame2 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// frame2.allocate();
///
/// let metrics = calculate_temporal_metrics(&frame1, &frame2)?;
/// assert!(metrics.coherence >= 0.0 && metrics.coherence <= 1.0);
/// ```
pub fn calculate_temporal_metrics(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
) -> CvResult<TemporalMetrics> {
    validate_frames(frame1, frame2)?;

    let plane1 = &frame1.planes[0];
    let plane2 = &frame2.planes[0];
    let width = frame1.width as usize;
    let height = frame1.height as usize;
    let stride = plane1.stride;

    // Calculate frame difference statistics
    let mut differences = Vec::new();
    let mut sum_diff = 0.0;
    let mut sum_abs_diff = 0.0;
    let mut max_diff: f64 = 0.0;

    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x;

            if idx < plane1.data.len() && idx < plane2.data.len() {
                let diff = i32::from(plane1.data[idx]) - i32::from(plane2.data[idx]);
                let abs_diff = diff.abs() as f64;

                differences.push(diff as f64);
                sum_diff += diff as f64;
                sum_abs_diff += abs_diff;
                max_diff = max_diff.max(abs_diff);
            }
        }
    }

    let count = differences.len() as f64;
    if count == 0.0 {
        return Ok(TemporalMetrics::new());
    }

    let mean_diff = sum_diff / count;
    let mean_abs_diff = sum_abs_diff / count;

    // Calculate standard deviation
    let variance = differences
        .iter()
        .map(|&d| (d - mean_diff) * (d - mean_diff))
        .sum::<f64>()
        / count;
    let std_dev = variance.sqrt();

    // Temporal coherence (inverse of difference magnitude)
    let coherence = 1.0 - (mean_abs_diff / 255.0).clamp(0.0, 1.0);

    // Detect flicker (high variance with low mean difference)
    let flicker = if mean_abs_diff < 10.0 && std_dev > 5.0 {
        (std_dev / 50.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Motion magnitude
    let motion = (mean_abs_diff / 128.0).clamp(0.0, 1.0);

    // Judder detection (irregular motion patterns)
    let judder = detect_judder(&differences)?;

    Ok(TemporalMetrics {
        coherence,
        flicker,
        motion,
        judder,
        mean_diff,
        std_dev,
        max_diff,
    })
}

/// Temporal quality metrics between frames.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalMetrics {
    /// Temporal coherence (0-1, higher is better).
    pub coherence: f64,

    /// Flicker artifact score (0-1, lower is better).
    pub flicker: f64,

    /// Motion magnitude (0-1).
    pub motion: f64,

    /// Judder artifact score (0-1, lower is better).
    pub judder: f64,

    /// Mean temporal difference.
    pub mean_diff: f64,

    /// Standard deviation of differences.
    pub std_dev: f64,

    /// Maximum difference.
    pub max_diff: f64,
}

impl TemporalMetrics {
    /// Create new temporal metrics with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            coherence: 1.0,
            flicker: 0.0,
            motion: 0.0,
            judder: 0.0,
            mean_diff: 0.0,
            std_dev: 0.0,
            max_diff: 0.0,
        }
    }

    /// Check if temporal quality is good.
    #[must_use]
    pub fn is_good_quality(&self) -> bool {
        self.coherence > 0.9 && self.flicker < 0.1 && self.judder < 0.1
    }

    /// Check if artifacts are present.
    #[must_use]
    pub fn has_artifacts(&self) -> bool {
        self.flicker > 0.3 || self.judder > 0.3
    }
}

impl Default for TemporalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect judder artifacts in temporal differences.
fn detect_judder(differences: &[f64]) -> CvResult<f64> {
    if differences.len() < 3 {
        return Ok(0.0);
    }

    // Judder is detected by analyzing regularity of motion
    // Calculate second derivative (acceleration)
    let mut accelerations = Vec::new();

    for i in 1..differences.len() - 1 {
        let accel = differences[i + 1] - 2.0 * differences[i] + differences[i - 1];
        accelerations.push(accel.abs());
    }

    // High variance in acceleration indicates judder
    let mean_accel = accelerations.iter().sum::<f64>() / accelerations.len() as f64;
    let variance = accelerations
        .iter()
        .map(|&a| (a - mean_accel) * (a - mean_accel))
        .sum::<f64>()
        / accelerations.len() as f64;

    let judder = (variance.sqrt() / 100.0).clamp(0.0, 1.0);

    Ok(judder)
}

/// Calculate spatial activity in a frame.
fn calculate_spatial_activity(frame: &VideoFrame) -> CvResult<f64> {
    if frame.planes.is_empty() {
        return Ok(0.0);
    }

    let plane = &frame.planes[0];
    let width = frame.width as usize;
    let height = frame.height as usize;
    let stride = plane.stride;

    let mut gradient_sum = 0.0;
    let mut count = 0;

    // Calculate Sobel gradients
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let grad = calculate_gradient_magnitude(&plane.data, x, y, stride);
            gradient_sum += grad;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(0.0);
    }

    Ok(gradient_sum / count as f64)
}

/// Calculate gradient magnitude at a pixel.
fn calculate_gradient_magnitude(data: &[u8], x: usize, y: usize, stride: usize) -> f64 {
    let get_pixel = |dx: isize, dy: isize| -> f64 {
        let px = (x as isize + dx) as usize;
        let py = (y as isize + dy) as usize;
        let idx = py * stride + px;
        if idx < data.len() {
            f64::from(data[idx])
        } else {
            0.0
        }
    };

    // Sobel operator
    let gx = -get_pixel(-1, -1)
        + 1.0 * get_pixel(1, -1)
        + -2.0 * get_pixel(-1, 0)
        + 2.0 * get_pixel(1, 0)
        + -get_pixel(-1, 1)
        + 1.0 * get_pixel(1, 1);

    let gy = -get_pixel(-1, -1)
        + -2.0 * get_pixel(0, -1)
        + -get_pixel(1, -1)
        + 1.0 * get_pixel(-1, 1)
        + 2.0 * get_pixel(0, 1)
        + 1.0 * get_pixel(1, 1);

    (gx * gx + gy * gy).sqrt()
}

/// Calculate temporal stability for a sequence of frames.
///
/// Analyzes temporal consistency across multiple frames.
///
/// # Errors
///
/// Returns an error if frames are incompatible or sequence is too short.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::temporal::calculate_temporal_stability;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut frames = vec![];
/// for _ in 0..10 {
///     let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
///     frame.allocate();
///     frames.push(frame);
/// }
///
/// let stability = calculate_temporal_stability(&frames)?;
/// assert!(stability >= 0.0 && stability <= 1.0);
/// ```
pub fn calculate_temporal_stability(frames: &[VideoFrame]) -> CvResult<f64> {
    if frames.len() < 2 {
        return Err(CvError::invalid_parameter(
            "frame_count",
            "Need at least 2 frames for temporal stability",
        ));
    }

    let mut coherence_scores = Vec::new();

    // Calculate coherence between consecutive frames
    for i in 0..frames.len() - 1 {
        let metrics = calculate_temporal_metrics(&frames[i], &frames[i + 1])?;
        coherence_scores.push(metrics.coherence);
    }

    // Stability is the average coherence
    let mean_coherence = coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64;

    // Also consider variance (lower variance = more stable)
    let variance = coherence_scores
        .iter()
        .map(|&c| (c - mean_coherence) * (c - mean_coherence))
        .sum::<f64>()
        / coherence_scores.len() as f64;

    let stability = mean_coherence * (1.0 - variance).max(0.0);

    Ok(stability.clamp(0.0, 1.0))
}

/// Detect scene changes in a frame sequence.
///
/// Returns indices of frames where scene changes occur.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::temporal::detect_scene_changes;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut frames = vec![];
/// for _ in 0..20 {
///     let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
///     frame.allocate();
///     frames.push(frame);
/// }
///
/// let scene_changes = detect_scene_changes(&frames, 0.3)?;
/// // scene_changes contains indices where scenes change
/// ```
pub fn detect_scene_changes(frames: &[VideoFrame], threshold: f64) -> CvResult<Vec<usize>> {
    if frames.len() < 2 {
        return Ok(Vec::new());
    }

    let mut scene_changes = Vec::new();

    for i in 0..frames.len() - 1 {
        let metrics = calculate_temporal_metrics(&frames[i], &frames[i + 1])?;

        // Scene change detected if coherence is low (large difference)
        if metrics.coherence < threshold {
            scene_changes.push(i + 1);
        }
    }

    Ok(scene_changes)
}

/// Calculate flicker severity across a sequence.
///
/// Returns average flicker score and maximum flicker.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
pub fn analyze_flicker_sequence(frames: &[VideoFrame]) -> CvResult<(f64, f64)> {
    if frames.len() < 3 {
        return Ok((0.0, 0.0));
    }

    let mut flicker_scores = Vec::new();

    for i in 0..frames.len() - 1 {
        let metrics = calculate_temporal_metrics(&frames[i], &frames[i + 1])?;
        flicker_scores.push(metrics.flicker);
    }

    let avg_flicker = flicker_scores.iter().sum::<f64>() / flicker_scores.len() as f64;
    let max_flicker = flicker_scores.iter().copied().fold(0.0_f64, f64::max);

    Ok((avg_flicker, max_flicker))
}

/// Validate frames for temporal analysis.
fn validate_frames(frame1: &VideoFrame, frame2: &VideoFrame) -> CvResult<()> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "dimensions",
            format!(
                "{}x{} vs {}x{}",
                frame1.width, frame1.height, frame2.width, frame2.height
            ),
        ));
    }

    if frame1.format != frame2.format {
        return Err(CvError::invalid_parameter(
            "pixel_format",
            "Frames must have the same pixel format",
        ));
    }

    if frame1.planes.is_empty() || frame2.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    Ok(())
}

/// Calculate motion vectors between frames (simplified).
///
/// Returns motion magnitude map.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
pub fn calculate_motion_vectors(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
) -> CvResult<Vec<Vec<f64>>> {
    validate_frames(frame1, frame2)?;

    let plane1 = &frame1.planes[0];
    let plane2 = &frame2.planes[0];
    let width = frame1.width as usize;
    let height = frame1.height as usize;

    const BLOCK_SIZE: usize = 16;
    let blocks_x = width / BLOCK_SIZE;
    let blocks_y = height / BLOCK_SIZE;

    let mut motion_map = vec![vec![0.0; blocks_x]; blocks_y];

    // Calculate SAD (Sum of Absolute Differences) for each block
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let sad = calculate_block_sad(
                &plane1.data,
                &plane2.data,
                bx * BLOCK_SIZE,
                by * BLOCK_SIZE,
                BLOCK_SIZE,
                plane1.stride,
            );

            // Normalize SAD to motion magnitude
            motion_map[by][bx] = (sad / (BLOCK_SIZE * BLOCK_SIZE) as f64).min(255.0);
        }
    }

    Ok(motion_map)
}

/// Calculate Sum of Absolute Differences for a block.
fn calculate_block_sad(
    data1: &[u8],
    data2: &[u8],
    x: usize,
    y: usize,
    block_size: usize,
    stride: usize,
) -> f64 {
    let mut sad = 0.0;

    for dy in 0..block_size {
        for dx in 0..block_size {
            let idx = (y + dy) * stride + (x + dx);
            if idx < data1.len() && idx < data2.len() {
                let diff = (i32::from(data1[idx]) - i32::from(data2[idx])).abs();
                sad += diff as f64;
            }
        }
    }

    sad
}
