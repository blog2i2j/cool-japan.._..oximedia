//! Temporal fingerprinting for video content.
//!
//! This module provides temporal analysis and fingerprinting for video sequences:
//!
//! - **Keyframe extraction**: Identifies representative frames from video
//! - **Shot boundary detection**: Detects scene changes and cuts
//! - **Temporal signatures**: Creates time-based feature vectors
//! - **Frame sampling**: Intelligent frame selection strategies
//! - **Segment hashing**: Hashes video segments for partial matching

use crate::error::{CvError, CvResult};
use rayon::prelude::*;

/// Frame sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniform sampling at fixed intervals.
    Uniform,
    /// Adaptive sampling based on content changes.
    Adaptive,
    /// Scene-based sampling (one frame per scene).
    SceneBased,
    /// Keyframe-only sampling.
    KeyframeOnly,
}

/// Shot boundary detection result.
#[derive(Debug, Clone)]
pub struct ShotBoundary {
    /// Frame index where shot starts.
    pub start_frame: usize,
    /// Frame index where shot ends.
    pub end_frame: usize,
    /// Confidence score (0.0-1.0).
    pub confidence: f64,
    /// Type of transition (cut, fade, dissolve).
    pub transition_type: TransitionType,
}

/// Type of shot transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Hard cut (instant change).
    Cut,
    /// Fade to/from black.
    Fade,
    /// Dissolve/crossfade between scenes.
    Dissolve,
}

/// Keyframe with metadata.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Frame index in original video.
    pub frame_index: usize,
    /// Frame image data (width, height, RGB).
    pub image: (u32, u32, Vec<u8>),
    /// Importance score (0.0-1.0).
    pub importance: f64,
}

/// Extracts keyframes from video frames.
///
/// # Arguments
///
/// * `frames` - Slice of video frames (width, height, RGB data)
/// * `min_interval` - Minimum frames between keyframes
/// * `max_keyframes` - Maximum number of keyframes to extract
/// * `scene_threshold` - Scene change threshold (0.0-1.0)
///
/// # Errors
///
/// Returns an error if extraction fails.
pub fn extract_keyframes(
    frames: &[(u32, u32, Vec<u8>)],
    min_interval: usize,
    max_keyframes: usize,
    scene_threshold: f64,
) -> CvResult<Vec<(u32, u32, Vec<u8>)>> {
    if frames.is_empty() {
        return Err(CvError::invalid_parameter("frames", "empty"));
    }

    if min_interval == 0 {
        return Err(CvError::invalid_parameter("min_interval", "0"));
    }

    // Detect scene boundaries
    let boundaries = detect_shot_boundaries(frames, scene_threshold)?;

    // Select keyframes based on scene boundaries and importance
    let mut keyframes = Vec::new();
    let mut last_keyframe_idx = 0;

    // Add first frame
    keyframes.push(frames[0].clone());

    // Add one keyframe per scene
    for boundary in &boundaries {
        let scene_start = boundary.start_frame;
        let scene_end = boundary.end_frame;

        if scene_start < last_keyframe_idx + min_interval {
            continue;
        }

        // Find most representative frame in scene (middle frame)
        let mid_frame = (scene_start + scene_end) / 2;
        if mid_frame < frames.len() && mid_frame >= last_keyframe_idx + min_interval {
            keyframes.push(frames[mid_frame].clone());
            last_keyframe_idx = mid_frame;
        }

        if keyframes.len() >= max_keyframes {
            break;
        }
    }

    // If we don't have enough keyframes, add uniform samples
    if keyframes.len() < max_keyframes {
        let interval = frames.len() / max_keyframes.max(1);
        let mut idx = interval;
        while idx < frames.len() && keyframes.len() < max_keyframes {
            if idx >= last_keyframe_idx + min_interval {
                keyframes.push(frames[idx].clone());
                last_keyframe_idx = idx;
            }
            idx += interval;
        }
    }

    Ok(keyframes)
}

/// Detects shot boundaries in video frames.
///
/// # Arguments
///
/// * `frames` - Slice of video frames
/// * `threshold` - Scene change threshold (0.0-1.0)
///
/// # Errors
///
/// Returns an error if detection fails.
pub fn detect_shot_boundaries(
    frames: &[(u32, u32, Vec<u8>)],
    threshold: f64,
) -> CvResult<Vec<ShotBoundary>> {
    if frames.len() < 2 {
        return Ok(Vec::new());
    }

    let mut boundaries = Vec::new();
    let mut scene_start = 0;

    // Compute frame differences
    let differences = compute_frame_differences(frames)?;

    for (i, &diff) in differences.iter().enumerate() {
        if diff > threshold {
            // Detected a scene change
            if i > scene_start {
                boundaries.push(ShotBoundary {
                    start_frame: scene_start,
                    end_frame: i,
                    confidence: diff.min(1.0),
                    transition_type: classify_transition(diff),
                });
            }
            scene_start = i + 1;
        }
    }

    // Add final scene
    if scene_start < frames.len() {
        boundaries.push(ShotBoundary {
            start_frame: scene_start,
            end_frame: frames.len() - 1,
            confidence: 1.0,
            transition_type: TransitionType::Cut,
        });
    }

    Ok(boundaries)
}

/// Computes differences between consecutive frames.
fn compute_frame_differences(frames: &[(u32, u32, Vec<u8>)]) -> CvResult<Vec<f64>> {
    let mut differences = Vec::with_capacity(frames.len() - 1);

    for i in 0..frames.len() - 1 {
        let (w1, h1, data1) = &frames[i];
        let (w2, h2, data2) = &frames[i + 1];

        if w1 != w2 || h1 != h2 {
            return Err(CvError::invalid_parameter(
                "frames",
                "inconsistent dimensions",
            ));
        }

        let diff = compute_frame_difference(data1, data2);
        differences.push(diff);
    }

    Ok(differences)
}

/// Computes difference between two frames.
fn compute_frame_difference(frame1: &[u8], frame2: &[u8]) -> f64 {
    if frame1.len() != frame2.len() {
        return 1.0; // Maximum difference
    }

    let mut sum = 0.0;
    let n = frame1.len() / 3; // Number of pixels

    for i in 0..n {
        let r1 = f64::from(frame1[i * 3]);
        let g1 = f64::from(frame1[i * 3 + 1]);
        let b1 = f64::from(frame1[i * 3 + 2]);

        let r2 = f64::from(frame2[i * 3]);
        let g2 = f64::from(frame2[i * 3 + 1]);
        let b2 = f64::from(frame2[i * 3 + 2]);

        let dr = (r1 - r2).abs();
        let dg = (g1 - g2).abs();
        let db = (b1 - b2).abs();

        sum += dr + dg + db;
    }

    // Normalize to [0, 1]
    sum / (n as f64 * 3.0 * 255.0)
}

/// Classifies transition type based on difference magnitude.
fn classify_transition(diff: f64) -> TransitionType {
    if diff > 0.7 {
        TransitionType::Cut
    } else if diff > 0.4 {
        TransitionType::Dissolve
    } else {
        TransitionType::Fade
    }
}

/// Computes temporal signature for video.
///
/// Creates a time-based feature vector capturing temporal patterns.
///
/// # Arguments
///
/// * `frames` - Slice of video frames
/// * `interval` - Sampling interval in frames
///
/// # Errors
///
/// Returns an error if computation fails.
pub fn compute_temporal_signature(
    frames: &[(u32, u32, Vec<u8>)],
    interval: usize,
) -> CvResult<Vec<f32>> {
    if frames.is_empty() {
        return Err(CvError::invalid_parameter("frames", "empty"));
    }

    if interval == 0 {
        return Err(CvError::invalid_parameter("interval", "0"));
    }

    let mut signature = Vec::new();

    // Sample frames at regular intervals
    let mut idx = 0;
    while idx < frames.len() {
        let features = extract_temporal_features(&frames[idx])?;
        signature.extend_from_slice(&features);

        idx += interval;
    }

    Ok(signature)
}

/// Extracts temporal features from a single frame.
fn extract_temporal_features(frame: &(u32, u32, Vec<u8>)) -> CvResult<Vec<f32>> {
    let (width, height, data) = frame;

    if *width == 0 || *height == 0 {
        return Err(CvError::invalid_dimensions(*width, *height));
    }

    let mut features = Vec::with_capacity(8);

    // Average brightness
    let brightness = compute_average_brightness(data);
    features.push(brightness);

    // Color moments (RGB)
    let (r_mean, g_mean, b_mean) = compute_color_moments(data);
    features.push(r_mean);
    features.push(g_mean);
    features.push(b_mean);

    // Edge density
    let edge_density = compute_edge_density(data, *width, *height);
    features.push(edge_density);

    // Spatial complexity
    let complexity = compute_spatial_complexity(data, *width, *height);
    features.push(complexity);

    // Color variance
    let variance = compute_color_variance(data);
    features.push(variance);

    // Contrast
    let contrast = compute_contrast(data);
    features.push(contrast);

    Ok(features)
}

/// Computes average brightness of frame.
fn compute_average_brightness(data: &[u8]) -> f32 {
    let sum: u32 = data.iter().map(|&x| u32::from(x)).sum();
    sum as f32 / data.len() as f32 / 255.0
}

/// Computes color moments (mean values).
fn compute_color_moments(data: &[u8]) -> (f32, f32, f32) {
    let n = data.len() / 3;
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;

    for i in 0..n {
        r_sum += u32::from(data[i * 3]);
        g_sum += u32::from(data[i * 3 + 1]);
        b_sum += u32::from(data[i * 3 + 2]);
    }

    let r_mean = r_sum as f32 / n as f32 / 255.0;
    let g_mean = g_sum as f32 / n as f32 / 255.0;
    let b_mean = b_sum as f32 / n as f32 / 255.0;

    (r_mean, g_mean, b_mean)
}

/// Computes edge density using simple gradient.
fn compute_edge_density(data: &[u8], width: u32, height: u32) -> f32 {
    let mut edge_count = 0u32;
    let threshold = 30u8;

    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let idx = ((y * width + x) * 3) as usize;
            let right_idx = ((y * width + x + 1) * 3) as usize;
            let down_idx = (((y + 1) * width + x) * 3) as usize;

            // Horizontal gradient
            let h_grad = data[idx].abs_diff(data[right_idx]);
            // Vertical gradient
            let v_grad = data[idx].abs_diff(data[down_idx]);

            if h_grad > threshold || v_grad > threshold {
                edge_count += 1;
            }
        }
    }

    edge_count as f32 / ((width - 1) * (height - 1)) as f32
}

/// Computes spatial complexity (variance of gradients).
fn compute_spatial_complexity(data: &[u8], width: u32, height: u32) -> f32 {
    let mut gradients = Vec::new();

    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let idx = ((y * width + x) * 3) as usize;
            let right_idx = ((y * width + x + 1) * 3) as usize;

            let grad = data[idx].abs_diff(data[right_idx]);
            gradients.push(f32::from(grad));
        }
    }

    if gradients.is_empty() {
        return 0.0;
    }

    let mean: f32 = gradients.iter().sum::<f32>() / gradients.len() as f32;
    let variance: f32 =
        gradients.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gradients.len() as f32;

    variance.sqrt() / 255.0
}

/// Computes color variance.
fn compute_color_variance(data: &[u8]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mean: f32 = data.iter().map(|&x| f32::from(x)).sum::<f32>() / n as f32;
    let variance: f32 = data
        .iter()
        .map(|&x| (f32::from(x) - mean).powi(2))
        .sum::<f32>()
        / n as f32;

    variance.sqrt() / 255.0
}

/// Computes contrast (difference between max and min).
fn compute_contrast(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // data is non-empty (checked above), so min() and max() always return Some.
    let min_val = data.iter().copied().min().unwrap_or(0);
    let max_val = data.iter().copied().max().unwrap_or(0);

    f32::from(max_val - min_val) / 255.0
}

/// Samples frames using specified strategy.
///
/// # Errors
///
/// Returns an error if sampling fails.
pub fn sample_frames(
    frames: &[(u32, u32, Vec<u8>)],
    strategy: SamplingStrategy,
    count: usize,
    scene_threshold: f64,
) -> CvResult<Vec<(u32, u32, Vec<u8>)>> {
    if frames.is_empty() {
        return Err(CvError::invalid_parameter("frames", "empty"));
    }

    if count == 0 {
        return Err(CvError::invalid_parameter("count", "0"));
    }

    match strategy {
        SamplingStrategy::Uniform => sample_uniform(frames, count),
        SamplingStrategy::Adaptive => sample_adaptive(frames, count),
        SamplingStrategy::SceneBased => sample_scene_based(frames, count, scene_threshold),
        SamplingStrategy::KeyframeOnly => {
            let min_interval = frames.len() / count.max(1);
            extract_keyframes(frames, min_interval, count, scene_threshold)
        }
    }
}

/// Samples frames uniformly.
fn sample_uniform(
    frames: &[(u32, u32, Vec<u8>)],
    count: usize,
) -> CvResult<Vec<(u32, u32, Vec<u8>)>> {
    let interval = frames.len() / count.max(1);
    let mut samples = Vec::new();

    let mut idx = 0;
    while idx < frames.len() && samples.len() < count {
        samples.push(frames[idx].clone());
        idx += interval;
    }

    Ok(samples)
}

/// Samples frames adaptively based on content changes.
fn sample_adaptive(
    frames: &[(u32, u32, Vec<u8>)],
    count: usize,
) -> CvResult<Vec<(u32, u32, Vec<u8>)>> {
    if frames.len() <= count {
        return Ok(frames.to_vec());
    }

    // Compute frame differences
    let differences = compute_frame_differences(frames)?;

    // Find frames with highest differences (most interesting)
    let mut indexed_diffs: Vec<(usize, f64)> = differences
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d))
        .collect();

    indexed_diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top 'count' frames
    let mut selected_indices: Vec<usize> =
        indexed_diffs.iter().take(count).map(|(i, _)| *i).collect();

    selected_indices.sort_unstable();

    let samples = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();

    Ok(samples)
}

/// Samples frames based on scene boundaries.
fn sample_scene_based(
    frames: &[(u32, u32, Vec<u8>)],
    count: usize,
    scene_threshold: f64,
) -> CvResult<Vec<(u32, u32, Vec<u8>)>> {
    let boundaries = detect_shot_boundaries(frames, scene_threshold)?;

    if boundaries.is_empty() {
        return sample_uniform(frames, count);
    }

    let mut samples = Vec::new();
    let scenes_per_sample = (boundaries.len() as f64 / count as f64).ceil() as usize;

    for (i, boundary) in boundaries.iter().enumerate() {
        if i % scenes_per_sample == 0 && samples.len() < count {
            let mid = (boundary.start_frame + boundary.end_frame) / 2;
            if mid < frames.len() {
                samples.push(frames[mid].clone());
            }
        }
    }

    Ok(samples)
}

/// Computes temporal correlation between two videos.
///
/// Returns a correlation score in [0.0, 1.0].
#[must_use]
pub fn compute_temporal_correlation(sig1: &[f32], sig2: &[f32]) -> f64 {
    if sig1.is_empty() || sig2.is_empty() {
        return 0.0;
    }

    // Use shorter signature for comparison
    let len = sig1.len().min(sig2.len());

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..len {
        let diff = (sig1[i] - sig2[i]).abs();
        sum += f64::from(1.0 - diff.min(1.0));
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Hashes a video segment.
///
/// # Errors
///
/// Returns an error if hashing fails.
pub fn hash_segment(frames: &[(u32, u32, Vec<u8>)], start: usize, end: usize) -> CvResult<Vec<u8>> {
    if start >= end || end > frames.len() {
        return Err(CvError::invalid_parameter(
            "range",
            format!("[{start}, {end})"),
        ));
    }

    let segment = &frames[start..end];
    let signature = compute_temporal_signature(segment, 1)?;

    // Convert f32 signature to bytes
    let mut bytes = Vec::new();
    for &val in &signature {
        bytes.push((val * 255.0) as u8);
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, brightness: u8) -> (u32, u32, Vec<u8>) {
        let data = vec![brightness; (width * height * 3) as usize];
        (width, height, data)
    }

    fn create_test_frames(count: usize) -> Vec<(u32, u32, Vec<u8>)> {
        (0..count)
            .map(|i| create_test_frame(64, 64, (i * 10) as u8))
            .collect()
    }

    #[test]
    fn test_extract_keyframes() {
        let frames = create_test_frames(100);
        let keyframes =
            extract_keyframes(&frames, 10, 5, 0.3).expect("extract_keyframes should succeed");
        assert!(!keyframes.is_empty());
        assert!(keyframes.len() <= 5);
    }

    #[test]
    fn test_detect_shot_boundaries() {
        let mut frames = create_test_frames(50);
        // Create a scene change at frame 25
        for i in 25..50 {
            frames[i] = create_test_frame(64, 64, 200);
        }

        let boundaries =
            detect_shot_boundaries(&frames, 0.3).expect("detect_shot_boundaries should succeed");
        assert!(!boundaries.is_empty());
    }

    #[test]
    fn test_compute_temporal_signature() {
        let frames = create_test_frames(10);
        let signature = compute_temporal_signature(&frames, 2)
            .expect("compute_temporal_signature should succeed");
        assert!(!signature.is_empty());
    }

    #[test]
    fn test_temporal_features() {
        let frame = create_test_frame(64, 64, 128);
        let features =
            extract_temporal_features(&frame).expect("extract_temporal_features should succeed");
        assert_eq!(features.len(), 8);
    }

    #[test]
    fn test_frame_difference() {
        let frame1 = vec![100u8; 300];
        let frame2 = vec![100u8; 300];
        let diff = compute_frame_difference(&frame1, &frame2);
        assert_eq!(diff, 0.0);

        let frame3 = vec![200u8; 300];
        let diff2 = compute_frame_difference(&frame1, &frame3);
        assert!(diff2 > 0.0);
    }

    #[test]
    fn test_sample_uniform() {
        let frames = create_test_frames(100);
        let samples = sample_uniform(&frames, 10).expect("sample_uniform should succeed");
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_sample_adaptive() {
        let frames = create_test_frames(100);
        let samples = sample_adaptive(&frames, 10).expect("sample_adaptive should succeed");
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_sample_scene_based() {
        let frames = create_test_frames(100);
        let samples =
            sample_scene_based(&frames, 10, 0.3).expect("sample_scene_based should succeed");
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_temporal_correlation() {
        let sig1 = vec![0.1, 0.2, 0.3, 0.4];
        let sig2 = vec![0.1, 0.2, 0.3, 0.4];
        let corr = compute_temporal_correlation(&sig1, &sig2);
        assert!((corr - 1.0).abs() < 0.01);

        let sig3 = vec![0.9, 0.8, 0.7, 0.6];
        let corr2 = compute_temporal_correlation(&sig1, &sig3);
        assert!(corr2 < 1.0);
    }

    #[test]
    fn test_hash_segment() {
        let frames = create_test_frames(10);
        let hash = hash_segment(&frames, 0, 5).expect("hash_segment should succeed");
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_brightness() {
        let data = vec![128u8; 300];
        let brightness = compute_average_brightness(&data);
        assert!((brightness - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_moments() {
        let data = vec![100u8, 150u8, 200u8];
        let (r, g, b) = compute_color_moments(&data);
        assert!(r > 0.0 && r < 1.0);
        assert!(g > 0.0 && g < 1.0);
        assert!(b > 0.0 && b < 1.0);
    }

    #[test]
    fn test_transition_classification() {
        assert_eq!(classify_transition(0.8), TransitionType::Cut);
        assert_eq!(classify_transition(0.5), TransitionType::Dissolve);
        assert_eq!(classify_transition(0.2), TransitionType::Fade);
    }

    #[test]
    fn test_empty_frames() {
        let frames: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        assert!(extract_keyframes(&frames, 10, 5, 0.3).is_err());
    }

    #[test]
    fn test_invalid_interval() {
        let frames = create_test_frames(10);
        assert!(extract_keyframes(&frames, 0, 5, 0.3).is_err());
    }
}
