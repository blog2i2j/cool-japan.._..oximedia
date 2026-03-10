//! Pattern analysis, field metrics, cadence detection, and motion compensation
//! for the IVTC filter.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use std::collections::VecDeque;

use oximedia_codec::{Plane, VideoFrame};

use super::{DetectionSensitivity, IvtcConfig, PostProcessMode, TelecinePattern};

/// Pattern analyzer for offline analysis.
///
/// This analyzer can be used to analyze video content and determine
/// the best IVTC settings without processing the entire video.
#[derive(Debug)]
pub struct PatternAnalyzer {
    /// Frames analyzed.
    frames_analyzed: usize,
    /// Detected patterns and their confidence.
    detected_patterns: Vec<(TelecinePattern, f64)>,
    /// Average comb score.
    avg_comb_score: f64,
    /// Match quality history.
    match_history: Vec<f64>,
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames_analyzed: 0,
            detected_patterns: Vec::new(),
            avg_comb_score: 0.0,
            match_history: Vec::new(),
        }
    }

    /// Analyze a frame.
    pub fn analyze_frame(&mut self, frame: &VideoFrame) {
        self.frames_analyzed += 1;

        // Calculate comb metric
        let comb_score = self.calculate_comb_metric(frame);
        self.avg_comb_score = (self.avg_comb_score * (self.frames_analyzed - 1) as f64
            + comb_score)
            / self.frames_analyzed as f64;

        self.match_history.push(comb_score);
    }

    /// Calculate combing metric for a frame.
    fn calculate_comb_metric(&self, frame: &VideoFrame) -> f64 {
        if frame.planes.is_empty() {
            return 0.0;
        }

        let plane = &frame.planes[0];
        let height = frame.height as usize;
        let width = frame.width as usize;

        let mut comb_score = 0u64;
        let mut samples = 0u64;

        for y in 1..height - 1 {
            let row_prev = plane.row(y - 1);
            let row_curr = plane.row(y);
            let row_next = plane.row(y + 1);

            for x in 0..width {
                let prev = row_prev.get(x).copied().unwrap_or(0) as i32;
                let curr = row_curr.get(x).copied().unwrap_or(0) as i32;
                let next = row_next.get(x).copied().unwrap_or(0) as i32;

                let interp = (prev + next) / 2;
                let diff = (curr - interp).abs();

                if diff > 15 {
                    comb_score += diff as u64;
                }
                samples += 1;
            }
        }

        if samples > 0 {
            comb_score as f64 / samples as f64
        } else {
            0.0
        }
    }

    /// Get analysis results.
    #[must_use]
    pub fn get_results(&self) -> PatternAnalysisResults {
        PatternAnalysisResults {
            frames_analyzed: self.frames_analyzed,
            avg_comb_score: self.avg_comb_score,
            recommended_pattern: self.recommend_pattern(),
            confidence: self.calculate_confidence(),
        }
    }

    /// Recommend a pattern based on analysis.
    fn recommend_pattern(&self) -> TelecinePattern {
        if self.match_history.len() < 10 {
            return TelecinePattern::Auto;
        }

        // Simple heuristic based on comb score variation
        let variance = self.calculate_variance();

        if variance > 15.0 {
            TelecinePattern::Pattern32
        } else if variance > 5.0 {
            TelecinePattern::Pattern22
        } else {
            TelecinePattern::Auto
        }
    }

    /// Calculate variance in match history.
    fn calculate_variance(&self) -> f64 {
        if self.match_history.is_empty() {
            return 0.0;
        }

        let mean = self.match_history.iter().sum::<f64>() / self.match_history.len() as f64;
        let variance: f64 = self
            .match_history
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.match_history.len() as f64;

        variance.sqrt()
    }

    /// Calculate detection confidence.
    fn calculate_confidence(&self) -> f64 {
        if self.frames_analyzed < 30 {
            return 0.0;
        }

        let variance = self.calculate_variance();
        (variance / 20.0).min(1.0)
    }
}

/// Results from pattern analysis.
#[derive(Clone, Debug)]
pub struct PatternAnalysisResults {
    /// Number of frames analyzed.
    pub frames_analyzed: usize,
    /// Average combing score.
    pub avg_comb_score: f64,
    /// Recommended telecine pattern.
    pub recommended_pattern: TelecinePattern,
    /// Detection confidence (0.0-1.0).
    pub confidence: f64,
}

impl PatternAnalysisResults {
    /// Check if content likely has telecine.
    #[must_use]
    pub fn has_telecine(&self) -> bool {
        self.avg_comb_score > 5.0 && self.confidence > 0.5
    }

    /// Get recommended configuration.
    #[must_use]
    pub fn recommended_config(&self) -> IvtcConfig {
        IvtcConfig::new()
            .with_pattern(self.recommended_pattern)
            .with_post_processing(if self.avg_comb_score > 10.0 {
                PostProcessMode::Medium
            } else {
                PostProcessMode::Light
            })
            .with_sensitivity(if self.confidence > 0.8 {
                DetectionSensitivity::Low
            } else if self.confidence > 0.5 {
                DetectionSensitivity::Medium
            } else {
                DetectionSensitivity::High
            })
    }
}

/// Field metrics for detailed analysis.
#[derive(Clone, Debug, Default)]
pub struct FieldMetrics {
    /// Top field comb score.
    pub top_field_comb: f64,
    /// Bottom field comb score.
    pub bottom_field_comb: f64,
    /// Interlaced score (higher = more interlaced).
    pub interlace_score: f64,
    /// Progressive score (higher = more progressive).
    pub progressive_score: f64,
    /// Motion score (amount of motion detected).
    pub motion_score: f64,
    /// Spatial complexity.
    pub spatial_complexity: f64,
}

impl FieldMetrics {
    /// Calculate comprehensive field metrics for a frame.
    #[must_use]
    pub fn calculate(frame: &VideoFrame) -> Self {
        let mut metrics = Self::default();

        if frame.planes.is_empty() {
            return metrics;
        }

        let plane = &frame.planes[0];
        let height = frame.height as usize;
        let width = frame.width as usize;

        let mut top_comb = 0u64;
        let mut bottom_comb = 0u64;
        let mut interlace = 0u64;
        let mut progressive = 0u64;
        let mut spatial = 0u64;
        let mut samples = 0u64;

        for y in 2..height - 2 {
            let row_m2 = plane.row(y - 2);
            let row_m1 = plane.row(y - 1);
            let row_0 = plane.row(y);
            let row_p1 = plane.row(y + 1);
            let row_p2 = plane.row(y + 2);

            for x in 0..width {
                let m2 = row_m2.get(x).copied().unwrap_or(0) as i32;
                let m1 = row_m1.get(x).copied().unwrap_or(0) as i32;
                let c = row_0.get(x).copied().unwrap_or(0) as i32;
                let p1 = row_p1.get(x).copied().unwrap_or(0) as i32;
                let p2 = row_p2.get(x).copied().unwrap_or(0) as i32;

                // Field combing
                let interp = (m1 + p1) / 2;
                let comb = (c - interp).abs();

                if y % 2 == 0 {
                    top_comb += comb as u64;
                } else {
                    bottom_comb += comb as u64;
                }

                // Interlace vs progressive detection
                let field_diff = (m1 - p1).abs();
                let same_field_diff = (m2 - c).abs() + (c - p2).abs();

                if field_diff > same_field_diff + 10 {
                    interlace += field_diff as u64;
                } else {
                    progressive += same_field_diff as u64;
                }

                // Spatial complexity
                let grad_h =
                    (c - row_0.get(x.saturating_sub(1)).copied().unwrap_or(0) as i32).abs();
                let grad_v = (c - m1).abs();
                spatial += (grad_h + grad_v) as u64;

                samples += 1;
            }
        }

        if samples > 0 {
            metrics.top_field_comb = top_comb as f64 / samples as f64;
            metrics.bottom_field_comb = bottom_comb as f64 / samples as f64;
            metrics.interlace_score = interlace as f64 / samples as f64;
            metrics.progressive_score = progressive as f64 / samples as f64;
            metrics.spatial_complexity = spatial as f64 / samples as f64;
        }

        metrics
    }

    /// Check if frame is likely interlaced.
    #[must_use]
    pub fn is_interlaced(&self) -> bool {
        self.interlace_score > self.progressive_score * 1.5
    }

    /// Get the preferred field (0 = top, 1 = bottom).
    #[must_use]
    pub fn preferred_field(&self) -> u8 {
        if self.top_field_comb < self.bottom_field_comb {
            0
        } else {
            1
        }
    }

    /// Get overall quality score (0.0-1.0, higher is better).
    #[must_use]
    pub fn quality_score(&self) -> f64 {
        let comb_score = 1.0 / (1.0 + (self.top_field_comb + self.bottom_field_comb) / 2.0);
        let prog_score =
            self.progressive_score / (self.progressive_score + self.interlace_score + 1.0);
        (comb_score * 0.6 + prog_score * 0.4).clamp(0.0, 1.0)
    }
}

/// Cadence detector for pattern locking.
#[derive(Debug)]
pub struct CadenceDetector {
    /// Pattern history.
    history: VecDeque<u8>,
    /// Current detected cadence.
    cadence: Option<Vec<u8>>,
    /// Confidence level.
    confidence: f64,
    /// Number of consecutive matches.
    match_count: usize,
}

impl Default for CadenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CadenceDetector {
    /// Create a new cadence detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(30),
            cadence: None,
            confidence: 0.0,
            match_count: 0,
        }
    }

    /// Add a frame match result.
    pub fn add_match(&mut self, is_duplicate: bool) {
        let value = if is_duplicate { 1 } else { 0 };
        self.history.push_back(value);

        if self.history.len() > 30 {
            self.history.pop_front();
        }

        self.detect_cadence();
    }

    /// Detect repeating cadence pattern.
    fn detect_cadence(&mut self) {
        if self.history.len() < 10 {
            return;
        }

        let history: Vec<u8> = self.history.iter().copied().collect();

        // Try different pattern lengths
        for pattern_len in 2..=10 {
            if let Some(pattern) = self.check_pattern(&history, pattern_len) {
                if self.cadence.as_ref() == Some(&pattern) {
                    self.match_count += 1;
                } else {
                    self.cadence = Some(pattern);
                    self.match_count = 1;
                }

                self.confidence = (self.match_count as f64 / 10.0).min(1.0);
                return;
            }
        }
    }

    /// Check if history matches a pattern of given length.
    fn check_pattern(&self, history: &[u8], pattern_len: usize) -> Option<Vec<u8>> {
        if history.len() < pattern_len * 2 {
            return None;
        }

        let pattern: Vec<u8> = history[..pattern_len].to_vec();
        let mut matches = 0;
        let mut total = 0;

        for i in 0..history.len() - pattern_len {
            let window = &history[i..i + pattern_len];
            if window == pattern.as_slice() {
                matches += 1;
            }
            total += 1;
        }

        if matches as f64 / total as f64 > 0.7 {
            Some(pattern)
        } else {
            None
        }
    }

    /// Get detected cadence.
    #[must_use]
    pub fn cadence(&self) -> Option<&[u8]> {
        self.cadence.as_deref()
    }

    /// Get confidence level.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Check if cadence is locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.confidence > 0.8 && self.match_count >= 5
    }

    /// Reset detector.
    pub fn reset(&mut self) {
        self.history.clear();
        self.cadence = None;
        self.confidence = 0.0;
        self.match_count = 0;
    }
}

/// Motion compensation for better field matching.
#[derive(Debug)]
pub struct MotionCompensation {
    /// Enable motion compensation.
    enabled: bool,
    /// Search range in pixels.
    search_range: i32,
}

impl Default for MotionCompensation {
    fn default() -> Self {
        Self::new()
    }
}

impl MotionCompensation {
    /// Create new motion compensation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: true,
            search_range: 8,
        }
    }

    /// Set search range.
    #[must_use]
    pub fn with_search_range(mut self, range: i32) -> Self {
        self.search_range = range;
        self
    }

    /// Calculate motion vector between two frames.
    #[must_use]
    pub fn calculate_motion_vector(
        &self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
        x: usize,
        y: usize,
    ) -> (i32, i32) {
        if !self.enabled || frame1.planes.is_empty() || frame2.planes.is_empty() {
            return (0, 0);
        }

        let plane1 = &frame1.planes[0];
        let plane2 = &frame2.planes[0];
        let height = frame1.height as usize;
        let width = frame1.width as usize;

        let block_size = 8;
        let mut best_mv = (0, 0);
        let mut best_sad = u64::MAX;

        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let mut sad = 0u64;
                let mut samples = 0;

                for by in 0..block_size {
                    for bx in 0..block_size {
                        let y1 = y + by;
                        let x1 = x + bx;
                        let y2 = (y as i32 + by as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        let x2 = (x as i32 + bx as i32 + dx).clamp(0, width as i32 - 1) as usize;

                        if y1 < height && x1 < width {
                            let p1 = plane1.row(y1).get(x1).copied().unwrap_or(0) as i32;
                            let p2 = plane2.row(y2).get(x2).copied().unwrap_or(0) as i32;
                            sad += (p1 - p2).unsigned_abs() as u64;
                            samples += 1;
                        }
                    }
                }

                if samples > 0 && sad < best_sad {
                    best_sad = sad;
                    best_mv = (dx, dy);
                }
            }
        }

        best_mv
    }

    /// Apply motion compensation to improve field matching.
    #[must_use]
    pub fn compensate_field(
        &self,
        source: &VideoFrame,
        _reference: &VideoFrame,
        motion_vector: (i32, i32),
    ) -> VideoFrame {
        if !self.enabled {
            return source.clone();
        }

        let mut output = VideoFrame::new(source.format, source.width, source.height);
        output.timestamp = source.timestamp;
        output.frame_type = source.frame_type;
        output.color_info = source.color_info;

        for (plane_idx, src_plane) in source.planes.iter().enumerate() {
            let (width, height) = source.plane_dimensions(plane_idx);
            let mut dst_data = vec![0u8; (width * height) as usize];

            let (mvx, mvy) = motion_vector;

            for y in 0..height as usize {
                for x in 0..width as usize {
                    let src_y = (y as i32 + mvy).clamp(0, height as i32 - 1) as usize;
                    let src_x = (x as i32 + mvx).clamp(0, width as i32 - 1) as usize;

                    let pixel = src_plane.row(src_y).get(src_x).copied().unwrap_or(0);
                    dst_data[y * width as usize + x] = pixel;
                }
            }

            output.planes.push(Plane::new(dst_data, width as usize));
        }

        output
    }
}
